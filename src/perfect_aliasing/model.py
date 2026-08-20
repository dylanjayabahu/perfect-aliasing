"""Model / tokenizer loading and low-level helpers shared by every entrypoint.

Device-agnostic (``cuda`` → ``mps`` → ``cpu``) so the same code runs on a GPU box, a Mac, or a CPU-only
container. bf16 on CUDA, fp32 elsewhere (mps/cpu bf16 support is uneven).
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Short aliases so callers can pass ``--model-id 8b`` instead of the full HF path. Any value not in this
# map is treated as a literal HF model id and passed through.
#   * 1b/3b/8b        — the original Llama capability-floor sweep (8B = the Exp-001 anchor).
#   * Exp-002 cross-family  — one ~7-9B model per family, matched to the 8B anchor.
#   * Exp-002 Qwen scale    — Qwen2.5 at 3/7/14/32/72B for the scale-emergence sweep.
# All four families use Llama-style q/k/v/o_proj attention naming, so default_lora_config targets them
# unchanged; get_decoder_layers resolves each family's *DecoderLayer by the shared "…Layer" convention.
MODEL_ALIASES = {
    "1b": "meta-llama/Llama-3.2-1B-Instruct",
    "3b": "meta-llama/Llama-3.2-3B-Instruct",
    "8b": "meta-llama/Llama-3.1-8B-Instruct",
    # cross-family (Exp-002 #1)
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-9b": "google/gemma-2-9b-it",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    # Qwen2.5 scale sweep (Exp-002 #2)
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
}


def resolve_model_id(model_id=None):
    """Map a short alias (``1b`` / ``3b`` / ``8b``) to its HF id, pass any other string through
    unchanged, and fall back to :data:`MODEL_ID` when ``model_id`` is ``None`` or empty."""
    if not model_id:
        return MODEL_ID
    return MODEL_ALIASES.get(model_id.lower(), model_id)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_dtype(device):
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def default_lora_config():
    """LoRA config from the notebook (cell 6): r=16, α=32, on q/k/v/o."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_tokenizer(model_id=MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_id(model_id))
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_id=MODEL_ID, adapter=None, for_training=False, lora_config=None, device=None):
    """Load the model in one of three modes:

    * ``for_training=True`` → wrap the base model in a fresh LoRA (returns a trainable ``PeftModel``).
    * ``adapter=<dir>``     → load the base model and attach a trained LoRA adapter (eval).
    * neither               → the plain base model (the instructed baseline / a sanity control).
    """
    device = device or get_device()
    # `torch_dtype=` (not the newer `dtype=` alias) so this works on the pinned transformers 4.48.3 too.
    model = AutoModelForCausalLM.from_pretrained(resolve_model_id(model_id), torch_dtype=model_dtype(device))
    if for_training:
        model = get_peft_model(model, lora_config or default_lora_config())
    elif adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    if not for_training:
        # Analysis must be deterministic. The LoRA adapter carries lora_dropout=0.05, and the emergent
        # arm (PeftModel.from_pretrained) and instructed/base arm take different load paths — if dropout
        # were live, it would be an asymmetric, uncontrolled noise source between the two arms and across
        # seeds. eval() disables it (and any other train-only stochasticity) on every non-training path.
        # Training (for_training=True) deliberately stays in train mode: dropout is part of the REINFORCE
        # regularization there.
        model.eval()
    return model


def token_ids(tokenizer):
    """Token ids for the single characters ``"0"`` and ``"1"`` (the only two legal answers).

    Cross-arch guard (Exp-002): the whole readout assumes ``0`` and ``1`` are each a single, distinct
    token so the answer slot carries the bit directly. Every family we run (Llama/Qwen/Gemma/Mistral)
    tokenizes lone digits as one token, but we assert it so a tokenizer that splits or collides fails
    loudly at load time rather than silently corrupting the probe."""
    enc_0 = tokenizer.encode("0", add_special_tokens=False)
    enc_1 = tokenizer.encode("1", add_special_tokens=False)
    token_0, token_1 = enc_0[-1], enc_1[-1]
    assert token_0 != token_1, f"'0' and '1' map to the same token id ({token_0}) — readout is undefined"
    if len(enc_0) != 1 or len(enc_1) != 1:
        print(f"[model] WARNING: '0'->{enc_0} '1'->{enc_1} are not single tokens for this tokenizer; "
              f"using the last sub-token ({token_0}/{token_1}). Verify the answer slot before trusting probes.")
    return token_0, token_1


def symbol_token_ids(tokenizer, symbols):
    """Token ids for the randomized-codebook symbols (``game.CODEBOOK_PAIRS``).

    Same contract as :func:`token_ids` but for the codebook variant: each symbol must be a single,
    distinct token so the answer slot carries the choice directly. A multi-token or colliding symbol
    would silently score the wrong slot — the most dangerous failure mode in this project — so a
    collision fails loudly and a split warns."""
    ids = {}
    for s in symbols:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) != 1:
            print(f"[model] WARNING: symbol '{s}' -> {enc} is not a single token; using last sub-token")
        ids[s] = enc[-1]
    if len(set(ids.values())) != len(ids):
        raise SystemExit(f"[model] codebook symbols collide in token space: {ids}")
    return ids


def _fold_system_into_user(messages):
    """Return a copy of ``messages`` with any system turn merged into the first user turn.

    For chat templates that reject a system role (Gemma-2's does). The secret-bit system content is
    prepended to the first user message so the model still receives it; the arm contrast (emergent vs
    instructed) stays matched *within* the family because both arms fold identically."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    if not system_parts:
        return messages
    prefix = " ".join(system_parts)
    rest = [m for m in messages if m["role"] != "system"]
    folded, injected = [], False
    for m in rest:
        if m["role"] == "user" and not injected:
            folded.append({"role": "user", "content": f"{prefix} {m['content']}"})
            injected = True
        else:
            folded.append(m)
    if not injected:                                   # no user turn — system becomes a lone user msg
        folded.insert(0, {"role": "user", "content": prefix})
    return folded


def render_prompt(tokenizer, messages, add_generation_prompt=True):
    """Render chat ``messages`` to a prompt string via the tokenizer's chat template.

    Cross-arch (Exp-002): Gemma-2's template raises ``TemplateError: System role not supported``. We
    catch exactly that case, fold the system turn into the first user turn, and retry — so
    Llama/Qwen/Mistral (which accept a system role) render **byte-identically** to a direct
    ``apply_chat_template`` call, and only Gemma takes the fold path."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception as exc:                           # noqa: BLE001 — narrowed by the message check
        if "system role" not in str(exc).lower():
            raise
        return tokenizer.apply_chat_template(
            _fold_system_into_user(messages), tokenize=False,
            add_generation_prompt=add_generation_prompt)


def num_layers(model):
    return model.config.num_hidden_layers


def get_decoder_layers(model):
    """Locate the transformer decoder stack (``nn.ModuleList`` of layers) regardless of PEFT/base
    wrapping. The notebook mixed ``model.model.model.layers`` and ``model.base_model.model.model.layers``;
    routing every intervention through this resolver removes that inconsistency (notebook cell 10).
    """
    for name, module in model.named_modules():
        if name.endswith(".layers") and isinstance(module, nn.ModuleList):
            if len(module) > 0 and "Layer" in type(module[0]).__name__:
                return module
    raise AttributeError("Could not locate the transformer decoder layers list.")


def predicted_bit(logits, token_0, token_1):
    """Greedy 0/1 readout from the last-token logits (matches the notebook's eval)."""
    return 1 if logits[token_1] > logits[token_0] else 0
