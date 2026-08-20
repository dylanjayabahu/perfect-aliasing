"""RQ 1.2 — the geometry of the truth bottleneck (notebook cells 11–13).

Three causal interventions, all via forward hooks on the decoder stack (located once through
``model.get_decoder_layers()`` — the notebook mixed ``model.model.model.layers`` and
``model.base_model.model.model.layers`` paths; this removes that inconsistency):

* **patch**  — copy the *ally* (truthful) last-token residual into a *rival* run at layer L. If the
  model then answers truthfully, layer L carries the causal "tell-the-truth" state.
* **ablate** — remove the linear "truth" direction on *ally* contexts at layer L. If truth-telling
  survives, that 1-D direction is not causally necessary. (Notebook: ally truth stayed 1.00.)
* **steer**  — add the truth direction on *rival* contexts at layer L. If the model turns honest,
  deception is a suppression the truth vector can override. (Notebook: rival truth only 0.19.)

plus, since 2026-08-17, a fourth:

* **subspace** — ablate a rank-*k* subspace on *ally* contexts, k swept, so "not a 1-D handle" can be
  turned into a measured dimensionality instead of an inference (see D3 below).

The truth direction for ablate/steer is estimated via ``--axis`` (``lr``, default, or ``diffmeans``).
Exp 001's ablation was null on the ``lr`` (logistic-probe) direction across all 3 seeds — it reads the
truth but isn't the causal lever — so ``--axis diffmeans`` (difference of class means) is the fix; see
:func:`extract_truth_direction`. Patching does not use a direction, so ``--axis`` never affects it.

Layers follow the notebook (direction at L11, ablate L11, steer L13 α=5.0, patch L13) but run over the
full ``--task`` game family (default ``single``; ``--task multi`` for the notebook 8-game/3-var version).
Run per arm via ``--adapter`` / ``--instructed``.

    python src/perfect_aliasing/interventions.py --adapter adapters/rl_deceiver --mode all --out data/interv_emergent.json
    python src/perfect_aliasing/interventions.py --instructed                   --mode all --out data/interv_instructed.json

--- FOUR DEFECTS IN THE ORIGINAL PROTOCOL, AND THE FLAGS THAT FIX THEM (2026-08-17) ----------------

Every causal null reported before this date came out of the runners above under their old defaults, and
each point below is a reason such a null could be an artefact rather than a result. All four fixes sit
behind NEW flags and the old defaults are untouched, so the numbers already reported reproduce exactly.

D1  UNSIGNED STEERING CAN CANCEL.  :func:`steering_hook` adds the SAME ``+α·d`` on every trial
    regardless of that trial's true bit, and the reported outcome is an *aggregate* rival truth rate.
    For a ``d`` oriented toward ``truth=1``, an unsigned push HELPS the ``truth=1`` trials and HURTS the
    ``truth=0`` ones, so a genuinely causal axis can cancel to ~0 once pooled. The whole 0.000–0.03
    sweep (L24/28/32 × α=5/10/20/40) therefore cannot distinguish "no causal role" from "symmetric
    causal role, hidden by pooling".
    → ``--steer-mode signed`` conditions the sign on the true bit (``+α·d`` when ``truth==1``, ``−α·d``
    when ``truth==0``, i.e. always toward the true answer); ``--steer-mode both`` runs the pair on the
    SAME eval episodes, and that comparison IS the result: unsigned≈0 with signed≫0 means cancellation,
    both≈0 means no linear steer exists at that layer/α.
    The unsigned run additionally reports its truth=0 / truth=1 sub-rates (``by_truth``), which shows
    cancellation directly, and every run persists per-trial records so any later stratification can be
    done without re-running the model.
    ⚠️ SIGNED STEERING IS AN **ORACLE** INTERVENTION. It reads the ground-truth label to choose the
    sign, so a high signed truth rate is NOT evidence that the model uses this axis to decide anything.
    It is only evidence that the axis moves the answer at all — the necessary condition the unsigned
    aggregate cannot establish. It must never be written up as "we steered the model into honesty".

D2  THE CAUSAL TESTS WERE SILENTLY OFF-TASK.  ``--task`` accepted only ``single``/``multi``, but the
    mixed-fit directions the paper offers as its contribution are fit by ``identify_probe.py`` on the
    ``codebook`` task (hence the saved files named ``mixed_cb_g9b_{in,em}.json``). Loading one via
    ``--load-direction`` and intervening on ``single`` was therefore a CROSS-TASK test, with nothing in
    the logs saying so — and frozen cross-task transfer of these directions is already measured to
    collapse (0.864 at best, 0.288 in the RL arm, i.e. below chance). So "the mixed-fit direction is
    causally null too" was confounded with known task-specificity.
    → ``--task codebook`` / ``--task infercode`` run the interventions on the same task the direction
    was fit on, with the codebook readout: the answer is scored over THIS TRIAL'S two legal symbols and
    mapped back to semantics, exactly as ``identify_probe.collect`` does — never two fixed global
    tokens (see :func:`_answer_reader` for why that distinction is load-bearing). A saved-probe file
    records the task it was fit on, so a mismatch now ABORTS unless ``--allow-cross-task`` says the
    mismatch is deliberate.

D3  NO DIMENSIONALITY EVIDENCE.  "The information is causally used, but not through a 1-D linear
    handle" does not follow from "whole-residual patching flips at 1.000 while 1-D interventions fail":
    a whole-residual patch replaces the role-, prompt-, action- AND truth-associated components at
    once, so it licenses no claim about rank.
    → ``--mode subspace --subspace-ranks 1,2,4,8,16,32`` ablates a rank-k subspace built by iterative
    fit-and-residualize and reports the behaviour-vs-k curve (see :func:`fit_inlp_subspace`).

D4  THE ABLATION NULL HAD NO POSITIVE CONTROL.  Ablation left ally truth at 1.000 — but nothing
    checked that the ablation REMOVED the information. Behavioural invariance is equally consistent
    with an operation that did nothing at all.
    → ``--positive-control on`` refits a probe on the POST-ablation activations at the same layer and
    emits ``post_ablation_auroc``. If that has NOT fallen to chance, THE BEHAVIOURAL NULL AT THAT LAYER
    IS UNINTERPRETABLE and must not be reported as a causal null; the JSON says so outright in
    ``behavioural_null_interpretable``. "Chance" is calibrated by a LABEL-PERMUTATION NULL rather than
    assumed to be 0.5, because the residual is 2048–4096-dimensional and the control runs a few hundred
    episodes — at ``n < d`` a refit probe overfits and its null decodability sits near 0.55–0.61, so a
    fixed band would report "information still present" on pure noise
    (:func:`probe_decodability`, :func:`_decodability`).

    python src/perfect_aliasing/interventions.py --model-id gemma-9b --task codebook \
        --load-direction analysis/mixed_cb_g9b_em.json --dir-layer 24 --steer-layer 24 \
        --mode steer --steer-mode both --steer-alphas 5,10,20,40 --out data/steer_signed.json
    python src/perfect_aliasing/interventions.py --model-id gemma-9b --task codebook \
        --mode subspace --dir-layer 24 --ablate-layer 24 --subspace-ranks 1,2,4,8,16,32 \
        --positive-control on --out data/subspace.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import game as game_mod
import model as model_mod
import instructed as instructed_mod


# --- forward hooks (operate on a chosen token position of a layer's output) -------------------------
def patching_hook(source_vector, pos=-1):
    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        source = source_vector.squeeze()
        if tensor.ndim == 3:
            tensor[0, pos, :] = source
        elif tensor.ndim == 2:
            tensor[pos, :] = source
        return output
    return hook


def ablation_hook(direction):
    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        d = direction.squeeze()
        if tensor.ndim == 3:
            act = tensor[0, -1, :]
            tensor[0, -1, :] = act - torch.dot(act, d) * d
        elif tensor.ndim == 2:
            act = tensor[-1, :]
            tensor[-1, :] = act - torch.dot(act, d) * d
        return output
    return hook


def subspace_ablation_hook(basis):
    """Remove the component of the last-token residual lying in the span of ``basis`` — the rank-k
    generalisation of :func:`ablation_hook` (D3). ``basis`` is a ``(k, d)`` tensor with ORTHONORMAL
    ROWS, which is what makes ``B^T (B a)`` the orthogonal projector onto the span; feeding it a
    non-orthonormal set would silently over- or under-remove, so :func:`fit_inlp_subspace` orthonormalises
    and reports the achieved rank rather than trusting the caller.

    ⚠️ THE PROJECTION IS DONE IN fp32 EVEN ON A bf16 MODEL, deliberately. bf16 carries ~3 decimal
    digits, and ``B a`` is a sum of d products per row with k rows accumulated back; at k=32 the
    rounding residue left behind is large enough to look like "the ablation did not remove the
    information", i.e. it would corrupt exactly the D4 positive control this mode exists to satisfy.
    The write-back is cast to the tensor's own dtype so the rest of the forward pass is unchanged.
    (:func:`ablation_hook` is left in its original bf16 form on purpose: rewriting it would move
    every rank-1 number already reported.)"""
    basis32 = basis.to(torch.float32)

    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.ndim == 3:
            act = tensor[0, -1, :]
        elif tensor.ndim == 2:
            act = tensor[-1, :]
        else:
            raise RuntimeError(f"[interventions] unexpected activation rank {tensor.ndim} at the hooked layer")
        a32 = act.to(torch.float32)
        act.copy_((a32 - basis32.T @ (basis32 @ a32)).to(act.dtype))
        return output
    return hook


def steering_hook(direction, alpha, sign=1.0):
    """Add ``sign * alpha * direction`` to the last-token residual.

    ``sign=1.0`` (the default) is the historical unsigned steer: the same push on every trial. The
    caller passes ``sign=-1.0`` on ``truth==0`` trials to get the signed/oracle steer of D1 — see
    :func:`run_steering`. The sign lives here rather than in the direction so the two modes share one
    code path and one direction object, and so the sign actually used lands in each per-trial record."""
    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        d = direction.squeeze()
        if tensor.ndim == 3:
            tensor[0, -1, :] = tensor[0, -1, :] + sign * alpha * d
        elif tensor.ndim == 2:
            tensor[-1, :] = tensor[-1, :] + sign * alpha * d
        return output
    return hook


# --- helpers ---------------------------------------------------------------------------------------
def _prompt_inputs(tokenizer, device, messages):
    prompt = model_mod.render_prompt(tokenizer, messages)
    return tokenizer(prompt, return_tensors="pt").to(device)


def _unit(vec, what="direction"):
    """Unit-normalise, refusing a zero/non-finite vector rather than propagating NaNs into a hook
    (same contract as :func:`load_saved_direction`: never silently substitute)."""
    v = np.asarray(vec, dtype=np.float64).ravel()
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0:
        raise SystemExit(f"[interventions] {what} is zero/non-finite — refusing to normalise it.")
    return v / norm


def _answer_reader(tokenizer, task):
    """Return ``read(logits, codebook) -> semantic 0/1`` appropriate to ``task``.

    For ``single``/``multi`` this is the plain ``"0"``/``"1"`` readout (``model.predicted_bit``). For
    ``codebook``/``infercode`` it scores ONLY THE TWO LEGAL SYMBOLS OF THIS TRIAL and maps the winner
    back to semantics — the same rule as ``identify_probe.collect``, down to its ``>=`` tie-break
    toward semantic 0, so the two tools' numbers stay comparable.

    ⚠️ THIS IS THE EASIEST WAY TO PRODUCE SILENT GARBAGE IN THIS FILE. Scoring two fixed global tokens
    on a codebook trial measures the surface character, not the answer: the orientation is re-randomised
    every trial, so even a perfect policy would score ~0.5 and EVERY intervention would come out null
    for a reason that has nothing to do with the model. The reader is therefore built once from the task
    at the top of each runner, and the trial's own codebook is a REQUIRED argument on the codebook path
    — there is deliberately no default that could quietly do the wrong thing."""
    if not game_mod.uses_codebook(task):
        token_0, token_1 = model_mod.token_ids(tokenizer)

        def read_bits(logits, codebook=None):
            return model_mod.predicted_bit(logits, token_0, token_1)
        return read_bits

    symbols = sorted({s for pair in game_mod.CODEBOOK_PAIRS for s in pair})
    sym_ids = model_mod.symbol_token_ids(tokenizer, symbols)

    def read_symbols(logits, codebook):
        if not codebook:
            raise SystemExit(
                "[interventions] codebook task but this trial carries no codebook — refusing to fall "
                "back to the '0'/'1' readout, which would score the surface character, not the answer.")
        id0, id1 = sym_ids[codebook[0]], sym_ids[codebook[1]]
        return 0 if float(logits[id0]) >= float(logits[id1]) else 1
    return read_symbols


def _trial_messages(task, ep, role, instruction):
    """Chat messages for episode ``ep`` under ``role``, dispatching on whether ``task`` carries a codebook.

    ⚠️ THE DISPATCH IS LOAD-BEARING AND MUST STAY PAIRED WITH :func:`_answer_reader`. Rendering a
    ``codebook``/``infercode`` trial through ``game.build_messages`` would demand "0 or 1" from the model
    while the readout scored the trial's two symbols; every trial would then score as noise. Hence the
    explicit refusal below instead of a fallback."""
    if game_mod.uses_codebook(task):
        if ep.codebook is None:
            raise SystemExit(
                f"[interventions] task={task} requires a per-trial codebook but this episode has none — "
                f"refusing to render a prompt the readout cannot score.")
        return game_mod.build_codebook_messages(
            ep.game, ep.values, ep.target_variable, role, ep.query_text, ep.codebook,
            instruction=instruction, system_text=ep.system_text)
    return game_mod.build_messages(
        ep.game, ep.values, ep.target_variable, role, ep.query_text, instruction=instruction,
        system_text=ep.system_text)


def _sample_paired_trial(rng, task):
    """One trial's CONTENT plus both an ally and a rival role for it — the paired draw patching needs.

    Returns ``(episode, rival_role)``, the episode carrying the ally role. Game, values, target, true
    bit, query and (on the codebook variants) the per-trial symbol mapping are all SHARED between the
    two runs, which is the entire point: the ally and rival prompts must differ only in the role word,
    or the patched residual is not a counterfactual.

    ⚠️ The draw order on ``single``/``multi`` reproduces the pre-2026-08-17 inline loop in
    :func:`run_patching` exactly — games, values, var_index, query, ally role, rival role, and NO
    codebook draw — so ``--task single`` still yields the same episodes for a given seed as the patching
    numbers already reported. Do not reorder these calls to tidy them up.
    ⚠️ ONE codebook is drawn and used for BOTH runs. A fresh draw per run would re-map the symbols
    between the source and the target prompt, and the "flip rate" would then be measuring the mapping
    rather than the residual."""
    if game_mod.uses_infer(task):
        # Premises + a comparison rule; the bit is the outcome and is never a token in the prompt.
        system_text, target, true_value, orientation = game_mod.sample_infer(rng)
        game, values, var_index = {"system": system_text, "vars": [target]}, [true_value], 0
        query = rng.choice(game_mod.INFER_QUERIES).format(target)
    else:
        base = game_mod.TASK_SPEC.get(task, {}).get("base", task)
        games, queries = game_mod.games_for(base), game_mod.queries_for(base)
        g = rng.choice(games)
        nv = len(g["vars"])
        values = [rng.choice([0, 1]) for _ in range(nv)]
        var_index = rng.randint(0, nv - 1)
        game, target, true_value = g, g["vars"][var_index], values[var_index]
        query = rng.choice(queries).format(target)
        system_text, orientation = None, None
    ally_role = rng.choice(game_mod.ALLIES)
    rival_role = rng.choice(game_mod.RIVALS)
    codebook = game_mod.sample_codebook(rng) if game_mod.uses_codebook(task) else None
    ep = game_mod.Episode(game, values, var_index, target, true_value, True, ally_role, query,
                          codebook=codebook, system_text=system_text, infer_orientation=orientation)
    return ep, rival_role


def _by_truth(trials):
    """Truth rate split by the trial's true bit — the direct read on D1's cancellation.

    An unsigned steer that helps ``truth=1`` and hurts ``truth=0`` by the same amount pools to ~chance
    and reads as "no causal effect". These two numbers show that instead of hiding it, and they cost
    nothing because the per-trial records already exist."""
    out = {}
    for bit in (0, 1):
        sel = [t for t in trials if t["truth"] == bit]
        out[f"truth{bit}_n"] = len(sel)
        out[f"truth{bit}_rate"] = (
            sum(int(t["emitted"] == t["truth"]) for t in sel) / len(sel)) if sel else None
    return out


def _alpha_rel(alpha, norm_median):
    """α expressed as a fraction of the median residual norm at the intervention layer (D1).

    A bare ``alpha=20`` is uninterpretable: residual norms differ by an order of magnitude between
    layers and between families, so the same α is a shove at L4 and a nudge at L40. ``None`` when the
    calibration was skipped, rather than a fake 1.0."""
    if not norm_median:
        return None
    return float(alpha) / float(norm_median)


def collect_layer_activations(model, tokenizer, device, instruction, layer, n, seed, task,
                              force_ally=True, make_hook=None, batch=1):
    """Last-token activations at ``layer`` plus the true bit, over ``n`` sampled episodes → ``(X, y)``.

    Shared by the direction fit, the subspace fit and the D4 positive control. ``make_hook`` (a callable
    ``episode -> (hook, meta)``) is what makes the control possible: with it, the activations are read
    WHILE the intervention is live, so a refit probe sees exactly what the model's later layers saw.
    Reading ``hidden_states[layer]`` is correct for that — HF appends the hidden state of layer L
    *after* layer L has run, hence after our forward hook has mutated its output in place, so the
    captured tensor is the POST-intervention residual.

    The episode stream is identical to the pre-2026-08-17 loop that lived inside
    :func:`extract_truth_direction` (same ``random.Random(seed)``, one ``sample_episode`` draw per
    trial, same order), so factoring this out moved no existing number.

    ``batch`` > 1 runs the forward passes in padded batches. **This is the fix for the real cost of every
    ``STAGE=interv`` job**: the INLP fit blamed for the evictions costs ~25 s, while this
    function's batch-size-1 loop over 4000 episodes costs ~70 minutes and does so at a GPU utilisation low
    enough to trip the <20 %-for-2h reaper — so the unprotectable pre-checkpoint stretch is HERE, not in the
    fit. ``batch=1`` is the default and runs the original loop, character for character, so no existing
    number can move.

    ⛔ **BATCHING IS REFUSED WHEN ``make_hook`` IS SET, and this is not conservatism — the hooks are
    batch-unsafe by construction.** :func:`ablation_hook` and :func:`subspace_ablation_hook` both index
    ``tensor[0, -1, :]``: hard-coded batch row 0, and ``-1`` meaning the last *padded* position. Under a
    batch of B they would intervene on one row out of B and read pad positions for the rest, leaving the D4
    positive control seeing mostly un-ablated activations — i.e. silently inflating post-ablation
    decodability, the exact quantity the control exists to measure. Fixing that means rewriting both hooks,
    which would move every ablation number already reported. So the hooked path (``control_n``, ~600
    episodes) stays per-episode and only the un-hooked fit (~4000 episodes) is batched, which is where the
    cost is anyway. **Do not "finish the job" by batching the control without rewriting the hooks first.**"""
    rng = random.Random(seed)
    layers = model_mod.get_decoder_layers(model) if make_hook is not None else None
    if int(batch) > 1 and make_hook is None:
        return _collect_batched(model, tokenizer, device, instruction, layer, n, rng, task,
                                force_ally, int(batch))
    X, y = [], []
    for _ in range(n):
        ep = game_mod.sample_episode(rng, task=task, force_ally=force_ally)
        messages = _trial_messages(task, ep, ep.role, instruction)
        handle = None
        if make_hook is not None:
            hook, _meta = make_hook(ep)
            handle = layers[layer - 1].register_forward_hook(hook)
        with torch.no_grad():
            outputs = model(**_prompt_inputs(tokenizer, device, messages), output_hidden_states=True)
        if handle is not None:
            handle.remove()
        X.append(outputs.hidden_states[layer][0, -1, :].detach().cpu().to(torch.float32).numpy())
        y.append(ep.true_value)
    return np.array(X), np.array(y)


def _collect_batched(model, tokenizer, device, instruction, layer, n, rng, task, force_ally, batch):
    """Batched twin of :func:`collect_layer_activations`'s loop. Same episodes, same order, same tensors.

    Three things make it equivalent rather than merely similar, and each is a place it could go wrong:

    * **Episode order.** Episodes are drawn from the *same* ``rng`` with the same one-``sample_episode``-
      per-trial cadence, so trial *i* here is trial *i* there. The draws happen before any forward pass,
      which is safe only because sampling consumes no model state.
    * **Padding side.** Padding is forced RIGHT for this call and restored afterwards. Real tokens then
      occupy positions ``0..L-1``, which is what HF's default ``position_ids`` assumes, so no position
      shifts relative to the unbatched pass.
    * **Which position is read.** NOT ``-1`` — that is a pad token for every row shorter than the longest
      in its batch. The last *real* index is ``attention_mask.sum(dim=1) - 1`` per row. Getting this wrong
      would silently read padding and produce activations that look plausible and mean nothing.
    """
    eps = [game_mod.sample_episode(rng, task=task, force_ally=force_ally) for _ in range(n)]
    prompts = [model_mod.render_prompt(tokenizer, _trial_messages(task, ep, ep.role, instruction))
               for ep in eps]
    prev_side, prev_pad = tokenizer.padding_side, tokenizer.pad_token
    if tokenizer.pad_token is None:                 # most causal LMs ship without one
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    X, y = [], []
    try:
        for i in range(0, n, batch):
            chunk = prompts[i:i + batch]
            enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**enc, output_hidden_states=True)
            h = outputs.hidden_states[layer]                       # (B, L, d)
            last = enc["attention_mask"].sum(dim=1) - 1           # last REAL token per row
            rows = h[torch.arange(h.shape[0], device=h.device), last, :]
            X.extend(rows.detach().cpu().to(torch.float32).numpy())
        y = [ep.true_value for ep in eps]
    finally:
        tokenizer.padding_side, tokenizer.pad_token = prev_side, prev_pad
    return np.array(X), np.array(y)


def extract_truth_direction(model, tokenizer, device, instruction, layer, n, seed, task, axis="lr"):
    """Extract a unit 'truth' direction at ``layer`` from ally-context activations (notebook cell 13,
    phase 1), returned as a tensor matching the model's dtype/device. Two estimators:

    * ``"lr"`` (default) — the logistic 'truth' probe's coefficient vector. Reads the true bit well, but
      Exp 001's ablation-null across all 3 seeds showed it is a *correlate the probe reads*, not the
      *causal axis the model acts on*.
    * ``"diffmeans"`` — ``mean(act | true=1) − mean(act | true=0)``, the difference-of-class-means axis
      (Marks & Tegmark, "Geometry of Truth"). Standard causally-effective direction; the fix for the
      ablation-null (STATUS §5 TODO #1).

    Both point toward ``true_value = 1``, so they are drop-in interchangeable in the ablation/steering
    hooks. ``axis`` has no effect on patching (which copies a full residual, not a direction).

    On ``--task codebook``/``infercode`` the calibration prompts now carry the per-trial symbol mapping
    too (via :func:`collect_layer_activations`), so a direction estimated here is estimated on the same
    prompt distribution the intervention will run on — D2."""
    X, y = collect_layer_activations(model, tokenizer, device, instruction, layer, n, seed, task,
                                    force_ally=True)
    if axis == "diffmeans":
        if not (np.any(y == 1) and np.any(y == 0)):
            raise ValueError(
                f"diff-of-means needs both true=0 and true=1 in calibration (got {np.bincount(y, minlength=2)}); "
                "raise --calib-n."
            )
        w = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    else:
        w = LogisticRegression(max_iter=1000).fit(X, y).coef_[0]
    w = w / np.linalg.norm(w)
    return torch.tensor(w, dtype=model_mod.model_dtype(device), device=device)


def _saved_layer_row(path, layer):
    """The row for ``layer`` in an ``identify_probe.py --save-probe`` file, plus the whole blob.

    Refuses a missing layer rather than falling back to a neighbour — a direction from the wrong layer
    would produce a plausible-looking number that means nothing."""
    blob = json.loads(Path(path).read_text())
    rows = blob.get("layers") or []
    match = [r for r in rows if int(r["layer"]) == int(layer)]
    if not match:
        raise SystemExit(
            f"[interventions] {path} has no probe for layer {layer} (has "
            f"{sorted(int(r['layer']) for r in rows)}). Refusing to silently substitute another layer.")
    return blob, match[0]


def _saved_meta(blob, path, layer, expect_dim, n_features):
    """Provenance of a loaded direction, and the wrong-model guard.

    ``n_features`` vs the model's hidden size is the same check ``identify_probe.run_layer`` makes on a
    frozen probe: a length mismatch means the file came from a different model (or a different layer
    convention) and every number downstream would be noise, so it aborts."""
    if expect_dim is not None and int(n_features) != int(expect_dim):
        raise SystemExit(
            f"[interventions] {path} @L{layer} has {n_features} features but this model's residual is "
            f"{expect_dim}-dimensional — wrong model or layer alignment. Refusing to intervene with it.")
    return {"path": str(path), "layer": int(layer), "task": blob.get("task"), "arm": blob.get("arm"),
            "regime": blob.get("regime"), "target": blob.get("target"), "n_features": int(n_features)}


def load_saved_direction(path, layer, device, expect_dim=None):
    """Load a probe direction saved by ``identify_probe.py --save-probe`` and use it for ablate/steer.
    Returns ``(unit_direction_tensor, meta)``.

    WHY THIS EXISTS.  Every causal null we report -- ablation leaving ally truth at 1.000, the steering
    sweep at 0.0000 -- was measured on a direction :func:`extract_truth_direction` estimates *itself*, from
    **ally-context** activations.  That is the ALLY-FIT direction: the unidentified one, the thing the paper
    criticises.  The direction the paper hands back as its contribution is the **MIXED-fit** probe, and
    until now there was no way to feed it to the intervention hooks at all, so it had never been causally
    tested.  This closes that gap.

    The saved file is ``{"layers": [{"layer": L, "coef": [...], "intercept": f}, ...]}``.  The intercept is
    deliberately ignored: a bias shifts the decision threshold but not the direction, and the hooks
    ablate/add along a unit vector.  Normalised here so ``alpha`` means the same thing as it does for the
    estimated axes.

    ``meta`` carries the task the probe was FIT on, which :func:`check_direction_task` compares against
    ``--task`` so a cross-task intervention can no longer happen silently (D2)."""
    blob, row = _saved_layer_row(path, layer)
    w = np.asarray(row["coef"], dtype=np.float64)
    norm = np.linalg.norm(w)
    if not np.isfinite(norm) or norm == 0:
        raise SystemExit(f"[interventions] probe direction at L{layer} in {path} is zero/non-finite.")
    meta = _saved_meta(blob, path, layer, expect_dim, w.size)
    return torch.tensor(w / norm, dtype=model_mod.model_dtype(device), device=device), meta


def load_saved_subspace(path, layer, expect_dim=None):
    """A SUPPLIED DIRECTION SET at ``layer`` → ``(m, d)`` float64 array of unit rows, plus ``meta``.

    Accepts either shape of file, so the same flag covers both cases the subspace mode must support:

    * a plain ``--save-probe`` file — one ``coef`` per layer, giving ``m = 1`` (the paper's mixed-fit
      direction, which :func:`fit_inlp_subspace` then tops up to rank k);
    * a layer row carrying an explicit ``basis`` (or ``coefs``) list of vectors — a fully supplied set,
      e.g. one exported from another tool.

    Rows are unit-normalised but NOT orthogonalised here; that happens in :func:`fit_inlp_subspace`,
    which is also where the achieved rank is measured and reported."""
    blob, row = _saved_layer_row(path, layer)
    raw = row.get("basis") or row.get("coefs")
    rows = [np.asarray(v, dtype=np.float64).ravel() for v in raw] if raw else [
        np.asarray(row["coef"], dtype=np.float64).ravel()]
    widths = {v.size for v in rows}
    if len(widths) != 1:
        raise SystemExit(f"[interventions] {path} @L{layer} has ragged direction rows {sorted(widths)}.")
    basis = np.array([_unit(v, what=f"supplied direction {i} at L{layer}") for i, v in enumerate(rows)])
    meta = _saved_meta(blob, path, layer, expect_dim, basis.shape[1])
    meta["supplied_rows"] = int(basis.shape[0])
    return basis, meta


def check_direction_task(meta, task, allow_cross_task, path):
    """D2: refuse a SILENT cross-task intervention; return the task the direction was fit on.

    ``identify_probe.py --save-probe`` records the task it fit on. Intervening on a different task is a
    legitimate experiment, but it is a DIFFERENT experiment: frozen cross-task transfer of these very
    directions is already measured to collapse (0.864 at best, 0.288 — below chance — in the RL arm), so
    a null measured across tasks is confounded with task-specificity and cannot be reported as "the
    mixed-fit direction is causally inert". Every null reported before 2026-08-17 was of exactly this
    kind, because ``--task`` could not even name ``codebook``. Hence: abort by default, and
    ``--allow-cross-task`` makes the choice explicit and lands it in the output JSON."""
    fit_task = meta.get("task")
    if fit_task is None:
        print(f"WARNING: {path} records no `task` field (it predates task provenance), so the "
              f"direction/task match against --task {task} CANNOT be verified. Treat the result as "
              f"unidentified with respect to D2.")
        return None
    if fit_task == task:
        return fit_task
    msg = (f"[interventions] {path} was fit on task={fit_task!r} but --task={task!r}. That is a "
           f"CROSS-TASK causal test, confounded with the measured collapse of frozen cross-task "
           f"transfer for these directions. Re-run with --task {fit_task} (the matched test), or pass "
           f"--allow-cross-task if the mismatch is the point of the run.")
    if not allow_cross_task:
        raise SystemExit(msg)
    print("WARNING: " + msg + "  (proceeding: --allow-cross-task)")
    return fit_task


def _basis_stats(basis):
    """Achieved rank and orthonormality error of a ``(k, d)`` basis.

    Reported alongside the requested ``k`` so a rank-deficient subspace can never be written up as
    though it had the rank that was asked for — the same discipline as never letting a log line
    misreport the effective axis."""
    basis = np.asarray(basis, dtype=np.float64)
    if basis.size == 0:
        return {"n_rows": 0, "rank": 0, "orthonormality_err": None}
    sv = np.linalg.svd(basis, compute_uv=False)
    rank = int((sv > max(sv[0], 1e-30) * 1e-8).sum())
    err = float(np.abs(basis @ basis.T - np.eye(basis.shape[0])).max())
    return {"n_rows": int(basis.shape[0]), "rank": rank, "orthonormality_err": err}


def fit_inlp_subspace(X, y, k, axis="lr", supplied=None):
    """Build a rank-``k`` subspace by ITERATIVE FIT-AND-RESIDUALIZE — INLP, after Ravfogel et al.,
    "Null It Out" (2020). Returns ``(basis, meta)`` with ``basis`` a ``(k', d)`` array of orthonormal
    rows (``k' <= k``; see the rank-deficiency note below).

    THE METHOD, and why it is the right construction for D3.  Ablating the top-k directions of k
    *independent* probe fits does not give a k-dimensional ablation: successive fits on the same data
    return nearly the same vector, so the "rank-8" subspace is really rank ~1 with rounding. INLP fixes
    that by removing what it has already found before looking again:

        1. fit a linear probe for the label on the current activations → w
        2. record ``w / ||w||``
        3. project the activations onto the NULLSPACE of w (``X ← X − (X w) wᵀ``)
        4. refit on the residual; repeat k times
        5. orthonormalise the collected directions (QR) and report the achieved rank

    Step 3 is what makes step 4 informative: each new direction carries information the previous ones
    could not, so ablating the span is a genuine rank-k intervention and the behaviour-vs-k curve is a
    dose-response in DIMENSION. The resulting subspaces are also NESTED — a prefix of one run — which is
    why :func:`main` fits once at ``max(ranks)`` and slices, rather than refitting per k.

    ``supplied`` (an ``(m, d)`` array, e.g. the paper's mixed-fit direction from
    :func:`load_saved_subspace`) seeds the basis: those rows go in first and are residualized out, then
    INLP tops the basis up to k with locally fit directions. So the k=1 point of a seeded sweep is
    exactly the 1-D test the paper already reports, and larger k extends it.

    ⚠️ RANK DEFICIENCY IS REPORTED, NOT PAPERED OVER. Once the residual carries no linear signal the
    refit returns a numerically zero (or already-spanned) vector; the loop then STOPS rather than
    appending a garbage row that would inflate the nominal rank. ``meta["rank_deficient"]`` says so, and
    the caller must report ``achieved_rank``, not ``k``.

    ⚠️ EXPECT THE SWEEP TO TERMINATE EARLY, AND READ THAT AS THE ANSWER. Measured on synthetic data with
    32 independent label-carrying axes (margins 4.0…0.6, n=400, d=64), LR-INLP drove the class-mean
    separation from 8.0 to 1e-16 in SIX iterations and the seventh refit returned an exactly zero
    coefficient vector. That is not a bug: because a logistic fit finds a single hyperplane that
    separates the classes, removing its normal removes most of the linear signal at once. So a
    ``1,2,4,8,16,32`` request will usually stop somewhere in single digits, and THE TERMINATION RANK IS
    ITSELF THE DIMENSIONALITY ESTIMATE D3 is asking for — "the label lives in a ~k-dimensional linear
    subspace here" — provided it is reported as ``achieved_rank`` and not silently rounded up to the k
    that was asked for.

    ⚠️ ``axis="diffmeans"`` CANNOT EXCEED RANK 1 IN THIS LOOP, by construction: step 3 residualizes the
    class-mean difference out, which sets the next iteration's class-mean difference to exactly zero.
    Use ``axis="lr"`` for a rank sweep; diffmeans remains available so the k=1 cell can be compared
    against the rank-1 result computed with the same estimator.

    ⚠️ n VS d. With ``n < d`` a logistic fit separates ANY labelling, so late directions are fitting
    noise and the behaviour-vs-k curve understates the effect of a real k-dimensional ablation.
    ``meta["underdetermined"]`` flags it; raise the fit n (``--subspace-n``) toward the residual width
    before reporting a curve.

    Pure numpy/sklearn on ``(n, d)`` activations — no model needed, which is what makes it unit-testable
    on synthetic data."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2:
        raise SystemExit(f"[interventions] INLP needs a 2-D (n, d) activation matrix, got {X.shape}.")
    if int(k) < 1:
        raise SystemExit(f"[interventions] subspace rank must be >= 1, got {k}.")
    d = X.shape[1]
    dirs, work, stopped = [], X.copy(), None
    n_supplied = 0

    if supplied is not None:
        for i, row in enumerate(np.asarray(supplied, dtype=np.float64).reshape(-1, d)):
            if len(dirs) >= int(k):
                break
            w = _unit(row, what=f"supplied direction {i}")
            for u in dirs:                      # a supplied SET need not be orthogonal
                w = w - np.dot(w, u) * u
            norm = np.linalg.norm(w)
            if not np.isfinite(norm) or norm < 1e-8:
                stopped = f"supplied direction {i} is (numerically) already spanned"
                break
            w = w / norm
            dirs.append(w)
            work = work - np.outer(work @ w, w)
            n_supplied += 1

    while len(dirs) < int(k) and stopped is None:
        if len(np.unique(y)) < 2:
            raise SystemExit(
                f"[interventions] INLP needs both classes present (got {np.bincount(y, minlength=2)}); "
                f"raise the fit n.")
        if axis == "diffmeans":
            w = work[y == 1].mean(axis=0) - work[y == 0].mean(axis=0)
        else:
            w = LogisticRegression(max_iter=1000).fit(work, y).coef_[0]
        norm = np.linalg.norm(w)
        if not np.isfinite(norm) or norm < 1e-12:
            stopped = f"refit at rank {len(dirs) + 1} returned a zero/non-finite direction"
            break
        w = w / norm
        # Re-orthogonalise against everything already removed. The refit can only return a direction
        # that is *nearly* in the nullspace of the previous ones; float error leaves a small in-basis
        # component, and left alone it makes the collected set non-orthogonal — so the projector would
        # over-remove and the achieved rank would come out below k for no substantive reason.
        for u in dirs:
            w = w - np.dot(w, u) * u
        norm = np.linalg.norm(w)
        if not np.isfinite(norm) or norm < 1e-8:
            stopped = f"residual at rank {len(dirs) + 1} carries no linear signal outside the span"
            break
        w = w / norm
        dirs.append(w)
        work = work - np.outer(work @ w, w)

    if not dirs:
        raise SystemExit(f"[interventions] INLP produced no directions at all ({stopped}).")
    # QR on the collected set: mathematically redundant after the explicit re-orthogonalisation above,
    # but it is the cheap numerically-clean way to guarantee the exact orthonormality the projector in
    # subspace_ablation_hook assumes. Sign flips are irrelevant — the projector is sign-invariant.
    q, _ = np.linalg.qr(np.array(dirs).T)
    basis = np.ascontiguousarray(q.T[:len(dirs)])
    stats = _basis_stats(basis)
    meta = {"method": "inlp", "axis": axis, "requested_k": int(k), "achieved_rank": stats["rank"],
            "n_directions": stats["n_rows"], "orthonormality_err": stats["orthonormality_err"],
            "n_supplied": n_supplied, "n_fitted": stats["n_rows"] - n_supplied,
            "basis_source": ("supplied+inlp" if n_supplied and stats["n_rows"] > n_supplied
                             else "supplied" if n_supplied else "inlp"),
            "rank_deficient": bool(stats["rank"] < int(k)), "stopped_early": stopped,
            "n_samples": int(X.shape[0]), "n_features": int(d),
            "underdetermined": bool(X.shape[0] < d)}
    return basis, meta


def probe_auroc(X, y, seed=0, test_size=0.3):
    """Held-out AUROC of a FRESHLY FIT logistic probe for ``y`` on ``X`` — the decodability measure the
    D4 positive control uses. ``None`` when a class is too thin to split, never a fabricated 0.5.

    Refitting (rather than reusing the ablated direction) is deliberate and is the *strong* form of the
    control: it asks whether ANY linear direction still decodes the label after the intervention, not
    just whether the one we removed does. That is also precisely why rank-1 ablation usually fails this
    control — which is the dimensionality story of D3, not a bug in the control."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    counts = np.bincount(y, minlength=2)
    if len(np.unique(y)) < 2 or counts.min() < 4:
        return None
    tr, te = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)
    if len(np.unique(y[te])) < 2:
        return None
    clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], clf.decision_function(X[te])))


def _decodability(auroc):
    """Orientation-free decodability: ``max(auroc, 1 − auroc)``.

    ⚠️ WHY THE RAW AUROC IS THE WRONG THING TO THRESHOLD.  Measured on synthetic activations in the
    ``d > n`` regime this file actually runs in (n=200, d=4096): after a rank-1 ablation the refit probe
    scored auroc = **0.000** on held-out data. Read as "distance from 0.5" that is a catastrophic miss —
    but 0.000 means the label is *perfectly* recoverable, just sign-flipped. Removing the direction that
    separates the training set over-corrects and inverts the residual. Collapsing to decodability makes
    "information is still linearly present" a single monotone quantity, which is the question D4 asks."""
    if auroc is None:
        return None
    return float(max(auroc, 1.0 - auroc))


def probe_decodability(X, y, seed=0, test_size=0.3, n_perm=20):
    """Decodability of ``y`` from ``X`` by a refit linear probe, WITH A LABEL-PERMUTATION NULL.

    Returns ``{"auroc", "decodability", "null_decodability_mean", "null_decodability_q95", "n_perm"}``.

    ⚠️ THE NULL IS NOT OPTIONAL AT THESE n AND d.  The residual is 2048–4096-dimensional and the control
    runs a few hundred episodes, so ``n < d`` and a logistic fit can separate ANY labelling of the
    training half. Its held-out decodability is then well above 0.5 even with no information present at
    all: measured at n=200, d=4096 the permuted-label null sat at 0.546 mean / **0.611** at the 95th
    percentile, not at 0.5. Thresholding against a fixed ``0.5 ± tol`` band in that regime would declare
    "information still present" on pure noise and every ablation would look ineffective.
    Permuting the labels and refitting reproduces exactly the same overfitting, so the null's own
    quantile is the honest chance band. ``n_perm=0`` disables it and falls back to the fixed band, which
    is only defensible when ``n >> d``."""
    auroc = probe_auroc(X, y, seed=seed, test_size=test_size)
    out = {"auroc": auroc, "decodability": _decodability(auroc), "n_perm": int(n_perm),
           "null_decodability_mean": None, "null_decodability_q95": None}
    if auroc is None or int(n_perm) < 2:
        return out
    y = np.asarray(y)
    nulls = []
    for i in range(int(n_perm)):
        yp = np.random.RandomState(seed + 1000 + i).permutation(y)
        dec = _decodability(probe_auroc(X, yp, seed=seed, test_size=test_size))
        if dec is not None:
            nulls.append(dec)
    if nulls:
        out["null_decodability_mean"] = float(np.mean(nulls))
        out["null_decodability_q95"] = float(np.quantile(nulls, 0.95))
    return out


def run_patching(model, tokenizer, device, instruction, layer, n, seed, task, pos=-1):
    """Flip rate: fraction of rival trials that answer truthfully after the ally residual is patched
    in at ``layer`` (notebook cell 12). ``pos`` selects the token position that is read from the ally
    run and overwritten in the rival run — default ``-1`` (the answer slot, as in seed 0). A
    context-aware non-final offset (e.g. ``-3``) makes the causal test independent of the probe's
    last-token read, so a flip there can't be dismissed as the same output-position confound.

    Returns a dict (``flip_rate`` plus the per-trial records) rather than a bare float, so a later
    stratification — by true bit, by role, by codebook orientation — needs no GPU re-run."""
    rng = random.Random(seed)
    layers = model_mod.get_decoder_layers(model)
    reader = _answer_reader(tokenizer, task)
    flips, trials = 0, []
    for _ in range(n):
        ep, rival_role = _sample_paired_trial(rng, task)

        ally_msgs = _trial_messages(task, ep, ep.role, instruction)
        with torch.no_grad():
            ally_out = model(**_prompt_inputs(tokenizer, device, ally_msgs), output_hidden_states=True)
        source = ally_out.hidden_states[layer][0, pos, :].detach()

        rival_msgs = _trial_messages(task, ep, rival_role, instruction)
        handle = layers[layer - 1].register_forward_hook(patching_hook(source, pos=pos))
        with torch.no_grad():
            logits = model(**_prompt_inputs(tokenizer, device, rival_msgs)).logits[0, -1, :]
        handle.remove()
        answer = reader(logits, ep.codebook)
        flips += int(answer == ep.true_value)
        trials.append({"truth": int(ep.true_value), "emitted": int(answer), "is_ally": False,
                       "role": rival_role, "intervention": "patch", "pos": int(pos),
                       "codebook": list(ep.codebook) if ep.codebook else None})
    return {"flip_rate": flips / max(n, 1), "n": n, "by_truth": _by_truth(trials), "trials": trials}


def _hooked_truth_rate(model, tokenizer, device, instruction, layer, make_hook, n, seed, task,
                       force_ally):
    """Shared loop for ablation/steering: run n episodes of a fixed context, apply the hook that
    ``make_hook(episode)`` builds at ``layer``, and report how often the model answers the *true* value.

    ``make_hook`` is a per-trial FACTORY, not a hook, because signed steering needs the trial's true bit
    to choose its sign (D1). It returns ``(hook, meta)`` and ``meta`` is merged into that trial's
    record, so what the hook ACTUALLY did (steered or not, α, sign, rank) is read out of the output JSON
    rather than inferred from the command line — the same rule that keeps the log lines honest."""
    rng = random.Random(seed)
    layers = model_mod.get_decoder_layers(model)
    reader = _answer_reader(tokenizer, task)
    truths, trials = 0, []
    for _ in range(n):
        ep = game_mod.sample_episode(rng, task=task, force_ally=force_ally)
        messages = _trial_messages(task, ep, ep.role, instruction)
        hook, meta = make_hook(ep)
        handle = layers[layer - 1].register_forward_hook(hook)
        with torch.no_grad():
            logits = model(**_prompt_inputs(tokenizer, device, messages)).logits[0, -1, :]
        handle.remove()
        answer = reader(logits, ep.codebook)
        truths += int(answer == ep.true_value)
        trials.append({"truth": int(ep.true_value), "emitted": int(answer),
                       "is_ally": bool(ep.is_ally), "role": ep.role,
                       "codebook": list(ep.codebook) if ep.codebook else None, **meta})
    return {"truth_rate": truths / max(n, 1), "n": n, "by_truth": _by_truth(trials), "trials": trials}


def _ablation_factory(direction):
    return lambda ep: (ablation_hook(direction), {"intervention": "ablate", "rank": 1, "steered": False})


def _subspace_factory(basis, rank):
    return lambda ep: (subspace_ablation_hook(basis),
                       {"intervention": "subspace_ablate", "rank": int(rank), "steered": False})


def run_ablation(model, tokenizer, device, instruction, direction, layer, n, seed, task):
    """Ally truth rate after ablating the truth direction (notebook cell 13, Test A).

    ⚠️ A null here is uninterpretable on its own — see D4 and :func:`run_positive_control`. Behaviour
    surviving the ablation is equally consistent with "the direction is not the causal lever" and with
    "the ablation removed nothing", and only the positive control tells those apart."""
    return _hooked_truth_rate(model, tokenizer, device, instruction, layer,
                              _ablation_factory(direction), n, seed, task, force_ally=True)


def run_subspace_ablation(model, tokenizer, device, instruction, basis, rank, layer, n, seed, task):
    """Ally truth rate after ablating a rank-``rank`` subspace at ``layer`` (D3).

    Same context and outcome as :func:`run_ablation` so the k=1 point is directly comparable to the
    rank-1 result already reported (modulo the fp32 projection noted in
    :func:`subspace_ablation_hook`)."""
    return _hooked_truth_rate(model, tokenizer, device, instruction, layer,
                              _subspace_factory(basis, rank), n, seed, task, force_ally=True)


def run_steering(model, tokenizer, device, instruction, direction, layer, alpha, n, seed, task,
                 signed=False):
    """Rival truth rate after steering along the truth direction (notebook cell 13, Test B).

    ``signed=False`` (default; the historical behaviour) adds the same ``+α·d`` on every trial.
    ``signed=True`` is the ORACLE steer of D1: ``+α·d`` when the true bit is 1, ``−α·d`` when it is 0,
    i.e. always toward the true answer, so a symmetric causal effect can no longer cancel in the
    aggregate.

    ⚠️ The oracle steer uses the ground-truth label to pick its sign. A high signed truth rate is
    therefore NOT evidence that the model reads this axis to decide anything — it is evidence that the
    axis can move the answer at all, which is the necessary condition the unsigned aggregate cannot
    establish. The publishable quantity is the unsigned-vs-signed CONTRAST, not the signed number."""
    def make(ep):
        sign = 1.0
        if signed:
            sign = 1.0 if ep.true_value == 1 else -1.0
        return steering_hook(direction, alpha, sign=sign), {
            "intervention": "steer", "steered": True, "alpha": float(alpha), "sign": float(sign),
            "steer_mode": "signed" if signed else "unsigned"}
    return _hooked_truth_rate(model, tokenizer, device, instruction, layer, make, n, seed, task,
                              force_ally=False)


def measure_residual_norm(model, tokenizer, device, instruction, layer, n, seed, task,
                          force_ally=False):
    """Median L2 norm of the last-token residual at ``layer`` — the α calibration factor (D1).

    Measured on the same context steering runs on (rival by default) so ``alpha_rel = α / median`` says
    what fraction of the residual's own scale the intervention adds. Without it, ``alpha=20`` is not
    comparable across layers, models or families, and "the steer was under-powered" cannot be ruled out
    or in."""
    X, _y = collect_layer_activations(model, tokenizer, device, instruction, layer, n, seed, task,
                                      force_ally=force_ally)
    return float(np.median(np.linalg.norm(X, axis=1)))


def run_positive_control(model, tokenizer, device, instruction, layer, make_hook, n, seed, task,
                         chance_tol=0.1, n_perm=20, force_ally=True):
    """D4 — did the ablation actually REMOVE the information it was supposed to remove?

    Collects the layer-``layer`` activations twice on the SAME episodes (same seed, hence paired): once
    clean and once with the ablation hook live, then refits a probe for the true bit on each. Reports:

    * ``pre_ablation_auroc``  — decodability before the intervention. This is also the sanity baseline:
      if it is not clearly above the null band, the layer never carried the label and nothing downstream
      means anything, which ``layer_carried_label`` records;
    * ``post_ablation_auroc`` — decodability after it;
    * ``chance_ceiling`` — the top of the honest chance band, ``max(0.5 + chance_tol, the 95th
      percentile of the label-permutation null on the POST-ablation activations)``. See
      :func:`probe_decodability`: at ``n < d`` a refit probe overfits, so the permuted-label null sits
      well above 0.5 and a fixed band would call noise "information";
    * ``information_removed`` — post-ablation *decodability* (orientation-free, see
      :func:`_decodability`) at or below ``chance_ceiling``;
    * ``behavioural_null_interpretable`` — the same boolean, named for how it must be used.

    ⚠️ THE INTERPRETATION GUARD.  If ``post_ablation_auroc`` has NOT dropped to ~chance, then a
    behavioural null at this layer (ally truth unchanged) is UNINTERPRETABLE and MUST NOT be reported as
    a causal null: the intervention did not remove the thing whose necessity was being tested. Only a
    behavioural null WITH ``information_removed = true`` is evidence that the information is not used
    through that subspace.

    ⚠️ EXPECT RANK-1 TO FAIL THIS CONTROL, AND EXPECT THAT TO BE THE FINDING.  Ablating an *estimated*
    direction leaves ``sin(θ)`` of the class separation behind, where θ is the estimation error — on
    synthetic data with a genuinely 1-D signal (n=400, d=64, margin 4.0) a rank-1 ablation of the
    fitted direction still left decodability at 0.73, while ablating the exact generating direction
    dropped it to 0.51. So "the ablation was ineffective" is the DEFAULT outcome for a 1-D handle, not a
    surprise, and it is exactly why every rank-1 causal null in this project needs this control before
    it can be reported.

    ⚠️ THE CONTROL'S EPISODES MUST BE DISJOINT FROM THE ONES THE ABLATED DIRECTIONS WERE FIT ON, and in
    :func:`main` they are (the fit uses ``--seed``, the control ``--seed + 400``). Do not "simplify" that
    by reusing the fit activations. Measured on synthetic data: the ablation directions are functions of
    the fit labels, so scoring the SAME sample leaks and the decodability-vs-k curve comes out
    non-monotone (0.792 → 0.596 → 0.726 → 0.749 for k=1,2,4,6), whereas on a disjoint sample it behaves
    (0.859 → 0.731 → 0.658 → 0.683). A non-monotone curve would be read as a substantive finding about
    rank; it would be an artefact of scoring the fit sample.

    ⚠️ POWER.  Raise ``--control-n`` toward (and past) the residual width before putting these numbers in
    a paper; the permutation null keeps the verdict *valid* at small n, but it cannot make it sharp."""
    X_pre, y_pre = collect_layer_activations(model, tokenizer, device, instruction, layer, n, seed,
                                             task, force_ally=force_ally)
    X_post, y_post = collect_layer_activations(model, tokenizer, device, instruction, layer, n, seed,
                                               task, force_ally=force_ally, make_hook=make_hook)
    pre = probe_decodability(X_pre, y_pre, seed=seed, n_perm=n_perm)
    post = probe_decodability(X_post, y_post, seed=seed, n_perm=n_perm)
    floor = 0.5 + float(chance_tol)
    ceiling = max(floor, post["null_decodability_q95"] or floor)
    removed = None if post["decodability"] is None else bool(post["decodability"] <= ceiling)
    carried = None if pre["decodability"] is None else bool(pre["decodability"] > ceiling)
    return {"ran": True, "layer": int(layer), "n": int(n),
            "pre_ablation_auroc": pre["auroc"], "post_ablation_auroc": post["auroc"],
            "pre_ablation_decodability": pre["decodability"],
            "post_ablation_decodability": post["decodability"],
            "chance_tol": float(chance_tol), "chance_ceiling": float(ceiling),
            "null_decodability_mean": post["null_decodability_mean"],
            "null_decodability_q95": post["null_decodability_q95"], "n_perm": int(n_perm),
            "information_removed": removed,
            "layer_carried_label": carried,
            "behavioural_null_interpretable": removed,
            "guard": ("post-ablation decodability must be at or below chance_ceiling for a behavioural "
                      "null at this layer to be reportable as a CAUSAL null; otherwise the ablation "
                      "removed nothing and the null is uninterpretable. If layer_carried_label is "
                      "false the layer never carried the label and the whole cell is void.")}


def _strip_trials(section, keep):
    """Drop the per-trial records from an output section unless ``--per-trial`` kept them.

    They are kept by DEFAULT (a statistics workstream needs them, and re-running costs GPU hours);
    ``--no-per-trial`` is for smoke runs where the JSON would otherwise dwarf its own summary."""
    if keep or not isinstance(section, dict):
        return section
    section.pop("trials", None)
    for row in section.get("rates", []) or []:
        if isinstance(row, dict):
            row.pop("trials", None)
    for row in section.get("ranks", []) or []:
        if isinstance(row, dict):
            row.pop("trials", None)
    return section


def main():
    ap = argparse.ArgumentParser(description="Causal interventions: patch / ablate / steer / subspace (RQ 1.2).")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--instructed", action="store_true")
    ap.add_argument("--mode", choices=["patch", "ablate", "steer", "subspace", "all"], default="all",
                    help="`all` is patch+ablate+steer, exactly as before. `subspace` (the rank-k "
                         "ablation of D3) is deliberately NOT in `all`: it costs len(--subspace-ranks) "
                         "eval passes plus the INLP fit, so it is opt-in.")
    ap.add_argument("--task", choices=["single", "multi", "codebook", "infercode"], default="single",
                    help="the game family the interventions RUN on. `codebook`/`infercode` mirror "
                         "identify_probe.py's --task choices, because those are the tasks the mixed-fit "
                         "directions are fit on — intervening on `single` with a codebook-fit direction "
                         "is a cross-task test (D2) and is now refused unless --allow-cross-task.")
    ap.add_argument("--model-id", default=None, help="base model: alias 1b/3b/8b or a full HF id (default: 1b)")
    # NOTE: the layer defaults below are tuned for the 16-layer 1B. On 3B (28) / 8B (32) rescale them
    # (e.g. proportionally) via the --*-layer flags — they are not auto-scaled to --model-id.
    ap.add_argument("--dir-layer", type=int, default=11, help="layer to extract the truth direction from")
    ap.add_argument("--ablate-layer", type=int, default=11)
    ap.add_argument("--steer-layer", type=int, default=13)
    ap.add_argument("--patch-layer", type=int, default=13)
    ap.add_argument("--patch-pos", type=int, default=-1,
                    help="token position patched (ally->rival); -1 = answer slot, e.g. -3 = context-aware non-final")
    ap.add_argument("--load-direction", default=None,
                    help="path to an identify_probe.py --save-probe JSON; use its MIXED-fit direction at "
                         "--dir-layer for ablate/steer instead of estimating an ally-fit axis. Overrides "
                         "--axis. This is the only way to causally test the direction the paper proposes. "
                         "In --mode subspace it SEEDS the INLP basis (rank 1 = exactly that direction).")
    ap.add_argument("--allow-cross-task", action="store_true",
                    help="permit --load-direction from a probe fit on a DIFFERENT --task. Off by default: "
                         "a cross-task null is confounded with the measured collapse of frozen "
                         "cross-task transfer, so it must be an explicit choice (D2).")
    ap.add_argument("--axis", choices=["lr", "diffmeans"], default="lr",
                    help="truth-direction estimator for ablate/steer: lr probe (correlate) or "
                         "diffmeans = mean(true=1)-mean(true=0) (causal). No effect on patch.")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--steer-alphas", default=None,
                    help="comma-separated α list (e.g. '5,10,20,40'); runs a steering α-sweep at "
                         "--steer-layer (records steering_sweep) instead of the single --alpha. Pair with "
                         "--dir-layer == --steer-layer to steer at the layer the direction was extracted from.")
    ap.add_argument("--steer-mode", choices=["unsigned", "signed", "both"], default="unsigned",
                    help="unsigned (default, historical) = same +α·d on every trial, which can CANCEL "
                         "across true bits (D1). signed = ORACLE steer, +α·d when truth==1 and -α·d when "
                         "truth==0. both = run the pair on the same eval episodes; that contrast is the "
                         "result. A signed rate alone is NOT 'we steered the model honest' — it uses the "
                         "label to pick the sign.")
    ap.add_argument("--norm-n", type=int, default=64,
                    help="episodes used to measure the median residual L2 norm at --steer-layer, so α is "
                         "reported calibrated (alpha_rel = α / median‖resid‖). 0 = skip the measurement "
                         "and report alpha_rel as null.")
    ap.add_argument("--subspace-ranks", default="1,2,4,8,16,32",
                    help="--mode subspace: comma-separated k sweep. The subspaces are NESTED prefixes of "
                         "one INLP fit at max(k), which is what makes the curve a dose-response in rank.")
    ap.add_argument("--subspace-n", type=int, default=None,
                    help="--mode subspace: activations for the INLP fit (default: --calib-n)")
    # --- THE PRE-CHECKPOINT COST FIX (added 2026-08-18). -----------------------------
    ap.add_argument("--collect-batch", type=int, default=1, metavar="B",
                    help="batch the activation-collection forward passes. DEFAULT 1 = the original "
                         "per-episode loop, byte-identical. B>1 is the fix for the real cost of this "
                         "stage: the ~4000-episode fit collection is ~70 min at batch 1 and runs at a GPU "
                         "utilization low enough to trip the <20%%-for-2h reaper, which is what actually "
                         "killed three subspace jobs (NOT the INLP fit, which costs ~25 s). Silently "
                         "ignored for the hooked positive-control collection, whose hooks index batch row "
                         "0 and position -1 and are therefore batch-unsafe.")
    ap.add_argument("--collect-batch-check-ref", type=int, default=1, metavar="R",
                    help="reference batch size the gate compares against (default 1, the historical "
                         "per-episode path). Set R>1 to compare two BATCHED configurations against each "
                         "other: neither is privileged, so whatever they disagree by is the INTRINSIC "
                         "floor of this comparison under bf16 reduction-order differences. That is how you "
                         "find out whether a gate threshold is measuring a bug or is simply set below the "
                         "noise floor — by measurement, instead of by relaxing it until it passes.")
    ap.add_argument("--collect-batch-check", type=int, default=0, metavar="M",
                    help="EQUIVALENCE GATE. Before doing any real work, collect M episodes BOTH ways "
                         "(batched and per-episode, same seed) and print the max |delta| plus the label "
                         "agreement, then exit non-zero if they disagree beyond tolerance. Run this once "
                         "per model/dtype before trusting --collect-batch on a science run: padding side, "
                         "position_ids and last-real-token indexing are all things that fail silently and "
                         "produce activations that look plausible and mean nothing.")
    ap.add_argument("--positive-control", choices=["auto", "on", "off"], default="auto",
                    help="D4: refit a probe on the POST-ablation activations and emit "
                         "post_ablation_auroc + behavioural_null_interpretable. auto = ON for --mode "
                         "subspace (a new mode, so no legacy default to preserve) and OFF for the rank-1 "
                         "--mode ablate (whose default must keep reproducing the reported numbers). "
                         "Pass `on` for any ablation null you intend to report.")
    ap.add_argument("--control-n", type=int, default=400,
                    help="episodes per side of the positive control (it runs the layer twice, clean and "
                         "ablated, on the SAME episodes). Push this toward the residual width — at n<d a "
                         "refit probe overfits and the control is valid but blunt.")
    ap.add_argument("--control-chance-tol", type=float, default=0.1,
                    help="FLOOR for the chance band: decodability <= 0.5+tol counts as removed. The band "
                         "actually used is max(that, the permutation null's 95th percentile), which is "
                         "what makes the verdict valid when n<d — see probe_decodability.")
    ap.add_argument("--control-perm", type=int, default=20,
                    help="label permutations used to calibrate the positive control's chance band (0/1 = "
                         "fall back to the fixed 0.5+tol band, only defensible when n >> d). Cheap: they "
                         "refit on cached activations, no extra forward passes.")
    ap.add_argument("--calib-n", type=int, default=300, help="samples for truth-direction calibration")
    ap.add_argument("--n", type=int, default=100, help="ablation/steering eval trials")
    ap.add_argument("--patch-n", type=int, default=200)
    ap.add_argument("--no-per-trial", action="store_true",
                    help="omit the per-trial records from the output JSON (they are kept by default so a "
                         "later stratification needs no GPU re-run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/interventions.json")
    args = ap.parse_args()

    model, tokenizer, device, instruction, label = instructed_mod.load_arm(
        args.adapter, args.instructed, model_id=args.model_id)

    if args.collect_batch_check:
        # Runs BEFORE any real work, so a mismatch costs one short job instead of a wrong result. The
        # tolerance is loose on purpose: batched and unbatched matmuls differ in reduction order, so bf16
        # activations of norm ~100-900 will not agree bit-for-bit. What must agree is the LABELS (identical
        # episode stream) and the activations to well inside the effect sizes this project reports.
        m = int(args.collect_batch_check)
        b = max(2, int(args.collect_batch))
        ref = max(1, int(args.collect_batch_check_ref))
        print(f"[{label}] === COLLECT-BATCH EQUIVALENCE GATE: m={m} episodes, batch={b} vs {ref}, "
              f"L{args.dir_layer}, task={args.task} ===")
        if ref > 1:
            print(f"[{label}] NOTE: reference batch is {ref}, not 1 — this run measures the INTRINSIC "
                  f"floor between two equally-valid batched passes, not agreement with history.")
        Xb, yb = collect_layer_activations(model, tokenizer, device, instruction, args.dir_layer,
                                           m, args.seed, args.task, force_ally=True, batch=b)
        X1, y1 = collect_layer_activations(model, tokenizer, device, instruction, args.dir_layer,
                                           m, args.seed, args.task, force_ally=True, batch=ref)
        lab_ok = bool(np.array_equal(yb, y1))
        print(f"[{label}] labels identical: {lab_ok} | shapes {Xb.shape} vs {X1.shape}")
        if Xb.shape != X1.shape:
            print(f"[{label}] GATE FAIL — shape mismatch, nothing else is worth computing.")
            raise SystemExit(3)

        # ⛔ THE FIRST VERSION OF THIS GATE USED max|delta| / mean|X| AND FAILED ON A GOOD RESULT.
        # That is not a relative error: it divides a MAXIMUM elementwise deviation by a MEAN absolute
        # value. Residual streams carry a handful of outlier coordinates orders of magnitude above the
        # typical one (mean|X| was 4.3 while the deviating coordinate sits in the hundreds), so the ratio
        # was ~0.7 for a deviation that is under one bf16 ULP at that coordinate's magnitude. Replaced with
        # per-row RELATIVE L2, which is the standard quantity, plus the diagnostic that settles the
        # outlier story directly: the magnitude of X at the argmax of |delta|.
        D = np.abs(Xb - X1)
        dmax = float(D.max())
        i, j = np.unravel_index(np.argmax(D), D.shape)
        relL2 = np.linalg.norm(Xb - X1, axis=1) / np.maximum(np.linalg.norm(X1, axis=1), 1e-30)
        num = (Xb * X1).sum(axis=1)
        den = np.linalg.norm(Xb, axis=1) * np.linalg.norm(X1, axis=1)
        cos = num / np.where(den == 0, 1, den)
        print(f"[{label}] max|delta| {dmax:.6g} at coord (row {i}, dim {j}) where "
              f"|X1| = {abs(float(X1[i, j])):.6g}  <- if this is huge, the delta is sub-ULP there")
        print(f"[{label}] per-row relative L2: max {relL2.max():.3e} mean {relL2.mean():.3e}")
        print(f"[{label}] per-row cosine:      min {cos.min():.8f} mean {cos.mean():.8f}")

        # THE DECISION-RELEVANT CHECK, and it is new rather than a loosened threshold. Nothing downstream
        # consumes raw activations; everything fits a LINEAR PROBE on them. So compare the probes: the
        # AUROC each gives, and the angle between the two fitted directions. If those agree, batching
        # cannot change a reported number regardless of elementwise noise.
        a1, a2 = probe_auroc(X1, y1, seed=0), probe_auroc(Xb, yb, seed=0)
        c1 = LogisticRegression(max_iter=1000).fit(X1, y1).coef_[0]
        c2 = LogisticRegression(max_iter=1000).fit(Xb, yb).coef_[0]
        dcos = float(np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2)))
        # probe_auroc returns None when a class is too thin to split — at small m that is a real
        # possibility, and it must FAIL the gate rather than be coerced to a number.
        auroc_ok = a1 is not None and a2 is not None and abs(a1 - a2) < 5e-3
        if a1 is None or a2 is None:
            print(f"[{label}] ⚠️ probe AUROC unavailable (unbatched={a1}, batched={a2}) — raise "
                  f"--collect-batch-check so both classes can be split; NOT treating this as a pass.")
        else:
            print(f"[{label}] probe AUROC unbatched {a1:.6f} vs batched {a2:.6f} "
                  f"(delta {abs(a1 - a2):.2e})")
        print(f"[{label}] cos(fitted direction_unbatched, direction_batched) = {dcos:.8f}")

        # ⛔ THE RELATIVE-L2 BOUND IS 3e-2, AND THE NUMBER IS MEASURED RATHER THAN CHOSEN.
        # The first version of this gate used 1e-2 and failed. That threshold was set below the achievable
        # floor: running the gate with --collect-batch-check-ref 8, i.e. comparing two equally-valid BATCHED
        # passes that share the padding path entirely, gives max per-row relative L2 = 1.211e-2 — the same
        # magnitude as batch-32-vs-unbatched (1.285e-2). So ~1.2e-2 is the intrinsic bf16 reduction-order
        # disagreement on a 42-layer model, and NO implementation can pass 1e-2, including the batched path
        # compared against itself. 3e-2 is ~2.5x that measured floor.
        # ⚠️ The other four criteria are UNCHANGED from the pre-registration and all passed as stated; only
        # the bound that was arithmetically impossible was re-derived, and it was re-derived from a control
        # run rather than by relaxing it until the result went green. If you widen it again, run the
        # ref-batch control first and quote the floor you measured, or you are fitting the gate to the data.
        # 🔑 The absolute deviation concentrates in ONE residual dimension (504 on Gemma-2-9B, |X| ~ 219-268
        # there) in both comparisons — an outlier/"massive activation" coordinate. That is why an elementwise
        # max is the wrong summary here and the probe-level checks below are the ones that decide.
        REL_L2_MAX, FLOOR_NOTE = 3e-2, "measured floor 1.21e-2 (batch 32 vs 8, same padding path)"
        ok = (lab_ok and float(relL2.max()) < REL_L2_MAX and float(cos.min()) > 0.9999
              and auroc_ok and dcos > 0.99)
        print(f"[{label}] GATE {'PASS' if ok else 'FAIL'} (require: labels identical; max per-row "
              f"relative L2 < {REL_L2_MAX:g} [{FLOOR_NOTE}]; min cosine > 0.9999; |dAUROC| < 5e-3; "
              f"cos(direction) > 0.99)")
        raise SystemExit(0 if ok else 3)

    per_trial = not args.no_per_trial
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    # NOTE: `steer_mode` is deliberately NOT recorded here. A top-level "steer_mode": "unsigned" on a
    # --mode ablate run would describe a steer that never happened; each steering section carries its own.
    results = {"arm": label, "mode": args.mode, "task": args.task, "axis": args.axis,
               "per_trial": per_trial, "seed": args.seed}

    # INCREMENTAL WRITE. Added 2026-08-17 after the GPU-utilization reaper evicted three `interv` jobs and
    # every one lost its whole run: `identify_probe.py` and `instrpair_probe.py` both checkpoint per unit of
    # work, this path wrote once at the very end. `STAGE=interv` is CPU-bound in sklearn (the INLP fit and
    # the positive control's probe fits) so it idles the GPU and is exactly the shape the reaper kills.
    # `partial` is True until the final write, so a consumer can tell a checkpoint from a finished run and
    # never mistakes a truncated rank sweep for a complete one.
    _out = Path(args.out)
    _out.parent.mkdir(parents=True, exist_ok=True)

    def _flush(done=False):
        """Checkpoint `results` to disk. Cheap relative to a forward pass; called after each block."""
        results["partial"] = not done
        _out.write_text(json.dumps(results, indent=2))

    _flush()

    if args.mode in ("patch", "all"):
        res = run_patching(model, tokenizer, device, instruction, args.patch_layer,
                           args.patch_n, args.seed + 100, args.task, pos=args.patch_pos)
        results["patching"] = _strip_trials(
            {"layer": args.patch_layer, "pos": args.patch_pos, "task": args.task,
             "flip_rate": res["flip_rate"], "n": res["n"], "by_truth": res["by_truth"],
             "trials": res["trials"]}, per_trial)
        print(f"[{label}] patch L{args.patch_layer} @pos{args.patch_pos} (task={args.task}): "
              f"rival→truth flip rate {res['flip_rate']:.4f}")
        _flush()

    # The EFFECTIVE axis, for every log line and every JSON field below. A past bug printed `(lr)` on a
    # --load-direction run because it used args.axis; a log line must never misreport what actually ran.
    axis_label, axis_source, dir_meta = args.axis, "estimated", None
    direction = None
    if args.mode in ("ablate", "steer", "subspace", "all"):
        if args.load_direction:
            direction, dir_meta = load_saved_direction(args.load_direction, args.dir_layer, device,
                                                       expect_dim=hidden_size)
            fit_task = check_direction_task(dir_meta, args.task, args.allow_cross_task,
                                            args.load_direction)
            axis_label = f"loaded-mixed@L{args.dir_layer}"
            axis_source = "loaded"
            results["axis"] = f"loaded:{Path(args.load_direction).name}@L{args.dir_layer}"
            results["direction_provenance"] = dir_meta
            results["direction_fit_task"] = fit_task
            results["cross_task"] = bool(fit_task is not None and fit_task != args.task)
            print(f"[{label}] using LOADED direction from {args.load_direction} @L{args.dir_layer} "
                  f"(mixed-fit, fit on task={fit_task}); --axis is ignored")
        elif args.mode != "subspace":
            direction = extract_truth_direction(model, tokenizer, device, instruction,
                                                args.dir_layer, args.calib_n, args.seed, args.task,
                                                axis=args.axis)
    results["axis_effective"], results["axis_source"] = axis_label, axis_source

    if args.mode in ("ablate", "all"):
        res = run_ablation(model, tokenizer, device, instruction, direction,
                           args.ablate_layer, args.n, args.seed + 200, args.task)
        section = {"dir_layer": args.dir_layer, "layer": args.ablate_layer, "task": args.task,
                   "axis": axis_label, "axis_source": axis_source, "rank": 1,
                   "ally_truth_rate": res["truth_rate"], "n": res["n"],
                   "by_truth": res["by_truth"], "trials": res["trials"]}
        print(f"[{label}] ablate L{args.ablate_layer} rank1 ({axis_label}, task={args.task}): "
              f"ally truth rate {res['truth_rate']:.4f}")
        # D4: the rank-1 default stays OFF so the reported numbers keep reproducing, but a null without
        # the control is not a result and the log must say so rather than let it read like one.
        if args.positive_control == "on":
            ctl = run_positive_control(model, tokenizer, device, instruction, args.ablate_layer,
                                       _ablation_factory(direction), args.control_n, args.seed + 400,
                                       args.task, chance_tol=args.control_chance_tol,
                                       n_perm=args.control_perm)
            section["positive_control"] = ctl
            print(f"[{label}] ablate L{args.ablate_layer} POSITIVE CONTROL: truth AUROC "
                  f"{ctl['pre_ablation_auroc']} → {ctl['post_ablation_auroc']} | decodability "
                  f"{ctl['pre_ablation_decodability']} → {ctl['post_ablation_decodability']} "
                  f"(chance ceiling {ctl['chance_ceiling']:.3f}, perm-null q95 "
                  f"{ctl['null_decodability_q95']}); information_removed="
                  f"{ctl['information_removed']} → behavioural null "
                  f"{'INTERPRETABLE' if ctl['behavioural_null_interpretable'] else 'UNINTERPRETABLE'}")
            if ctl["layer_carried_label"] is False:
                print(f"[{label}] ⚠️ L{args.ablate_layer} did NOT carry the truth label above the "
                      f"permutation null even BEFORE ablation — this whole cell is void, not a null.")
        else:
            section["positive_control"] = {
                "ran": False, "information_removed": None, "behavioural_null_interpretable": None,
                "guard": "no positive control was run; this ablation null is NOT reportable as a causal null."}
            print(f"[{label}] ⚠️ ablate L{args.ablate_layer} ran WITHOUT the D4 positive control "
                  f"(--positive-control {args.positive_control}): a truth rate of "
                  f"{res['truth_rate']:.4f} does NOT establish a causal null, because nothing here shows "
                  f"the ablation removed the information. Re-run with --positive-control on to report it.")
        results["ablation"] = _strip_trials(section, per_trial)
        _flush()

    if args.mode == "subspace":
        # D3: rank-k ablation. One INLP fit at max(k); the k-subspaces are its nested prefixes.
        ranks = sorted({int(r) for r in args.subspace_ranks.split(",") if r.strip() != ""})
        if not ranks or min(ranks) < 1:
            raise SystemExit(f"[interventions] --subspace-ranks {args.subspace_ranks!r} is empty/invalid.")
        fit_n = args.subspace_n or args.calib_n
        if args.axis == "diffmeans" and max(ranks) > 1:
            # Residualizing the class-mean difference sets the NEXT iteration's class-mean difference to
            # exactly zero, so diffmeans-INLP terminates at rank 1 by construction. Say so before
            # burning the fit rather than letting the run report a "rank sweep" that is one point.
            print(f"[{label}] ⚠️ --axis diffmeans cannot exceed rank 1 under INLP (residualizing the "
                  f"class-mean difference zeroes it by construction), so ranks above 1 in "
                  f"{args.subspace_ranks!r} WILL be skipped. Use --axis lr for the rank sweep.")
        supplied = None
        if args.load_direction:
            supplied, sup_meta = load_saved_subspace(args.load_direction, args.dir_layer,
                                                     expect_dim=hidden_size)
            print(f"[{label}] seeding the INLP basis with {sup_meta['supplied_rows']} SUPPLIED "
                  f"direction(s) from {args.load_direction} @L{args.dir_layer}; ranks above that are "
                  f"topped up by local INLP fits ({args.axis})")
        X, y = collect_layer_activations(model, tokenizer, device, instruction, args.dir_layer,
                                        fit_n, args.seed, args.task, force_ally=True,
                                        batch=args.collect_batch)
        basis_np, inlp_meta = fit_inlp_subspace(X, y, max(ranks), axis=args.axis, supplied=supplied)
        print(f"[{label}] INLP subspace @L{args.dir_layer} ({axis_label}, task={args.task}, n={fit_n}): "
              f"requested k={max(ranks)}, achieved rank {inlp_meta['achieved_rank']} "
              f"(source {inlp_meta['basis_source']}, orthonormality err "
              f"{inlp_meta['orthonormality_err']:.2e}"
              f"{', STOPPED EARLY: ' + inlp_meta['stopped_early'] if inlp_meta['stopped_early'] else ''})")
        if inlp_meta["rank_deficient"]:
            print(f"[{label}] ⚠️ the basis is RANK-DEFICIENT w.r.t. the request "
                  f"({inlp_meta['achieved_rank']} < {max(ranks)}): report achieved_rank, never k. This "
                  f"is the EXPECTED outcome — a logistic fit removes most of the linear signal in one "
                  f"hyperplane, so the termination rank is itself the dimensionality estimate (D3).")
        if inlp_meta["underdetermined"]:
            print(f"[{label}] ⚠️ the INLP fit is UNDERDETERMINED (n={inlp_meta['n_samples']} < "
                  f"d={inlp_meta['n_features']}): a logistic fit separates any labelling at these n, so "
                  f"the later directions are partly noise and the behaviour-vs-k curve understates. "
                  f"Raise --subspace-n toward d before reporting the curve.")
        basis_t = torch.tensor(basis_np, dtype=model_mod.model_dtype(device), device=device)

        rank_rows = []
        for k in ranks:
            if k > inlp_meta["achieved_rank"]:
                print(f"[{label}] skipping k={k}: the basis only reaches rank "
                      f"{inlp_meta['achieved_rank']}. Refusing to run a 'k={k}' cell that would ablate "
                      f"fewer dimensions than its label claims.")
                continue
            bk = basis_t[:k]
            stats = _basis_stats(basis_np[:k])
            res = run_subspace_ablation(model, tokenizer, device, instruction, bk, k,
                                        args.ablate_layer, args.n, args.seed + 200, args.task)
            row = {"k": k, "achieved_rank": stats["rank"],
                   "orthonormality_err": stats["orthonormality_err"],
                   "ally_truth_rate": res["truth_rate"], "n": res["n"],
                   "by_truth": res["by_truth"], "trials": res["trials"]}
            # auto = ON here: `subspace` is a new mode, so there is no legacy default to protect, and a
            # rank sweep whose ablations were never verified to remove anything says nothing about rank.
            if args.positive_control in ("auto", "on"):
                ctl = run_positive_control(model, tokenizer, device, instruction, args.ablate_layer,
                                           _subspace_factory(bk, k), args.control_n, args.seed + 400,
                                           args.task, chance_tol=args.control_chance_tol,
                                           n_perm=args.control_perm)
                row["positive_control"] = ctl
                print(f"[{label}] subspace-ablate L{args.ablate_layer} k={k} (rank {stats['rank']}, "
                      f"{axis_label}, task={args.task}): ally truth rate {res['truth_rate']:.4f} | "
                      f"truth decodability {ctl['pre_ablation_decodability']} → "
                      f"{ctl['post_ablation_decodability']} (ceiling {ctl['chance_ceiling']:.3f}) | "
                      f"null {'INTERPRETABLE' if ctl['behavioural_null_interpretable'] else 'UNINTERPRETABLE'}")
                if ctl["layer_carried_label"] is False:
                    # Same guard as the rank-1 branch: "information_removed" is trivially true when the
                    # layer never carried the label, and that must not read like a passed control.
                    print(f"[{label}] ⚠️ k={k}: L{args.ablate_layer} did NOT carry the truth label above "
                          f"the permutation null even BEFORE ablation — this cell is VOID, not a null.")
            else:
                row["positive_control"] = {
                    "ran": False, "information_removed": None,
                    "behavioural_null_interpretable": None,
                    "guard": "no positive control was run; this cell says nothing about rank."}
                print(f"[{label}] subspace-ablate L{args.ablate_layer} k={k} (rank {stats['rank']}, "
                      f"{axis_label}, task={args.task}): ally truth rate {res['truth_rate']:.4f} "
                      f"| ⚠️ NO positive control (--positive-control off)")
            rank_rows.append(_strip_trials(row, per_trial))
            # Checkpoint PER RANK. This is the block the reaper actually killed: the INLP fit plus one
            # positive control per k is the long CPU-bound stretch, and losing a 5-rank sweep at k=16
            # because k=32 was still running is the exact failure that cost three jobs on 2026-08-17.
            # A partial rank curve is a usable result; an empty directory is not.
            results["subspace_ablation"] = {
                "dir_layer": args.dir_layer, "layer": args.ablate_layer, "task": args.task,
                "axis": axis_label, "axis_source": axis_source, "fit_n": fit_n,
                "requested_ranks": ranks, "inlp": inlp_meta, "ranks": rank_rows,
                "ranks_completed": [r["k"] for r in rank_rows],
                "ranks_outstanding": [k2 for k2 in ranks if k2 not in [r["k"] for r in rank_rows]]}
            _flush()
        results["subspace_ablation"] = {"dir_layer": args.dir_layer, "layer": args.ablate_layer,
                                        "task": args.task, "axis": axis_label,
                                        "axis_source": axis_source, "fit_n": fit_n,
                                        "requested_ranks": ranks, "inlp": inlp_meta,
                                        "ranks": rank_rows,
                                        "ranks_completed": [r["k"] for r in rank_rows],
                                        "ranks_outstanding": []}
        _flush()

    if args.mode in ("steer", "all"):
        # α calibration (D1): a bare α is not comparable across layers or models, so measure the scale
        # of the thing we are adding to. Measured on rival contexts — the contexts steering runs on.
        norm_med = None
        if args.norm_n > 0:
            norm_med = measure_residual_norm(model, tokenizer, device, instruction, args.steer_layer,
                                             args.norm_n, args.seed + 500, args.task, force_ally=False)
            print(f"[{label}] median ‖resid‖₂ at L{args.steer_layer} = {norm_med:.3f} "
                  f"(n={args.norm_n}, rival contexts) — α is also reported as alpha_rel = α/‖resid‖")
        # α = 0 BASELINE. Added 2026-08-17: every steering number this project had reported was compared
        # against a rate measured in a DIFFERENT run, because this path recorded no unperturbed arm. That
        # makes "the steer changed nothing" an inference across runs rather than a within-run contrast.
        #
        # Run through run_steering with alpha=0.0 rather than by skipping the hook, deliberately: the hook
        # still fires and adds sign*0*d, so this controls for the whole harness — hook registration, the
        # same episodes (identical seed), the same generation path — and not merely for the value of α. Any
        # difference between this and an α>0 cell is therefore attributable to α alone.
        #
        # Computed ONCE, outside the mode loop: at α=0 the sign is multiplied by zero, so signed and
        # unsigned are the same measurement and running both would be two names for one number.
        base_r = run_steering(model, tokenizer, device, instruction, direction,
                              args.steer_layer, 0.0, args.n, args.seed + 300, args.task, signed=False)
        results["steering_baseline"] = _strip_trials(
            {"dir_layer": args.dir_layer, "layer": args.steer_layer, "task": args.task,
             "axis": axis_label, "axis_source": axis_source, "alpha": 0.0, "alpha_rel": 0.0,
             "steer_mode": "none (alpha=0 control)",
             "rival_truth_rate": base_r["truth_rate"], "n": base_r["n"],
             "by_truth": base_r["by_truth"], "trials": base_r["trials"],
             "meaning": ("unperturbed rival truth rate on the SAME episodes as every steered cell below, "
                         "measured through the same hook with alpha=0. Compare steered rates against THIS, "
                         "not against a behaviour run from another job.")}, per_trial)
        print(f"[{label}] steer L{args.steer_layer} α0 [BASELINE, hook live, no push] "
              f"({axis_label}, task={args.task}): rival truth rate {base_r['truth_rate']:.4f} "
              f"(truth0 {base_r['by_truth']['truth0_rate']}, truth1 {base_r['by_truth']['truth1_rate']}) "
              f"— every steered rate below is to be read against this number")
        _flush()

        modes = {"unsigned": [False], "signed": [True], "both": [False, True]}[args.steer_mode]
        for signed in modes:
            suffix = "_signed" if signed else ""
            tag = "signed/ORACLE" if signed else "unsigned"
            common = {"dir_layer": args.dir_layer, "layer": args.steer_layer, "task": args.task,
                      "axis": axis_label, "axis_source": axis_source,
                      "steer_mode": "signed" if signed else "unsigned",
                      "residual_norm_median": norm_med}
            if args.steer_alphas:
                # α-sweep: same direction + same eval episodes (seed fixed), vary only α, so the curve is
                # a clean read of steering strength. Disambiguates a null single-α steer (under-powered α
                # vs no linear steer exists). Pair --dir-layer == --steer-layer for a same-layer test.
                # The signed and unsigned passes share the seed too, so `both` is a PAIRED comparison.
                alphas = [float(a) for a in args.steer_alphas.split(",") if a.strip() != ""]
                rates = []
                for a in alphas:
                    r = run_steering(model, tokenizer, device, instruction, direction,
                                     args.steer_layer, a, args.n, args.seed + 300, args.task,
                                     signed=signed)
                    rates.append({"alpha": a, "alpha_rel": _alpha_rel(a, norm_med),
                                  "rival_truth_rate": r["truth_rate"], "n": r["n"],
                                  "by_truth": r["by_truth"], "trials": r["trials"]})
                    # Report the delta against the in-run α=0 arm, so an under-powered steer and a real
                    # null are distinguishable on the log line itself rather than only after a JSON diff.
                    _d = r["truth_rate"] - base_r["truth_rate"]
                    print(f"[{label}] steer L{args.steer_layer} α{a} [{tag}] ({axis_label}, "
                          f"task={args.task}): rival truth rate {r['truth_rate']:.4f} "
                          f"(vs α0 baseline {base_r['truth_rate']:.4f}, Δ {_d:+.4f}) "
                          f"(truth0 {r['by_truth']['truth0_rate']}, truth1 {r['by_truth']['truth1_rate']}"
                          f"{f', α/‖resid‖={a / norm_med:.3f}' if norm_med else ''})")
                    rates[-1]["delta_vs_baseline"] = _d
                results[f"steering_sweep{suffix}"] = _strip_trials({**common, "rates": rates}, per_trial)
                _flush()
            else:
                r = run_steering(model, tokenizer, device, instruction, direction,
                                 args.steer_layer, args.alpha, args.n, args.seed + 300, args.task,
                                 signed=signed)
                results[f"steering{suffix}"] = _strip_trials(
                    {**common, "alpha": args.alpha, "alpha_rel": _alpha_rel(args.alpha, norm_med),
                     "rival_truth_rate": r["truth_rate"], "n": r["n"], "by_truth": r["by_truth"],
                     "delta_vs_baseline": r["truth_rate"] - base_r["truth_rate"],
                     "trials": r["trials"]}, per_trial)
                print(f"[{label}] steer L{args.steer_layer} α{args.alpha} [{tag}] ({axis_label}, "
                      f"task={args.task}): rival truth rate {r['truth_rate']:.4f} "
                      f"(vs α0 baseline {base_r['truth_rate']:.4f}, "
                      f"Δ {r['truth_rate'] - base_r['truth_rate']:+.4f}) "
                      f"(truth0 {r['by_truth']['truth0_rate']}, truth1 {r['by_truth']['truth1_rate']}"
                      f"{f', α/‖resid‖={args.alpha / norm_med:.3f}' if norm_med else ''})")
                _flush()
        if args.steer_mode == "both":
            # State the D1 contrast in the log so nobody has to reconstruct it from two JSON keys — and
            # so the oracle caveat travels with the number.
            key = "steering_sweep" if args.steer_alphas else "steering"
            g = lambda k: results.get(k, {})
            uns = ([row["rival_truth_rate"] for row in g(key).get("rates", [])]
                   if args.steer_alphas else [g(key).get("rival_truth_rate")])
            sgn = ([row["rival_truth_rate"] for row in g(key + "_signed").get("rates", [])]
                   if args.steer_alphas else [g(key + "_signed").get("rival_truth_rate")])
            print(f"[{label}] D1 CONTRAST at L{args.steer_layer}: α0 baseline "
                  f"{base_r['truth_rate']:.4f} | unsigned {uns} vs signed/oracle {sgn} "
                  f"— signed >> unsigned means a symmetric causal axis was cancelling in the pooled "
                  f"unsigned rate; both ~equal and low means no linear steer exists here. ⚠️ Read BOTH "
                  f"against the α0 baseline: rates that merely equal the baseline are a null, not an "
                  f"effect, and at a baseline of 0.0000 only an INCREASE was ever detectable (a floor). "
                  f"The signed number is an ORACLE result and is not 'steering the model into honesty'.")

    _flush(done=True)
    print(f"Wrote {_out}")


if __name__ == "__main__":
    main()
