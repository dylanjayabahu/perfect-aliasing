"""Exp 001 debug — why is the 0/1 readout at chance while a probe decodes the truth at ~0.8?

Hypothesis: we score `logits[0,-1,:]` at the generation-prompt position and compare only the bare
`"0"` / `"1"` token ids, but the model's actual next token there is something else (a leading space,
the space-prefixed ` 0`/` 1` variant, or a preamble), so the compare is between two low, noisy logits
even though the answer is computed internally.

This dumps, per episode: the true bit, the top-k next-token predictions at the scored position (with
`repr()` so whitespace is visible), the logits of the bare vs space-prefixed digit tokens, and a short
greedy generation — then tallies which readout recovers the truth. READ-ONLY; no fix.

    python src/perfect_aliasing/diagnose_readout.py                 # base model, honest framing
    python src/perfect_aliasing/diagnose_readout.py --adapter <dir>  # the RL model
    python src/perfect_aliasing/diagnose_readout.py --instructed     # base + deception directive

Capability-floor check (does the base model retrieve+emit the bit *at all*, with no reason to lie?):

    python src/perfect_aliasing/diagnose_readout.py --model-id 3b --force-ally --n 50
"""
import argparse
import random

import torch

import game as game_mod
import model as model_mod
import instructed as instructed_mod


def first_id(tokenizer, s):
    return tokenizer.encode(s, add_special_tokens=False)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--instructed", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--gen-tokens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=["single", "multi"], default="single")
    ap.add_argument("--model-id", default=None, help="base model: alias 1b/3b/8b or a full HF id (default: 1b)")
    ap.add_argument("--force-ally", action="store_true",
                    help="only sample ally (honest) contexts — the clean capability-floor check")
    args = ap.parse_args()

    model, tok, device, instruction, label = instructed_mod.load_arm(
        args.adapter, args.instructed, model_id=args.model_id)
    rng = random.Random(args.seed)

    t0, t1 = first_id(tok, "0"), first_id(tok, "1")
    sp0, sp1 = first_id(tok, " 0"), first_id(tok, " 1")
    resolved = model_mod.resolve_model_id(args.model_id)
    force_ally = True if args.force_ally else None
    print(f"=== model={resolved} arm={label} force_ally={bool(args.force_ally)} | "
          f"bare ids 0={t0} 1={t1} | space ids ' 0'={sp0} ' 1'={sp1} ===")

    tally = {"bare argmax(0,1)": 0, "space argmax( 0, 1)": 0, "greedy first digit": 0}
    total = 0
    for i in range(args.n):
        ep = game_mod.sample_episode(rng, task=args.task, force_ally=force_ally)
        messages = game_mod.build_messages(
            ep.game, ep.values, ep.target_variable, ep.role, ep.query_text, instruction=instruction
        )
        prompt = model_mod.render_prompt(tok, messages)
        inputs = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        top = torch.topk(probs, args.topk)
        top_str = ", ".join(
            f"{repr(tok.decode([idx]))}:{p:.3f}" for p, idx in zip(top.values.tolist(), top.indices.tolist())
        )

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.gen_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen_str = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        bare = 1 if logits[t1] > logits[t0] else 0
        space = 1 if logits[sp1] > logits[sp0] else 0
        gd = next((c for c in gen_str if c in "01"), None)
        greedy = int(gd) if gd is not None else None

        total += 1
        tally["bare argmax(0,1)"] += int(bare == ep.true_value)
        tally["space argmax( 0, 1)"] += int(space == ep.true_value)
        tally["greedy first digit"] += int(greedy == ep.true_value)

        print(f"\n[{i}] true={ep.true_value}  {'ally' if ep.is_ally else 'rival'}  var={ep.target_variable}")
        print(f"    top{args.topk}: {top_str}")
        print(f"    greedy gen: {repr(gen_str)}")
        print(f"    bare 0/1 logit {logits[t0]:.2f}/{logits[t1]:.2f}->{bare} | "
              f"space  0/ 1 logit {logits[sp0]:.2f}/{logits[sp1]:.2f}->{space} | greedy-digit={greedy}")

    print(f"\n=== readout accuracy over {total} episodes (arm={label}) ===")
    for k, v in tally.items():
        print(f"    {k:22s}: {v}/{total} = {v / max(total, 1):.3f}")


if __name__ == "__main__":
    main()
