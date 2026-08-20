"""DIRECT GEOMETRIC COMPARISON of the D2 directive rungs' activations (Exp-002, §refit).

WHAT THIS CLOSES. §refit currently concedes, in the paper's own words: *"cross-scoring shows one decoder
suffices for all 14 variants, so the refit spread is not necessary to explain the activations, but it does
not prove the underlying representations are identical, which would need a direct geometric comparison we
do not run."* This is that comparison. It asks whether the residual stream at the read position actually
differs across directive wordings, rather than whether a refitted probe's number moves.

THE DESIGN, and why each piece is load-bearing:

* **The episode stream is PAIRED.** Every rung is collected from ``random.Random(SEED)`` re-seeded to the
  same value, so trial *i* is the same game, the same values, the same role and the same query for every
  rung. The ONLY thing that differs between two rungs' prompts is the directive text. That turns the
  comparison into a paired one and removes episode variance entirely — without it, a difference between
  rungs is confounded with a difference between samples.

* **Every layer comes from ONE forward pass.** ``output_hidden_states=True`` returns all layers, so
  collecting L layers costs the same as collecting one. Calling a single-layer collector L times would
  multiply the GPU cost by L for no information.

* 🔑 **Three reference scales, because a similarity number alone is uninterpretable.** "CKA = 0.98" means
  nothing without knowing what CKA reads for two samples that are *definitionally* the same thing. This is
  the batch-gate lesson (a threshold must be derived from the quantity it bounds), applied up front rather
  than after a failed gate:
    1. **WITHIN-rung split-half** — split one rung's own trials in two and compute every statistic across
       the halves. This is the floor imposed by finite n at d ≫ n, which is the regime we are in.
    2. **NUMERICAL floor** — the reference rung is collected a SECOND time at a different batch size. The
       activations are mathematically identical and differ only by bf16 reduction order, so the paired
       relative-L2 between them is the numerical noise floor. (The batch gate measured ~1.2e-2 for
       Gemma-9B/L32; this measures it in-run rather than importing that number.)
    3. **BETWEEN-rung** — the quantity of interest, read against 1 and 2.

* **Two role arms, and the contrast is informative.** ``DIRECTIVES = ally_clause + rival_clause``, so the
  prompt text differs across rungs on ally trials too, even though the rival clause is not behaviourally
  operative there. Running both arms therefore decomposes the effect: a difference present on ally trials
  as well is the model *reading different text*; an excess on rival trials is the directive doing work.

⚠️ WHAT THIS CANNOT SHOW. Geometric similarity at one read position is not representational identity in
general, and n < d means every subspace estimate is noisy — which is exactly why reference (1) exists and
why no subspace number should be read without it. A null here bounds the claim; it does not prove sameness.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import game as game_mod
import instructed as instructed_mod
import interventions as iv_mod
import model as model_mod


def collect_multi(model, tokenizer, device, instruction, layers, n, seed, task, force_ally, batch):
    """Last-real-token activations at every layer in ``layers``, over ``n`` PAIRED episodes.

    Padding discipline is copied from ``interventions._collect_batched``, which is the cleared path: pad
    RIGHT so HF's default ``position_ids`` are correct, and read ``attention_mask.sum(1) - 1`` — the last
    REAL token — never ``-1``, which is a pad position for every row shorter than the longest in its batch.
    Reading ``-1`` would return padding and produce activations that look plausible and mean nothing."""
    rng = random.Random(seed)          # re-seeded per rung => identical episodes across rungs
    eps = [game_mod.sample_episode(rng, task=task, force_ally=force_ally) for _ in range(n)]
    prompts = [model_mod.render_prompt(
        tokenizer, iv_mod._trial_messages(task, ep, ep.role, instruction)) for ep in eps]

    prev_side, prev_pad = tokenizer.padding_side, tokenizer.pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    out = {l: [] for l in layers}
    try:
        for i in range(0, n, batch):
            enc = tokenizer(prompts[i:i + batch], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                res = model(**enc, output_hidden_states=True)
            last = enc["attention_mask"].sum(dim=1) - 1
            ar = torch.arange(last.shape[0], device=res.hidden_states[0].device)
            for l in layers:
                rows = res.hidden_states[l][ar, last, :]
                out[l].extend(rows.detach().cpu().to(torch.float32).numpy())
    finally:
        tokenizer.padding_side, tokenizer.pad_token = prev_side, prev_pad
    return {l: np.asarray(v) for l, v in out.items()}


# --- the statistics ---------------------------------------------------------------------------------
def _cka(X, Y):
    """Linear CKA on column-centred activations. 1.0 = the two sets span the same geometry up to
    rotation/scale; it is invariant to isotropic scaling, which is why it is reported beside the
    mean-shift (which is not)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = float(np.linalg.norm(Y.T @ X, "fro") ** 2)
    xx = float(np.linalg.norm(X.T @ X, "fro"))
    yy = float(np.linalg.norm(Y.T @ Y, "fro"))
    return xty / (xx * yy) if xx > 0 and yy > 0 else None


def _paired_rel_l2(X, Y):
    """Mean per-trial ‖x-y‖/‖x‖. THE most direct statistic here, and only meaningful because the episodes
    are paired: trial i is the same game under two directives, so this is the displacement the wording
    causes, with episode variance removed rather than averaged over."""
    num = np.linalg.norm(X - Y, axis=1)
    den = np.linalg.norm(X, axis=1)
    ok = den > 0
    return float(np.mean(num[ok] / den[ok])), float(np.max(num[ok] / den[ok]))


def _mean_shift(X, Y):
    """‖mean(X)-mean(Y)‖ normalised by the typical row norm. Catches a pure translation, which CKA is
    blind to by construction (it centres first) -- so the two are reported together, never one alone."""
    d = float(np.linalg.norm(X.mean(0) - Y.mean(0)))
    scale = float(np.mean(np.linalg.norm(X, axis=1)))
    return d / scale if scale > 0 else None


def _subspace_cos(X, Y, k):
    """Mean cosine of the principal angles between the top-``k`` PCA subspaces. 1.0 = the same k-dim
    subspace. ⚠️ n < d here, so this is noisy by construction; read it ONLY against the within-rung
    split-half reference computed at the same n."""
    def basis(A):
        A = A - A.mean(0, keepdims=True)
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        return vt[:k].T
    Ba, Bb = basis(X), basis(Y)
    sv = np.linalg.svd(Ba.T @ Bb, compute_uv=False)
    return float(np.mean(np.clip(sv, 0, 1)))


def compare(X, Y, k):
    mean_rel, max_rel = _paired_rel_l2(X, Y)
    return {"cka": _cka(X, Y), "paired_rel_l2_mean": mean_rel, "paired_rel_l2_max": max_rel,
            "mean_shift_rel": _mean_shift(X, Y), "subspace_cos_topk": _subspace_cos(X, Y, k)}


def main():
    ap = argparse.ArgumentParser(description="Direct geometric comparison of D2 directive rungs.")
    ap.add_argument("--model-id", default="8b")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="single")
    ap.add_argument("--layers", default="", help="comma-separated 1-indexed layers ('' = a sparse default)")
    ap.add_argument("--rungs", default="", help="comma-separated D2 rungs ('' = the whole ladder)")
    ap.add_argument("--reference", default="default", help="rung every other rung is compared against")
    ap.add_argument("--role", choices=["rival", "ally"], default="rival",
                    help="rival = the directive is behaviourally operative; ally = it is present in the "
                         "text but inert, which separates 'reads different text' from 'behaves differently'")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ref-batch", type=int, default=8,
                    help="the reference rung is collected a SECOND time at this batch size; the paired "
                         "difference between the two passes IS the bf16 numerical floor for this model")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", default="geom.json")
    args = ap.parse_args()

    rungs = ([r.strip() for r in args.rungs.split(",") if r.strip()]
             or list(instructed_mod.DIRECTIVE_LADDER))
    bad = [r for r in rungs + [args.reference] if r not in instructed_mod.DIRECTIVE_LADDER]
    if bad:
        raise SystemExit(f"[geom] unknown rung(s) {bad}; choose from {instructed_mod.DIRECTIVE_LADDER}")
    if args.reference not in rungs:
        rungs = [args.reference] + rungs
    force_ally = (args.role == "ally")

    # `load_arm` is the shared arm loader every other analysis entrypoint uses (it returns the resolved
    # instruction and label too); we take model/tokenizer/device from it and then swap the directive text
    # per rung with `instruction_for`, so the model is loaded ONCE for all 15 rungs.
    model, tokenizer, device, _instr, label = instructed_mod.load_arm(
        args.adapter, True, model_id=args.model_id, directive=args.reference)
    n_layers = model_mod.num_layers(model)
    layers = ([int(x) for x in args.layers.split(",") if x.strip()]
              or sorted({max(1, round(n_layers * f)) for f in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)}))
    print(f"[geom] model={args.model_id} n_layers={n_layers} layers={layers} role={args.role} "
          f"n={args.n} seed={args.seed} batch={args.batch} rungs={len(rungs)} ref={args.reference}")

    acts = {}
    for r in rungs:
        acts[r] = collect_multi(model, tokenizer, device, instructed_mod.instruction_for(True, r),
                                layers, args.n, args.seed, args.task, force_ally, args.batch)
        print(f"[geom]   collected {r}")
    # Reference rung a SECOND time at a different batch size -> the in-run numerical floor.
    ref2 = collect_multi(model, tokenizer, device, instructed_mod.instruction_for(True, args.reference),
                         layers, args.n, args.seed, args.task, force_ally, args.ref_batch)
    print(f"[geom]   collected {args.reference} AGAIN at batch={args.ref_batch} (numerical floor)")

    out = {"model_id": args.model_id, "adapter": args.adapter, "task": args.task, "role": args.role,
           "n": args.n, "seed": args.seed, "batch": args.batch, "ref_batch": args.ref_batch,
           "topk": args.topk, "reference": args.reference, "rungs": rungs, "layers": layers,
           "n_layers": n_layers, "per_layer": []}

    for l in layers:
        Xr = acts[args.reference][l]
        row = {"layer": l, "d": int(Xr.shape[1])}
        # (1) numerical floor: same rung, same episodes, different batch size.
        row["numerical_floor"] = compare(Xr, ref2[l], args.topk)
        # (2) within-rung split-half at the SAME n as every between-rung number. Halves are disjoint
        # episode sets, so the paired statistic is undefined for it -- recorded as None, never as 0.
        h = Xr.shape[0] // 2
        wr = compare(Xr[:h], Xr[h:2 * h], args.topk)
        # 🔴 The halves are DISJOINT episode sets, so any statistic requiring row alignment is undefined
        # here and is nulled rather than emitted as a number a consumer might treat as a floor. This
        # originally nulled only `paired_rel_l2` and left `cka` in place; the first run showed why that was
        # wrong (split-half CKA 0.017 vs between-rung 0.99), so `cka` is nulled too and the reason travels
        # with it in the JSON. `mean_shift` and `subspace_cos` are distributional and stay valid.
        wr["paired_rel_l2_mean"] = wr["paired_rel_l2_max"] = wr["cka"] = None
        wr["alignment_dependent_stats_null_reason"] = (
            "halves are disjoint episodes; cka and paired_rel_l2 require row-aligned inputs, so they are "
            "undefined as a within-condition reference. Use the numerical_floor row for those two.")
        row["within_rung_split_half"] = wr
        # (3) between-rung, the quantity of interest.
        row["between"] = {r: compare(Xr, acts[r][l], args.topk)
                          for r in rungs if r != args.reference}
        out["per_layer"].append(row)

    Path(args.out).write_text(json.dumps(out, indent=2))

    f = lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else "   n/a"
    print(f"\n=== geometry vs `{args.reference}` ({args.role} trials, n={args.n}) ===")
    for row in out["per_layer"]:
        nf, wr = row["numerical_floor"], row["within_rung_split_half"]
        print(f"\nL{row['layer']}  (d={row['d']})")
        print(f"  {'REFERENCE numerical floor':<28} cka {f(nf['cka'])} "
              f"paired_relL2 {f(nf['paired_rel_l2_mean'])} meanshift {f(nf['mean_shift_rel'])} "
              f"subcos {f(nf['subspace_cos_topk'])}")
        print(f"  {'REFERENCE within-rung n/2':<28} cka {f(wr['cka'])} "
              f"paired_relL2      n/a meanshift {f(wr['mean_shift_rel'])} "
              f"subcos {f(wr['subspace_cos_topk'])}")
        for r, m in sorted(row["between"].items(),
                           key=lambda kv: -(kv[1]["paired_rel_l2_mean"] or 0)):
            print(f"  {r:<28} cka {f(m['cka'])} paired_relL2 {f(m['paired_rel_l2_mean'])} "
                  f"meanshift {f(m['mean_shift_rel'])} subcos {f(m['subspace_cos_topk'])}")
    print(f"\n[geom] wrote {args.out}")
    # 🔴 CORRECTED 2026-08-19 AFTER THE FIRST RUN. The original wording here told the reader to compare
    # between-rung CKA against the within-rung split-half value. THAT IS WRONG, and the first run made it
    # obvious: split-half CKA came back 0.017-0.045 while every between-rung CKA was 0.77-0.999, i.e. the
    # supposed "floor" sat two orders of magnitude BELOW the measurements it was meant to bound.
    # Cause: CKA compares representational geometry over a COMMON set of inputs, so it needs ROW-ALIGNED
    # matrices. The split-half halves are different episodes, and for anisotropic data (a real residual
    # stream) the centred cross-product E[Y^T X] -> 0 when rows are independent, so the numerator only
    # fluctuates while the denominator is set by the dominant eigenvalues => CKA collapses toward 0.
    # (Verified: synthetic anisotropic data gives 0.005 unaligned vs 1.000 aligned. An ISOTROPIC synthetic
    # test gives 0.95 unaligned and HIDES this -- which is how the bad advice survived being sanity-checked.)
    print("[geom] HOW TO READ THIS -- each statistic has its OWN reference, and they are not interchangeable:")
    print("  * paired_rel_l2 and cka  -> compare against the NUMERICAL FLOOR row (same rung, ROW-ALIGNED,")
    print("    differing only by batch reduction order). That is the only within-condition reference for")
    print("    a paired/aligned statistic.")
    print("  * mean_shift and subspace_cos -> compare against the WITHIN-RUNG SPLIT-HALF row. These are")
    print("    distributional, so disjoint halves are a legitimate same-condition reference at this n.")
    print("  ⛔ Do NOT compare between-rung CKA to the split-half CKA: the halves are not row-aligned, so")
    print("     that number is a category error, not a noise floor.")


if __name__ == "__main__":
    main()
