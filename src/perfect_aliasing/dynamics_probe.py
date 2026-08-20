"""Training-dynamics probe (Exp-002 experiment #3) — *when* does the truth axis invert?

`train_rl.py --checkpoint-every N` dumps a LoRA checkpoint series (`<out>/checkpoints/epoch_XXXX` plus an
`index.json` recording each checkpoint's epoch and the reward at that point). This script walks that series
and runs the **same** cross-context truth probe used in `probes.py` on every checkpoint, so the
answer-token inversion can be plotted against the reward curve.

**The claim it tests.** Exp-001 showed the finished RL model computes the true bit mid-stack and then
inverts it at the generation token. That is compatible with two very different stories:

* *suppression* — the model learns the truth first (reward converges), and the inversion is layered on
  top afterwards; or
* *belief drift* — the model stops representing the truth as it trains (the Obfuscation Atlas reading,
  2602.15515, "the model genuinely no longer considers hardcoding deceptive").

These make opposite predictions about **onset order**. If the inversion appears only *after* behaviour has
converged, belief drift is ruled out and suppression is the account. That is the whole point of #3, and it
is the paper's rebuttal to the closest prior work — so the epoch axis has to be measured, not argued.

Efficiency note: the base model is loaded **once** and each checkpoint is attached as a *named* PEFT
adapter (`load_adapter` + `set_adapter`). Reloading an 8B base 21 times would dominate the runtime.

    python src/perfect_aliasing/dynamics_probe.py --model-id 8b \
        --checkpoints runs/rl_8b_s0_dyn/checkpoints \
        --n 400 --out analysis/dynamics_8b_s0/dynamics.json
"""
import argparse
import json
from pathlib import Path

from peft import PeftModel

import model as model_mod
import probes as probes_mod


def _load_index(ckpt_root: Path):
    """Read ``index.json`` and return checkpoint records sorted by epoch.

    We trust the index for the epoch/reward pairing (it is written by the trainer at save time) but
    re-verify that each directory actually exists — a truncated or interrupted train would otherwise
    produce a dynamics curve with silent holes in it.
    """
    index_path = ckpt_root / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"no index.json at {index_path} — was the train run with CHECKPOINT_EVERY>0?")
    records = json.loads(index_path.read_text())
    resolved, missing = [], []
    for rec in sorted(records, key=lambda r: r["epoch"]):
        ckpt_dir = ckpt_root / rec["dir"]
        (resolved if ckpt_dir.is_dir() else missing).append((rec, ckpt_dir))
    if missing:
        print(f"[dynamics] WARNING: {len(missing)} checkpoint dir(s) in index.json are absent on disk: "
              f"{[r['dir'] for r, _ in missing]} — the curve will have gaps.", flush=True)
    if not resolved:
        raise SystemExit(f"index.json lists {len(records)} checkpoints but none exist under {ckpt_root}")
    print(f"[dynamics] {len(resolved)} checkpoints: epochs {[r['epoch'] for r, _ in resolved]}", flush=True)
    return resolved


def _summarize(layers):
    """Pull the headline numbers out of a per-layer probe table.

    ``last_*`` is the final layer (the answer-token readout that inverts) and ``peak_*`` is the layer
    where the *ally* probe is strongest — the "does it still know the truth" mid-stack reference. Keeping
    both per checkpoint is what lets F3 show the two curves separating as training proceeds.
    """
    if not layers:
        return {}
    last = layers[-1]
    peak = max(layers, key=lambda r: (r.get("ally_iid") or 0.0))
    return {
        "last_layer": last["layer"],
        "last_ally_iid": last.get("ally_iid"),
        "last_rival_ood": last.get("rival_ood"),
        "last_rival_auroc": last.get("rival_ood_auroc"),
        "last_rival_auroc_optsign": last.get("rival_ood_auroc_optsign"),
        "peak_layer": peak["layer"],
        "peak_ally_iid": peak.get("ally_iid"),
        "peak_rival_ood": peak.get("rival_ood"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Probe every training checkpoint (Exp-002 #3).")
    ap.add_argument("--checkpoints", required=True,
                    help="checkpoint root written by train_rl.py (the dir containing index.json)")
    ap.add_argument("--model-id", default=None, help="base model: alias 1b/3b/8b/... or a full HF id")
    ap.add_argument("--n", type=int, default=400,
                    help="episodes per checkpoint (default 400: 21 checkpoints makes this the hot loop)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=["single", "multi"], default="single")
    ap.add_argument("--alt-pos", type=int, default=-3,
                    help="offset of the context-aware non-final read (the off-position control)")
    ap.add_argument("--every", type=int, default=1,
                    help="subsample the checkpoint series, keeping every Kth (default 1 = all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most this many checkpoints (0 = no limit). For smoke runs.")
    ap.add_argument("--out", default="data/dynamics.json")
    args = ap.parse_args(argv)          # argv=None reads sys.argv (CLI); a list lets Modal call this in-process

    ckpt_root = Path(args.checkpoints)
    resolved = _load_index(ckpt_root)
    if args.every > 1:
        resolved = resolved[::args.every]
    if args.limit:
        resolved = resolved[:args.limit]
    if args.every > 1 or args.limit:
        print(f"[dynamics] SUBSAMPLED to {len(resolved)} checkpoints "
              f"(every={args.every} limit={args.limit or 'none'}): "
              f"epochs {[r['epoch'] for r, _ in resolved]} — NOT the full series", flush=True)

    device = model_mod.get_device()
    tokenizer = model_mod.load_tokenizer(args.model_id)
    base = model_mod.load_model(model_id=args.model_id, device=device)   # eval mode, no adapter
    n_layers = model_mod.num_layers(base)

    model = None
    series = []
    for rec, ckpt_dir in resolved:
        name = f"ep{rec['epoch']:04d}"
        if model is None:
            model = PeftModel.from_pretrained(base, str(ckpt_dir), adapter_name=name)
            model.to(device)
        else:
            model.load_adapter(str(ckpt_dir), adapter_name=name)
        model.set_adapter(name)
        model.eval()            # LoRA carries dropout=0.05; analysis must be deterministic

        # readout_directions depends only on the frozen base (lm_head + final norm), but recompute per
        # checkpoint so the value recorded alongside each probe is unambiguously the one used.
        readout_dir, readout_dir_normed = probes_mod.readout_directions(model, tokenizer)
        positions, behavior = probes_mod.collect(
            model, tokenizer, device, None, args.n, args.seed, n_layers, args.task, alt_pos=args.alt_pos
        )

        def run(pos):
            p = positions[pos]
            return probes_mod.probe_layers(
                p["acts"], p["labels"], p["contexts"], n_layers, readout_dir, readout_dir_normed)

        layers, layers_alt = run("last"), run("alt")
        entry = {
            "epoch": rec["epoch"],
            "reward": rec.get("reward"),
            "behavior": behavior,
            "summary": _summarize(layers),
            "summary_altpos": _summarize(layers_alt),
            "layers": layers,
            "layers_altpos": layers_alt,
        }
        series.append(entry)

        s = entry["summary"]
        auroc = s.get("last_rival_auroc")
        auroc_s = f"{auroc:.3f}" if auroc is not None else " n/a"
        print(f"[dynamics] epoch {rec['epoch']:>4} | reward {rec.get('reward')} | "
              f"rival_deception {behavior['rival_deception_rate']:.3f} | "
              f"L{s['last_layer']} rival {s['last_rival_ood']:.3f} (AUROC {auroc_s}) | "
              f"peak L{s['peak_layer']} rival {s['peak_rival_ood']:.3f}", flush=True)

        # Write incrementally: a 21-checkpoint sweep is long, and a crash at checkpoint 19 should not
        # throw away the first 18 probes.
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"model_id": args.model_id, "checkpoints_root": str(ckpt_root), "n": args.n,
             "seed": args.seed, "alt_pos": args.alt_pos, "n_layers": n_layers, "series": series},
            indent=2))

    print(f"Wrote {args.out} ({len(series)} checkpoints)")


if __name__ == "__main__":
    main()
