# Experiment 001 — Emergent vs instructed deception, head-to-head

## What this experiment is

The project started as one Colab notebook that did two things: (1) **RL-trained** a small model
(`Llama-3.2-1B-Instruct` + LoRA) to lie to rivals and be honest to allies — deception that *emerges*
from a reward, not from an instruction — and (2) ran three mechanistic checks on it (a cross-context
"truth" probe, activation patching, and truth-direction ablation/steering).

Experiment 001 began as three things:

1. **Replicate** — turn that throwaway notebook into the project's reusable instrument
   ([`src/perfect_aliasing/`](../../src/perfect_aliasing/)) and reproduce its behavior from clean, organized code.
2. **Fix** — repair the bugs the notebook carried (below).
3. **Baseline** — add the **instructed-deception baseline**: the *same game*, but the model is *told*
   to lie (a system-prompt directive) instead of *trained* to. This is the head-to-head control the
   paper needs — it lets every later probe compare **emergent** vs **instructed** deception on
   identical inputs (STATUS §5).

…but it outgrew that scope. Establishing the **matched pair** (an RL deceiver and an instructed deceiver
that behave the same on the surface) exposed a **1B capability floor**, which forced a scale-up to **8B**
(writeup-1); the 8B head-to-head then produced the project's headline **divergence** (writeup-2): the
emergent liar overwrites its internal truth direction (`rival_ood → 0.00`) while the instructed liar keeps
it decodable (`0.97`), with activation patching confirming it causally. So 001 is no longer just setup —
it's the first head-to-head that found the effect. *Hardening* that result — seed replication, the
diff-of-means axis, layer/α sweeps — landed in [`writeup-3`](writeup-3-confound-controls.md) §4–4d;
**figures** are the main piece left.

## How it's done

The instrument is split into small, single-purpose modules; each analysis is a CLI that takes
`--adapter <dir>` (the **emergent** arm) or `--instructed` (the **instructed** arm), so the exact same
code runs on both.

| step | code | output |
|---|---|---|
| RL training (REINFORCE) | [`train_rl.py`](../../src/perfect_aliasing/train_rl.py) | LoRA adapter + `history.json` |
| behavior eval | [`behavior.py`](../../src/perfect_aliasing/behavior.py) | ally truth-rate / rival deception-rate |
| cross-context truth probe (RQ 1.1) | [`probes.py`](../../src/perfect_aliasing/probes.py) | per-layer ally-IID vs rival-OOD accuracy |
| patch / ablate / steer (RQ 1.2) | [`interventions.py`](../../src/perfect_aliasing/interventions.py) | flip / ally-truth / rival-truth rates |
| the game + reward (shared) | [`game.py`](../../src/perfect_aliasing/game.py) | — |
| model loading + layer locator | [`model.py`](../../src/perfect_aliasing/model.py) | — |
| the instructed directive + arm loader | [`instructed.py`](../../src/perfect_aliasing/instructed.py) | — |

### The three fixes carried in from the notebook

1. **Inverted training gate.** The notebook trained only `if file_path.is_file()` — i.e. only when the
   adapter *already* existed. Removed: training is now its own entrypoint (`train_rl.py`); analyses
   load the produced adapter via `--adapter`.
2. **Inconsistent layer access.** The notebook reached the decoder stack via
   `model.model.model.layers` in one place and `model.base_model.model.model.layers` in another. All
   intervention hooks now go through `model.get_decoder_layers()` (one robust resolver).
3. **Hard-coded CUDA/bf16.** Loading is now device-agnostic (`cuda → mps → cpu`, bf16 on CUDA else
   fp32) so it runs on a GPU box, a Mac, or a CPU container.

Behavior is otherwise identical to the notebook (same LoRA r=16/α=32 on q,k,v,o; AdamW lr=1e-5;
1000 epochs × batch 8; reward ally +1/−1, rival +1.5/−2.0; softmax-over-{"0","1"} REINFORCE).

## Commands (reproduce)

Full pipeline (both arms → figures), local-first path:

```bash
bash run.sh                 # needs a GPU to be practical
EPOCHS=2 N=20 bash run.sh   # tiny CPU smoke to validate the pipeline
```

Or step-by-step, e.g. the head-to-head probe:

```bash
python ../../src/perfect_aliasing/train_rl.py --out ../../adapters/rl_deceiver
python ../../src/perfect_aliasing/probes.py --adapter ../../adapters/rl_deceiver --out data/probe_emergent.json
python ../../src/perfect_aliasing/probes.py --instructed                        --out data/probe_instructed.json
python make_figures.py
```

The heavy run is executed on GPU (public path: `modal run ../../src/perfect_aliasing/modal_app.py::train`;
see the repo README's Compute section). Figures land in [`figures/`](figures/), raw JSON in
[`data/`](data/).

## Writeups (the narrative, in phases)

Experiment 001 is written up in phases; read them in order:

- **[`writeup-1.md`](writeup-1.md)** — replication + the instructed baseline, the chance-level result,
  the capability-floor root cause, the three-size sweep, and the decision to move to 8B.
- **[`writeup-2-scaling-to-8b.md`](writeup-2-scaling-to-8b.md)** — the 8B phase: retrain the emergent
  deceiver and run the head-to-head (PHASE A probes + PHASE B interventions) *(landed; read with the ⚠
  scope correction at its top)*.
- **[`writeup-3-confound-controls.md`](writeup-3-confound-controls.md)** — hardening the instrument: the
  dropout-asymmetry fix, the last-token/output-readout confound controls, the **position sweep** (the
  inversion is a single-token effect), **3-seed replication**, and the **diff-of-means axis + steering
  α-sweep** (§4d) *(controls + causal-axis checks done §4–4d)*.
- **[`writeup-4-synthesis.md`](writeup-4-synthesis.md)** — ⭐ the standalone, **figure-driven** synthesis of
  the whole experiment for a non-specialist reader (embeds all of [`figures/`](figures/)). Start here for
  the narrative; the phase writeups above are the lab notebook.

## Results — see the writeups for the full narrative

> **Headline (8B, [`writeup-2 §5–6`](writeup-2-scaling-to-8b.md)): a clean divergence.** On 8B the RL
> deceiver plays the game perfectly (ally-truth **1.00** / rival-deception **1.00**; training converged)
> and the instructed baseline follows its directive (0.90 / 0.74). The cross-context truth probe then
> splits them **at the answer-generation token**: both hold the true bit to L11–13, then the **emergent**
> arm reads **inverted** there by the final layer (`rival_ood_auroc → 0.00`, the truth axis rotated into
> the lie) while the **instructed** arm stays decodable (**0.998** at L32).
> **⚠ Scope (see [`writeup-3`](writeup-3-confound-controls.md)):** confound + position controls localize
> this to the **answer token only** — at content tokens (−5+) the truth is fully, correctly decodable in
> *both* arms, and the causal patch is answer-token-specific too (flip **1.00** @−1 vs **0.00** @−3). So
> the honest claim is *"a **last-token** truth probe is inverted on the RL liar at the generation site,
> recoverable a couple tokens upstream"* — **not** "the model overwrites its truth direction." Now
> **reproduces across 3 RL seeds** (writeup-3 §4c).
> *The 1B result below is the writeup-1 record of why we had to scale up.*

**On 1B, both arms came back at chance** on the multi-variable game — neither the RL model nor the
instructed baseline plays the game. Root cause (confirmed by diagnostics): Llama-3.2-1B can't reliably
emit the queried bit — it has a strong "answer 0" habit and occasionally refuses — so the "deception"
measurements are dominated by a capability floor. A follow-up honest-retrieval sweep across sizes
(**1B 0.48 / 3B 0.82 / 8B 1.00**) settled the fix: we move to **Llama-3.1-8B**, which clears the floor
natively. Full story, method explanations, and interpretation in [`writeup-1.md`](writeup-1.md).

| arm | ally truth | rival deception | probe (rival OOD, best layer) | patch flip | ablate (ally truth) | steer (rival truth) |
|---|---|---|---|---|---|---|
| emergent (RL) | 0.52 | 0.49 | ~0.72 (L10–12) | 0.49 | 0.54 | 0.52 |
| instructed | 0.54 | 0.42 | ~0.66 (L16) | 0.49 | 0.54 | 0.52 |

One real signal: the ally-trained truth probe reaches **~0.82** (emergent) / **~0.84** (instructed) by
L16 — the true bit *is* linearly represented internally even though the model doesn't emit it. The
interventions (identical across arms **and** across a 3-epoch smoke) are noise, as expected when
behavior is at chance.

**Status:** 1B capability floor resolved by moving to **8B**. The 8B head-to-head is **done through PHASE B
and hardened**: PHASE A (probe divergence), PHASE B (causal patch), then confound + position + **3-seed**
controls ([`writeup-3`](writeup-3-confound-controls.md)). **Verdict:** the inversion is real and seed-robust
but **answer-token-localized** — *"a last-token probe is inverted on the RL liar at the generation site,
recoverable upstream,"* not a general overwrite. The diff-of-means axis + steering α-sweep are **done**
(§4d: ablation still null = redundant encoding; steering *is* sufficient at L12/α≥20). Remaining before
paper-ready: **figures**, and — for main-track — the *why/generalization* arc. (The original notebook's
ablation=1.00 / steer=0.19 came from a *single fixed prompt* — it does not hold on a varied distribution.)

## Files

- [`writeup-4-synthesis.md`](writeup-4-synthesis.md) — ⭐ figure-driven synthesis (read first); phase
  writeups [`writeup-1`](writeup-1.md) → [`writeup-2`](writeup-2-scaling-to-8b.md) →
  [`writeup-3`](writeup-3-confound-controls.md)
- [`run.sh`](run.sh) — the reproduction recipe / CPU-smoke driver
- [`make_figures.py`](make_figures.py) — renders `figures/*.png` from `data/figuredata.json` (skip-soft)
- `data/` — dumps incl. `figuredata.json` (gitignored except `.gitkeep`; pull via the fetch job)
- [`figures/`](figures/) — the 6 rendered PNGs (embedded in writeup-4)
