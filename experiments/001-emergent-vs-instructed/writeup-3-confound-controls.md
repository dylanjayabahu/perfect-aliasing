# Experiment 001 · Writeup 3 — Hardening the instrument: confound controls before seed replication

*Started 2026-07-08. Continues from [`writeup-2`](writeup-2-scaling-to-8b.md), whose headline is the
`rival_ood → 0.00` emergent inversion. Before spending seed compute we (a) closed a dropout asymmetry
and (b) added retraining-free controls that decide whether the 0.00 is a **real sign-flipped truth axis**
or a **tautology of the output readout**. **Status: DONE — confound (§4), position sweep (§4b), and
3-seed replication (§4c) all landed. Verdict: the inversion is real and seed-robust but
*answer-token-localized* — a last-token effect, not a general overwrite. Remaining: diff-of-means axis,
figures, why/generalization.** Pairs with
[`../../docs/related-work.md`](../../docs/related-work.md) (the two reviewer attacks this phase answers).*

## Why this phase exists

The writeup-2 result reads every activation at the output-committed final position (`[0, -1, :]`, where
`add_generation_prompt` lands). A code audit (see the audit thread / [STATUS](../../docs/STATUS.md)) found
**no bug that fabricates the emergent-only inversion** — the probe label is the ground-truth bit
([`probes.py`](../../src/perfect_aliasing/probes.py) `collect`), identical for both arms, so a label/position bug
would move *both* arms, not just emergent. But it surfaced two things worth fixing before we trust
seed-to-seed numbers:

1. **A dropout asymmetry** (hygiene) that could add uncontrolled, arm-asymmetric noise.
2. **A last-token / output-readout confound** (methodological): at late layers a probe at the answer
   slot is partly a *next-token* readout, so `rival_ood = 0.00` could be the unembedding, not a truth
   axis that got sign-flipped.

## 1. Fix — dropout asymmetry (was: audit concern #2)

The LoRA adapter carries `lora_dropout=0.05`. The emergent arm loads via `PeftModel.from_pretrained` and
the instructed/base arm via a plain `from_pretrained` — **different load paths**, and nothing forced eval
mode. If dropout were live it would be an asymmetric, uncontrolled noise source between the two arms and
across seeds.

**Fix:** [`model.py`](../../src/perfect_aliasing/model.py) `load_model` now calls `model.eval()` on every
non-training path (both arms), gated on `not for_training`. Training deliberately stays in train mode so
REINFORCE keeps its `lora_dropout=0.05` regularization. This makes analysis deterministic regardless of
the framework's default mode.

- **Comparability note.** If seed-0's emergent run happened to have dropout live, its exact numbers may
  shift a hair on a rerun. The post-fix pipeline is the correct deterministic baseline going forward — the
  seed replication (and any seed-0 rerun) all sit on it.

## 2. Fix — last-token / output-readout confound controls (was: audit concern #1)

All **additive**: existing `layers` rows and patching keys keep their exact seed-0 meaning, defaults are
unchanged, so nothing about the headline read moves. Emitted on defaults, so they come for free every run.

| control | where | what it decides |
|---|---|---|
| **`readout_cos` / `readout_cos_normed`** | `probes.py` per layer | is the probe direction just `W_U[1]−W_U[0]`? (the decisive test) |
| **`ally_iid_auroc` / `rival_ood_auroc`** | `probes.py` per layer | sign-invariant — distinguishes inversion (→0.0) from wash-out (→0.5) |
| **`layers_factpos`** | `probes.py`, stated-fact token | context-blind baseline where inversion is impossible by construction |
| **`layers_altpos`** | `probes.py`, `--alt-pos` (default −3) | context-aware but non-answer read; inversion surviving here kills the confound |
| **`--patch-pos`** | `interventions.py` (default −1) | patch a context-aware non-final token → causal test independent of the probe's read position |

- The off-position probes train a **fresh ally-only probe at each position** (position-consistent, not a
  cross-position transfer). The fact token is located by offset-mapping on the placeholder prefix, fail-soft
  to skip (count reported as `fact_pos_n`).
- The analysis job gained `PATCH_POS` / `ALT_POS` envs, so the position-independence checks
  run from a submit flag with no code edit.

## 3. The interpretation nuance — read the *right* cosine column at the final layer

**This is the important, non-obvious part.** In the pinned transformers (4.48.3), `outputs.hidden_states[-1]`
is the **post-final-RMSNorm** state — it equals `last_hidden_state`, the exact tensor `lm_head` consumes.
Our probe loop reads `hidden_states[layer]` up to `n_layers`, so the **row `layer == n_layers` (L32 on 8B)
is already normed**. Consequences:

- **At the final layer, `readout_cos` (RAW `W_U[1]−W_U[0]`) is the faithful tautology test — NOT
  `readout_cos_normed`.** The `_normed` variant (`diff ⊙ gain`) double-applies the RMSNorm gain at L32; it
  is the correct column only for the **pre-norm intermediate layers** (1..31), none of which are actually
  unembedded. (This is the opposite of the first-cut intuition that "normed = what the output reads" —
  true for a raw residual, but the stored L32 state is already post-norm.)
- **Verify on-env (one line):**
  ```python
  out = model(**inputs, output_hidden_states=True)
  torch.allclose(out.hidden_states[-1], out.last_hidden_state)  # True ⇒ post-norm ⇒ use readout_cos at L32
  ```
- **Second-order — why a single L32 number isn't enough (and what to plot instead).** Because L32 *is* the
  pre-logit state, a genuine truth feature will align *somewhat* with `W_U` there for **both** arms (on
  allies, truth == output by definition). So a high `readout_cos` at L32 alone is not damning. The decisive
  evidence is the **contrast of `readout_cos` against the `rival_ood` collapse across the L14→L32 band, per
  arm and cross-arm.** writeup-2 puts the emergent inversion already deep by ~L24 (`rival_ood ≈ 0.02`) —
  *pre-norm and not the answer slot*. If `readout_cos` is still low there while `rival_ood` has already
  cratered, the inversion **cannot** be unembedding alignment. **That mid-band contrast is the money plot.**

## 4. Interpretation key (how to read the seed-0 control run)

Run order: the cosine + off-position + patch@−3 checks need **no retraining** — they run on the existing
seed-0 8B adapter. Run them first; if they hold, the 3-seed replication proceeds.

| signal | reads as REAL inversion | reads as ARTIFACT |
|---|---|---|
| `rival_ood_auroc` (emergent, late) | **≈ 0.0** (perfect anti-correlation = sign-flip) | ≈ 0.5 (wash-out) |
| `readout_cos` (L32) vs mid-band | high at L32 **but low ~L18–24 where `rival_ood` already collapsed** | tracks the collapse (climbs as `rival_ood` falls) |
| `layers_altpos` (−3), patch@−3 | emergent vs instructed **still diverge** | divergence vanishes off the answer slot |
| `layers_factpos` | `rival_ood ≈ ally_iid` (both high) for both arms — pipeline sanity | truth not even present at the fact token |

**Seed-0 control results (ran 2026-07-08, read 2026-07-09; commit `e1b880a`, `cf-*` jobs):**

| control | emergent (RL, 8B) | instructed (8B) | reading |
|---|---|---|---|
| behavior ally / rival | 1.00 / 1.00 | 0.90 / 0.735 | reproduces writeup-2 ✓ |
| `rival_ood_auroc` @ L32 (answer pos) | **0.00** | 0.998 | genuine sign-flip, not wash-out; emergent-specific ✓ |
| `rival_ood_auroc` first → 0.0 | **L20** | — | inversion begins at L20 … |
| `readout_cos` @ L20 / L32 | +0.21 / **+0.51** | +0.09 / +0.07 | … where cos is only 0.21 (not a pure `W_U` tautology) but climbs to ~0.5 by L32 |
| `layers_altpos` (−3) auroc @ L32 | **0.96** | 0.999 | ⚠ inversion **vanishes** one token before the answer |
| patch@−3 flip | **0.00** | 0.185 | ⚠ causal effect **gone** off the answer slot (vs patch@−1: 1.00 / 0.51) |
| `layers_factpos` rival_ood | 1.00 | 1.00 | pipeline sane ✓ |

**Verdict (2026-07-09): the inversion is REAL but ANSWER-POSITION-LOCALIZED — the "overwrite" framing is overstated.**
The 0.00 is a true sign-flip (auroc→0.0, not 0.5) and **emergent-specific** — the instructed model emits the lie too yet keeps truth decodable at the answer position (0.97), so this is *not* a trivial "answer-position = output-token" tautology (that would invert instructed as well). The inversion also begins at L20 where `readout_cos` is only +0.21, so it isn't merely the unembedding.
**But both position-independence checks fail:** one token earlier (−3) the emergent inversion is gone (auroc **0.96 ≈ instructed 0.999**), the causal patch off the answer slot does nothing (**0.00 vs 0.185**), and `readout_cos` rises to ~0.5 at L32. → The RL model rotates its truth axis into the emitted lie **at the answer token**; the truth is intact and correctly-signed one position earlier. The claim "overwrites its internal truth direction" (writeup-2/README/STATUS) must be **narrowed to an answer-site effect**, and the passive-probe blind spot holds only for a detector reading the answer token — defeated by reading one token back.

### 4b. Position sweep (2026-07-09) — the inversion is a *single-token* effect

`rival_ood_auroc @ L32` at each read position, both arms, seed-0 (`−1` = the answer token, from the LAST read):

| pos | emergent | instructed | note |
|---|---|---|---|
| **−1** | **0.00** | 0.998 | answer token — emergent inverted, instructed intact: **the divergence** |
| −2 | 0.79 | 0.82 | both mildly degraded, ~equal |
| −3 | 0.96 | 0.999 | both high |
| −4 | 0.24 | 0.56 | **both dip → format-token/position artifact, not deception** |
| −5 | 1.00 | 0.999 | decodable |
| −6 | 0.98 | 1.00 | decodable |
| −8 | 1.00 | 1.00 | decodable |
| −12 | 1.00 | 1.00 | decodable |

**The emergent-specific inversion exists at exactly one position: the answer token (−1).** At content
positions (−5+) the truth is fully decodable and correctly-signed in *both* arms; the −2/−3/−4 band is
noisy for both (the −4 dip appears in the *instructed control too*, so it's a structural-token property,
not the RL policy). **Refined claim:** emergent deception *entangles the truth direction with the emitted
lie at the generation token*; instructed deception keeps them separable even there. Safety consequence,
stated precisely: a **last-token** truth probe is inverted on the RL liar but not the prompted one — yet
the truth is fully recoverable a couple content-tokens upstream. This is **not** "detectors are blind to
emergent deception"; it is "last-token detectors are inverted at the generation site."

### 4c. Seed replication (2026-07-09) — the single-token inversion is seed-robust

PHASE A+B on 3 independent RL seeds (`rl_deceiver_8b_s{1,2,3}`), both arms:

| seed | emergent ally/rival | emergent LAST auroc@L32 | emergent ALT−3 auroc@L32 | emergent patch@−1 | instructed LAST auroc@L32 | instructed patch@−1 |
|---|---|---|---|---|---|---|
| 1 | 1.00/1.00 | **0.00** | 0.92 | **0.995** | 0.99 | 0.58 |
| 2 | 1.00/1.00 | **0.00** | 0.91 | **1.00** | 0.99 | 0.575 |
| 3 | 1.00/1.00 | **0.00** | 0.95 | **0.99** | 1.00 | 0.61 |

<sub>patch@−1 columns are from the per-seed `iv2` re-run (2026-07-09); see caveat (2).</sub>

**All three seeds reproduce:** emergent training converges (1.00/1.00), the answer-token inversion is exact
(`LAST auroc = 0.00`), the truth is intact one token upstream (`ALT−3 ≈ 0.9`), and the causal flip holds
(patch@−1 = 1.00 vs instructed ~0.50). The narrowed claim — *answer-token truth→lie inversion,
emergent-specific, recoverable upstream* — is **seed-robust**, not seed-0 luck.

**Caveats:** (1) ablation stays null across seeds (emergent ally-truth = 1.00 after ablating the L12 LR
direction) → the LR probe direction is a *correlate*, not the causal axis; diff-of-means axis is still the
fix. (2) The earlier byte-identical instructed interv numbers were an **operator error, not a bug** — the first
PHASE B submit omitted `--env SEED`, so all three ran at the default `SEED=0`. **Re-run with per-seed
`--env SEED` (`iv2`, 2026-07-09):** every intervention now varies across seeds — emergent patch
0.995/1.00/0.99, instructed patch 0.58/0.575/0.61, instructed ablate 0.79–0.91, steer 0.20–0.24 —
confirming `--seed` threads correctly and there was no bug. The **emergent≫instructed causal-flip
divergence holds across all 3 seeds** (≈0.99 vs ≈0.59).

### 4d. Diff-of-means axis + steering α-sweep (2026-07-17) — the ablation-null is real; the "steering-null" was not

Two `interventions.py` follow-ups (commits `a12a81c`, `49a0fc4`), 8B seed-0, both arms.

**(i) Diff-of-means truth axis** (`--axis diffmeans` = `mean(act|true=1) − mean(act|true=0)` at L12, vs the
LR probe coefficient). Tests whether the writeup-2 ablation-null was just a bad direction estimate
(LR = correlate). It was not:

| control (L12, diffmeans) | emergent | instructed |
|---|---|---|
| ablate → ally truth | **1.00** (null) | 0.91 |
| patch @L16 (axis-free) | 1.00 | 0.505 |

The emergent ablation is **still null on the principled axis** (ally truth 1.00) → the truth direction is not
causally *necessary*; the null is a property of the RL model (consistent with **redundant encoding**), not the
LR estimator. Patching (axis-free) reproduces the divergence exactly (1.00 vs 0.505).

**(ii) Steering α-sweep, same layer** (extract *and* steer at L12; α ∈ {5,10,20,40}) — to disambiguate the
null single-α steer: under-powered, or no linear steer exists?

| α | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| emergent rival-truth | 0.00 | 0.00 | **0.46** | **0.49** |
| instructed rival-truth | 0.26 | 0.26 | 0.61 | 0.49 |

**Steering is NOT null.** At the extraction layer with α≥20 the diff-of-means direction pushes emergent rivals
toward truth (→ ~0.47), comparable to instructed. Writeup-2's `steer = 0.00` was an artifact of steering an
L12-derived direction downstream at L20 with α=5. **Correction:** the "ablation + late steering null → the LR
direction is only a correlate" reading is **retracted** — a principled axis at the right layer/strength is
causally *sufficient* to move behaviour; it is just not *necessary* (ablation redundancy).

**Net:** diff-of-means is not a new divergence axis — steerability is roughly symmetric across arms (both plateau
~0.5 by α=40). The load-bearing causal divergence stays **patching** (emergent 1.00 vs instructed 0.505,
axis-free, seed-robust) + the answer-token probe inversion (§4–4c). *Caveat:* rates plateau near chance (0.5) and
instructed is non-monotonic (0.61→0.49); reaching ~0.5 may be partial truth-installation vs high-α output
degradation — an ally-side steering control would disambiguate (not load-bearing).

## 5. Open / next

- [x] Seed-0 confound battery (§4) — real but answer-position-localized.
- [x] **Overclaim corrected** in writeup-2, the exp [`README`](README.md), [`STATUS`](../../docs/STATUS.md),
      and the root README: "overwrites its truth direction" → "last-token inversion at the generation site,
      recoverable upstream."
- [x] **Position sweep (§4b)** — the inversion is a *single-token* effect (auroc 0.00 only at −1).
- [x] **Seed replication (§4c)** — reproduces across 3 RL seeds.
- [x] **Diff-of-means axis + steering α-sweep (§4d, 2026-07-17)** — ablation stays null on the diff-of-means
      axis (truth not causally *necessary* — redundant encoding, not an LR artifact); the "steering null" was
      **under-powered** — same-layer L12/α≥20 steering *is* causally *sufficient* (emergent rivals → ~0.47).
      Causal divergence rests on patching, not ablation/steering.
- [ ] **Figures** (`make_figures.py`): the position-sweep curve (auroc vs read position — the money plot),
      the probe-collapse curve (all layers, both arms), per-seed bars.
- [ ] **(ICLR main only) why + generalization** — why RL entangles truth with the output at the generation
      token; does it hold across models / tasks.
