# Experiment 001 · Writeup 2 — Scaling to 8B: a clean instrument for the head-to-head

*Started 2026-07-07, undergrad level. Continues directly from
[`writeup-1`](writeup-1.md), which ended by choosing Llama-3.1-8B. **Status: PHASE A + PHASE B landed on
8B; the divergence is real but — per the ⚠ correction below and [`writeup-3`](writeup-3-confound-controls.md) —
*answer-token-localized* and now *seed-robust* (3 seeds). The diff-of-means axis + steering α-sweep are
**done** (writeup-3 §4d) — they supersede this writeup's §6–§7 "ablation/steering null → LR is only a
correlate" reading: ablation stays null on the diff-of-means axis too (redundant encoding), and steering
*is* sufficient at L12/α≥20 (the earlier steer-null was under-powered). Remaining: figures, and
(main-track only) the why/generalization arc.**
This writeup grows as results land.*

> **⚠️ CORRECTION (2026-07-09) — read first.** The confound + position controls in
> [`writeup-3`](writeup-3-confound-controls.md) show the inversion described below is **localized to the
> answer-generation token**, not a general representational overwrite. At content tokens (−5 and beyond)
> the truth is fully and correctly decodable in *both* arms; the emergent-specific inversion exists at
> exactly one position (−1), and the causal patch is answer-token-specific too (flip 1.00 @−1 vs 0.00
> @−3). The "overwrites/destroys the truth direction" and "distributed suppression circuit" language in
> §5b–6 **overstates scope** — read those sections as *what was measured at the answer position*, and take
> the corrected claim from writeup-3 §4/§4b: *a **last-token** probe is inverted on the RL liar at the
> generation site, recoverable a couple tokens upstream.*

## TL;DR

**The question (plain version):** if a model *learns* to lie because lying gets rewarded, does it lie
*differently on the inside* than a model you simply *tell* to lie in its prompt? If so, the lie-detectors
we build by studying told-to-lie models could have a blind spot for the more dangerous, self-taught kind.

**What we did:** we set up the *same* lying game for two models — be honest to "allies," lie to "rivals."
One model **learned** the lying strategy from a reward (reinforcement learning) — we call it *emergent*.
The other was just **told** to lie by its system prompt — *instructed*. Both run on Llama-3.1-8B (the 1B
we started with was too small to even play the game — see [writeup-1](writeup-1.md)). Then we opened up
each model and asked: *is the true answer still represented inside the network while it's lying?* — using
a "truth probe" (read the true bit off the internal activations) and causal edits (force the truth in and
watch what the model does).

**What we found:** both models lie the same on the surface — but the **learned** liar *hides the truth
inside itself*, while the **told-to-lie** model doesn't.

| what we measured | emergent (RL-trained liar) | instructed (told-to-lie) | what it means |
|---|---|---|---|
| lies to rivals, on the surface? | yes (100%) | yes (74%) | both lie — you can't tell them apart from the outside |
| is the true answer still readable *inside*, at the last layer? | **no — 0.00** | **yes — 0.97** | only the learned liar erases the truth internally* |
| force-feed the truth mid-network — does it come clean? | **yes, flips 100%** | no effect (51%) | the learned lie lives *in the network's middle*; the told-to-lie one lives in the prompt |

<sub>*0.00 isn't "truth is gone" (that'd be ~0.50, a coin flip). It's the truth-reading flipped *upside
down* — the model rewrites its internal "truth" signal to point at the lie.</sub>

**Takeaway (scoped — see the ⚠ correction above):** a *passive* **last-token** truth-probe is **inverted**
on the learned liar *at the answer token*, while it reads the told-to-lie model fine there — but the truth
is fully recoverable a couple tokens upstream (and a causal edit surfaces it too). So the blind spot is
real and emergent-specific, but **narrow**: it lives at the generation token only. **Caveat:** the
answer-token inversion now reproduces across **3 RL seeds** (writeup-3 §4c), but two causal edits (ablation,
late steering) remain inconclusive and the diff-of-means axis is still to-do — a promising, precisely-scoped
signal, not a finished result. Details and honest caveats in §5–6a; the scope fix in writeup-3 §4/§4b.

---

## 1. Where writeup-1 left us

The one-line recap of the decision (full reasoning in [`writeup-1` §8–9](writeup-1.md)):

| model | honest-retrieval accuracy (base, ally-only, n=50) |
|---|---|
| Llama-3.2-1B | 0.48 (chance) |
| Llama-3.2-3B | 0.82 |
| **Llama-3.1-8B** | **1.00** |

We adopt **8B**. 3B was rejected because its 18% honest-retrieval error would contaminate the very
comparison we're trying to make; 8B clears the floor cleanly, so the planned few-shot/SFT warmup is
**dropped** — scale removed the problem it was meant to work around.

## 2. What this phase does

Exactly the study Experiment 001 was built for, now on a model that can do the task:

1. **Retrain the emergent arm on 8B** (single-bit task, our clean instrument). Same REINFORCE loop,
   same reward table, same LoRA recipe — only the base model changes, threaded through the new
   `--model-id` flag ([`model.py`](../../src/perfect_aliasing/model.py)).
2. **Re-run the head-to-head** — behavior + the three internal measurements
   ([`behavior.py`](../../src/perfect_aliasing/behavior.py), [`probes.py`](../../src/perfect_aliasing/probes.py),
   [`interventions.py`](../../src/perfect_aliasing/interventions.py)) on **both** arms: emergent (the RL adapter)
   and instructed (base 8B + the deception directive).
3. **Read the result against the paper's question** — does emergent deception look different, inside,
   from instructed deception? (See §4.)

The point of re-running on 8B is that in writeup-1 all of these numbers were dominated by "the model
can't emit a `1`." With honest retrieval at 1.00, ally-truth should sit near the ceiling, so any
difference we see toward *rivals* is attributable to the deception policy — not to a capability leak.

## 3. The one fix a bigger model forces

The interventions in writeup-1 hard-coded layer indices (`L11`/`L13`) tuned for the **16-layer** 1B.
The 8B has **32 layers**, so those indices point at the wrong depth and would produce meaningless
patching/ablation/steering numbers. Rather than guess a rescaled layer, we **read it off the data**:
the cross-context probe ([`probes.py`](../../src/perfect_aliasing/probes.py)) already sweeps *every* layer and
reports where the truth is most linearly decodable. We take that peak layer (and its neighborhood) as
the site for the causal interventions. So the order matters — **probe first, then set the intervention
layers from the probe's peak**, not before. (The `--dir/ablate/steer/patch-layer` flags on
`interventions.py` exist exactly for this.)

## 4. What would count as a result

The paper's target hasn't changed — this phase is what finally lets us test it:

- **Divergence (the headline).** A measurement where the RL-trained deceiver behaves differently from
  the prompted deceiver — behaviorally, or in the probe/patching/steering geometry. That would mean our
  safety tools, validated on *instructed* liars, have a **blind spot for emergent deception**.
- **Non-divergence (weaker but real).** The two look the same on every measurement → "the detectors
  generalize from instructed to emergent deception." Publishable, less exciting.

Either way, the prerequisite is the same and now met: a model whose *behavior* is real, so that the
*representations* we compare mean something.

## 5. PHASE A results — behavior + the cross-context probe

*Run: `--model-id 8b`, behavior + probes only, both arms, n=1000 (≈513 ally / 487 rival), seed 0.
Single seed — replication across seeds is a to-do (§7).*

### 5a. Behavior — the floor is gone, the pair exists

| arm | ally truth | rival deception |
|---|---|---|
| **emergent (RL, 8B)** | **1.00** | **1.00** |
| **instructed (8B)** | 0.90 | 0.74 |

This is the whole reason we moved to 8B. The RL model plays the game *perfectly* — honest to every
ally, deceptive to every rival — and the instructed model genuinely follows its directive (0.90 / 0.74),
where the 1B version of both arms sat at chance. We now have the **matched pair** the paper needs: two
models that both lie to rivals on the surface, ready to be compared inside.

**Training converged.** REINFORCE reward climbed from epoch 1's **−0.56** to a **+1.2–1.4 plateau** by
epoch ~100 and held it for the remaining ~900 (`first50 = −0.14`, `last50 = +1.25`; ceiling is ~+1.25–1.5
given the reward table). Contrast the 1B run, which crawled to **+0.19** and never learned the policy —
8B gave the gradient something real to climb.

### 5b. The cross-context truth probe — where they diverge

A logistic "truth" probe is trained on **ally** activations at each layer and tested on **rival**
activations (`rival_ood`). `ally_iid` (in-distribution) hits **1.00** from ~L11 on for *both* arms, so
the probe itself is perfect — every difference below lives in the **rival representation**.

Both arms compute and hold the true bit identically through the mid-stack (**peak L11–13 = 1.00**). Then
they separate, and the separation is the result:

| layer band | emergent `rival_ood` | instructed `rival_ood` |
|---|---|---|
| L6–13 (peak) | 0.99 – **1.00** | 0.99 – **1.00** |
| L14–16 | 0.97 → 0.66 | 0.90 / 0.73 |
| L17–23 | ~0.42–0.45 (below chance) | ~0.51–0.86 (noisy, stays high) |
| **L24–32** | **0.02 → 0.00** | **0.67 → 0.97** |
| **final layer (L32)** | **0.00** | **0.97** |
| mean over L24–32 | **≈ 0.01** | **≈ 0.77** |

- **Instructed:** the truth stays linearly decodable from rival activations across the *entire* stack,
  ending at **0.97**. The model produces the lie without ever corrupting its internal representation of
  the truth — precisely the setting in which honesty probes / truth directions were developed and shown
  to work. Our detector is happy.
- **Emergent:** the truth feature is **actively destroyed** downstream. `rival_ood` does not decay to
  chance (0.5, "truth washed out") — it goes to **0.00**, *perfect anti-correlation*. Along the exact
  axis that reads TRUE on allies, the late-layer rival stream now reads FALSE. The RL model doesn't hide
  the truth; it **overwrites the truth direction with the lie** it's about to emit.

So the same probe that reads the true bit off the instructed liar (0.97) is **inverted** on the emergent
liar (0.00). A deception detector validated on prompted liars would report "this model is being honest"
about the one that learned to deceive. **That is a divergence, and it's the paper's headline.**

## 6. What it means (and what PHASE B must confirm)

The 0.00-vs-0.5 detail is the interesting part. If emergent deception were a **routing failure** — the
truth computed early, then simply *ignored* downstream — the probe would land at chance (~0.5) in the
late layers, because the truth direction would carry no usable signal. Instead it lands at **0.00**,
which means the truth direction is still *there and still linear*, just **sign-flipped**. That is the
signature of an **active suppression circuit** (RQ 1.2): something between ~L14 and ~L24 *subtracts or
negates* the truth vector rather than dropping it. The collapse is monotone across that band (0.97 → 0.66
→ 0.45 → 0.02 → 0.00), so the suppression is **distributed over a span of mid-late layers**, not a single
switch.

PHASE B (running now) is the causal test of exactly this story, with the layers read **off this curve**
rather than guessed:

- **`--dir-layer 12`** — extract the truth direction where it is maximal and identical in both arms
  (`rival_ood = 1.00`), so the axis is unambiguous.
- **`--ablate-layer 12`** — project that direction out on **allies** at the same depth. If ally-truth
  craters, the axis is causally load-bearing for honest output (not just a correlate the LR probe found).
- **`--steer-layer 20`** — add +truth back into **rivals** *inside* the emergent collapse zone
  (emergent `rival_ood ≈ 0.45` there). Does re-injecting the direction rescue a truthful answer, or has
  the suppression already committed?
- **`--patch-layer 16`** — patch the ally (truthful) residual into a rival run at mid-collapse. If the
  answer flips to truth, the truth *content* is causally sufficient there; if it doesn't, the downstream
  circuit re-suppresses even a hand-inserted truth — which would localize the suppression to L16→L24.

These are single-layer pins for the first pass; the layer/α sweeps that would firm them up are in the
TODO list (§8).

## 6a. PHASE B results — the causal test

Interventions at the layers pinned from the §5b probe curve (`--dir-layer 12 --ablate-layer 12
--steer-layer 20 --patch-layer 16`, α=5, single seed):

| intervention (layer) | emergent (RL, 8B) | instructed (8B) | reads as |
|---|---|---|---|
| **patch ally→rival (L16)**, flip→truth | **1.00** | **0.51** | where does the lie decision live? |
| ablate truth dir (L12), ally-truth | 1.00 | 0.91 | is the LR axis load-bearing for honesty? |
| steer +truth (L20, α5), rival-truth | 0.00 | 0.28 | can re-injected truth override the lie? |

*(ally-truth baselines: 1.00 / 0.90; rival-truth baselines: 0.00 / 0.26.)*

**Patching is the clean, causal divergence.** Swapping the ally (truthful) residual into a rival run at
L16 flips the **emergent** model to truth **100%** of the time, but leaves the **instructed** model at
**chance (0.51)**. The two lies have different causal loci: emergent deception is **computed in the
mid-stack residual** (overwrite L16 and it's gone), while instructed deception is a **prompt/readout-level
override** that a mid-layer swap doesn't touch. This confirms the §5b probe divergence with a causal
handle — the strongest form of the paper's claim so far.

**The blind spot is specifically about *passive* probes.** The linear probe is inverted on the emergent
liar (rival_ood → 0.00, §5b), yet activation **patching recovers the truth** (flip 1.00). So the correct
claim is narrower and more defensible than "detectors fail on emergent deception": a **passive read-out
probe** is blind, but a **causal intervention** can still surface the truth the RL policy buried.

**Two honest negatives — the LR direction is not the causal axis (yet).**
- **Ablation is null in both arms** (ally-truth 1.00 / 0.91, i.e. unchanged). Removing the L12
  logistic-probe direction does not break honesty → it's a **correlate, not the causal truth axis** —
  exactly the caveat flagged in [STATUS §6](../../docs/STATUS.md) and writeup-1. The fix is a
  **diff-of-means** axis and/or multi-direction ablation (to-do §7).
- **Steering barely moves either** (emergent 0.00, instructed 0.28). L20 sits deep in the emergent
  collapse (rival_ood ≈ 0.45 → 0.00), so a rank-1 nudge along a weak axis can't reverse a nearly complete,
  distributed suppression; the instructed model (truth intact) is mildly pushable. Steering **earlier
  (L13–14) with an α-sweep** could change this story and is the priority follow-up.

**Caveats.** The emergent 1.00 / 0.00 extremes are partly its **sharp, near-deterministic policy**
(behavior 1.00 / 1.00) — read *direction and contrast*, not the exact 1.0/0.0. Single seed. Steering was
tested at only one layer/α. None of this is publishable until the seed replication + diff-of-means +
sweeps land; but the **patching divergence** (1.00 vs 0.51) is a real, interpretable signal to build on.

## 7. Status — what's done

- [x] `--model-id` threaded through the whole pipeline (train / behavior / probes / interventions /
      diagnostic + the runners); `1b`/`3b`/`8b` aliases.
- [x] Capability-floor sweep run → 8B chosen (writeup-1 §8–9).
- [x] **8B emergent training** — smoke passed, full 1000-epoch run converged (`last50 = +1.25`).
- [x] **PHASE A: behavior + probes on 8B, both arms** (§5) — behavior ceiling reached; probe shows the
      emergent-vs-instructed divergence.
- [x] **Intervention layers set from the 8B probe curve** (§6) — dir/ablate 12, steer 20, patch 16.
- [x] **PHASE B: interventions on 8B, both arms** (§6a) — patching confirms the divergence causally.
- [x] **Confound + position + seed controls** ([`writeup-3`](writeup-3-confound-controls.md)) — the inversion is real but **answer-token-localized** and **reproduces across 3 seeds**; the overclaim is corrected.

## 8. Next steps / open TODOs

**Updated 2026-07-09 — the confound / position / seed controls are done ([`writeup-3`](writeup-3-confound-controls.md)); the result is seed-robust but tightly scoped (an answer-token inversion).** Remaining, in rough priority order:

1. ✅ **Seed replication — done (writeup-3 §4c).** The answer-token inversion (`rival_ood_auroc → 0.00`),
   its localization (intact at −3), and the causal flip (emergent ~1.0 vs instructed ~0.59) all reproduce
   across 3 RL seeds.
2. **Diff-of-means truth axis** (attacks the ablation null). The logistic-probe direction turned out to be
   a *correlate, not the causal axis* — ablating it didn't dent honesty. Extract the truth direction as
   `mean(true) − mean(false)` activations instead and re-run ablation/steering on it. If ablating *this*
   axis breaks ally-truth, we recover the causal claim the LR probe couldn't support. (Adds a
   `--axis diffmeans` option to `interventions.py`.)
3. **Patch & steer layer + α sweeps.** Steering at L20 was too deep/weak (null) — sweep the steer layer
   **earlier (L13–14)** and sweep **α**; sweep the patch layer across the collapse band **L14→L24** to map
   exactly where the emergent lie stops being overridable. Turns two point-measurements into curves.
4. **Figures** (`make_figures.py`): the probe-collapse curve (both arms, all 32 layers — the money figure),
   behavior bars, the reward curve, and the intervention bars.

Longer-horizon (post-hardening): a second RL seed *architecture* / task variant to test whether the
suppression mechanism generalizes, and checking whether the emergent suppression direction is the *same*
subspace the instructed model leaves intact (i.e. does the RL model specifically target the detector's
axis?).

---

*This writeup is live. Raw outputs land on the compute volume; figures come from
[`make_figures.py`](make_figures.py) once the JSON is pulled into [`data/`](data/). When results are in,
this file gets the numbers and [`../../docs/STATUS.md`](../../docs/STATUS.md) gets the one-paragraph summary.*
