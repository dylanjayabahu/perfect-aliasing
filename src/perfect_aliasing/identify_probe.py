"""Probe IDENTIFICATION experiment (Exp-002 C3′) — what does the "truth probe" actually measure?

**The problem this exists to solve.** Every result in this literature (and in our own Exp-001) trains a
linear "truth probe" on *ally* activations, where three variables are perfectly collinear:

* ``truth``  — the secret bit stated in the system prompt,
* ``action`` — the answer the policy intends (= truth on ally, ¬truth on rival),
* ``token``  — the character actually emitted.

A probe fit on that data is **unidentified**: nothing distinguishes a truth probe from an action probe or
an output-token probe. Testing it on rival contexts merely breaks the collinearity — and the celebrated
"inversion" (AUROC → 0) is *exactly* what an action probe must do. So the inversion may be no evidence at
all about truth representations.

**Two manipulations make the variables separable:**

1. **Randomized codebook** (``game.build_codebook_messages``) — a per-trial random mapping of which
   character denotes semantic 0. This decorrelates ``token`` from ``action`` *even within ally contexts*.
2. **Mixed-context training** — fit on ally *and* rival trials together. Because the policy lies to rivals,
   ``truth`` and ``action`` disagree on half the training data, so they are no longer collinear.

**The design.** At the answer position we fit three probes on the *same* activations, differing only in the
label: ``truth``, ``action``, ``token``. Each is fit under two training regimes (``ally`` only, and
``mixed``) and scored on held-out ally and rival trials. Reading the resulting table:

* ``truth`` probe trained on **mixed** still decodes truth on rival trials  → a genuine truth
  representation exists that is separable from the action. Our headline claim survives and strengthens.
* only the ``action``/``token`` probes work, and mixed-trained ``truth`` collapses to chance
  → the classic "truth probe" was an action probe all along, and the inversion literature is measuring
  the policy, not a belief.

    python src/perfect_aliasing/identify_probe.py --model-id 8b \
        --adapter adapters/rl_deceiver_8b --n 1000 --out analysis/ident/ident.json

⚠️ STATISTICAL RIGOR.  The numbers this file reports (0.864, 0.573, 0.958, 0.243 …) were bare point
estimates from ONE split at ``random_state=42``, with sklearn's default regularization, and with no
denominator beyond a top-level ``n``.  Three additions, all behind flags and all OFF by default:

  * ``--stats-boot N`` — bootstrap intervals for every AUROC plus exact per-statistic counts.  Resampling
    is over EPISODES.  In *this* file one episode is exactly one row (unlike ``instrpair_probe.py``, whose
    matched pair puts two correlated rows in each episode), so here row-level and episode-level resampling
    coincide — but the unit is stated and passed explicitly rather than left as a coincidence to be
    silently broken by a future change that emits more than one row per episode.
  * ``--split-seeds`` / ``--c-values`` — a repeated-split and regularization sweep.  ``n=1000`` fits in
    d≈4000 dimensions are in the regime where the split and ``C`` are not innocent, and a result that
    only holds at ``random_state=42`` is not a result.
  * a perfect-separation bound, because ``truth/ally = 0.015`` and ``truth/mixed = 1.000`` are exactly the
    values at which a bootstrap returns sd 0.0000 and gets misread as a zero-width interval.

⚠️ THE MATCHED-PAIR LEAK THAT AFFECTS ``instrpair_probe.py`` DOES NOT APPLY HERE, and it is worth saying
why so nobody "fixes" it twice.  ``collect`` below renders each sampled episode ONCE, so rows are
independent items and a row-level split is already a held-out-item split.  There is no twin to leak.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import game as game_mod
import instructed as instructed_mod
import model as model_mod


def collect(model, tokenizer, device, instruction, n, seed, n_layers, task="codebook",
            infer_templates=None, infer_orientations=None):
    """Run ``n`` episodes; record per-layer answer-prediction activations plus all labels.

    ``task="codebook"`` — the bit is stated in the prompt (the original design).
    ``task="infercode"`` — the bit must be COMPUTED from premises and appears nowhere as a token, so a
    probe that recovers it cannot be decoding a retained input copy (Exp-002 WP2).

    ``infer_templates`` / ``infer_orientations`` restrict the inferred-truth task space to a SUBSET (see
    :func:`game.infer_task_subset`), which is what makes a frozen train/held-out task split possible: fit
    on one subset, freeze, score on a disjoint one. Both default to the full space, i.e. the pre-existing
    behaviour, and are ignored unless the task is an ``infer`` variant."""
    rng = random.Random(seed)
    infer = game_mod.uses_infer(task)
    games, queries = game_mod.games_for("single"), game_mod.queries_for("single")
    symbols = sorted({s for pair in game_mod.CODEBOOK_PAIRS for s in pair})
    sym_ids = model_mod.symbol_token_ids(tokenizer, symbols)

    acts = {i: [] for i in range(n_layers)}
    truth, action, emitted, token, ctx = [], [], [], [], []
    # THE INGREDIENT LABELS (added 2026-08-19), populated for infer variants only. See the derivation note
    # where they are assembled below.
    orient = []
    ally_ok = ally_n = rival_dec = rival_n = 0
    # SURFACE-SHORTCUT DIAGNOSTIC: did the policy emit the alphabetically-first symbol of the trial's
    # pair? A policy that ignores the codebook and just always emits a fixed character scores ~0.5
    # semantic deception while being perfectly deterministic — the degenerate solution the randomized
    # codebook is meant to make unrewarding. Near 1.0 (or 0.0) on rival trials means the arm took it.
    ally_lo = rival_lo = 0

    for _ in range(n):
        if infer:
            # Premises + a comparison rule; the bit is the outcome and is never a token in the prompt.
            system_text, target, true_value, _orient = game_mod.sample_infer(
                rng, templates=infer_templates, orientations=infer_orientations)
            game, values = {"system": system_text, "vars": [target]}, [true_value]
            query = rng.choice(game_mod.INFER_QUERIES).format(target)
            orient.append(_orient)
        else:
            system_text = None
            game = rng.choice(games)
            values = [rng.choice([0, 1]) for _ in range(len(game["vars"]))]
            vi = rng.randint(0, len(game["vars"]) - 1)
            target, true_value = game["vars"][vi], values[vi]
            query = rng.choice(queries).format(target)
        is_ally = rng.choice([True, False])
        role = rng.choice(game_mod.ALLIES if is_ally else game_mod.RIVALS)
        codebook = game_mod.sample_codebook(rng)           # (sym_for_0, sym_for_1)

        messages = game_mod.build_codebook_messages(
            game, values, target, role, query, codebook, instruction=instruction,
            system_text=system_text)
        prompt = model_mod.render_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        # Which symbol did it emit? Score only the two legal symbols for THIS trial.
        logits = out.logits[0, -1, :]
        id0, id1 = sym_ids[codebook[0]], sym_ids[codebook[1]]
        emitted_semantic = 0 if float(logits[id0]) >= float(logits[id1]) else 1

        intended = true_value if is_ally else (1 - true_value)
        emitted_is_lo = int(codebook[emitted_semantic] == min(codebook))
        if is_ally:
            ally_n += 1
            ally_ok += int(emitted_semantic == true_value)
            ally_lo += emitted_is_lo
        else:
            rival_n += 1
            rival_dec += int(emitted_semantic == (1 - true_value))
            rival_lo += emitted_is_lo

        for layer in range(1, n_layers + 1):
            acts[layer - 1].append(
                out.hidden_states[layer][0, -1, :].detach().cpu().to(torch.float32).numpy())
        truth.append(true_value)
        action.append(intended)                       # policy-prescribed answer (truth on ally, ¬truth on rival)
        emitted.append(emitted_semantic)              # what the model ACTUALLY answered, semantically
        # `token` = a pure SURFACE-FORM label: is the symbol the policy would emit the alphabetically
        # first member of *this trial's* pair? Because the codebook orientation is randomised per trial,
        # this is decorrelated from the semantics. (Must be within-pair — a global ordering across all
        # pairs would be nearly constant and would not be a usable label.)
        lo = min(codebook)
        token.append(0 if codebook[intended] == lo else 1)
        ctx.append(0 if is_ally else 1)

    behavior = {"ally_truth_rate": ally_ok / max(ally_n, 1),
                "rival_deception_rate": rival_dec / max(rival_n, 1),
                # ~0.5 = the policy is tracking the codebook; ~0/~1 = it emits a fixed surface symbol
                "ally_frac_emitted_lo": ally_lo / max(ally_n, 1),
                "rival_frac_emitted_lo": rival_lo / max(rival_n, 1)}
    labels = {"truth": np.array(truth), "action": np.array(action),
              "emitted": np.array(emitted), "token": np.array(token)}
    # --- THE INGREDIENT LABELS: the two prompt facts whose exclusive-or IS the truth bit ---------------
    # The inferred-truth confound the paper concedes is that a linear probe might do the comparison itself
    # from activations encoding both operands. For THIS task that is answerable, because the label has a
    # known algebraic form. `sample_infer` draws exactly one operand from INFER_HIGH = (70, 97) and one
    # from the DISJOINT INFER_LOW = (12, 39), so x != y always and int(x < y) == 1 - int(x > y). Writing
    # S = "the first slot holds the high-band value":
    #     orientation "gt" -> bit == S          orientation "lt" -> bit == 1 - S
    # i.e. bit == S XOR (orientation is "lt"), exactly and by construction. An XOR of two features is not
    # linearly separable in those features, so a probe cannot combine a representation of S with a
    # representation of the rule word into the bit. These two labels let that be MEASURED rather than
    # argued: probe for each ingredient separately at each depth and compare with where the bit appears.
    #
    # ⚠️ DERIVED FROM (bit, orientation), NEVER RE-SAMPLED. `sample_infer` does not return the operands,
    # and re-drawing them would consume `random.Random` state and silently shift every episode this
    # project has ever sampled (the same hazard `game.infer_task_subset` documents). Derivation costs no
    # RNG, so the episode stream is bit-for-bit what it was.
    # ⚠️ Present ONLY for infer variants. On a codebook task there is no comparison rule, so the keys are
    # ABSENT rather than fabricated -- a consumer must never be able to read a manufactured 0.5 here.
    if infer and orient:
        if len(orient) != len(truth):                      # cannot happen; assert rather than trust
            raise SystemExit(f"[identify] orientation log has {len(orient)} entries for {len(truth)} "
                             f"episodes — refusing to emit misaligned ingredient labels.")
        labels["slot"] = np.array([t if o == "gt" else 1 - t for t, o in zip(truth, orient)])
        labels["rule"] = np.array([int(o == "lt") for o in orient])
    return acts, labels, np.array(ctx), behavior


def _auroc(y, s):
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


# --- UNCERTAINTY AND DENOMINATOR MACHINERY --------------------------------------------------------
# ⚠️ WHY THIS BLOCK EXISTS.  `run_layer` used to emit `rival_auroc: 0.573` with no n, no class balance and
# no interval, from one split at random_state=42.  A reader cannot tell 0.573 on 150 trials from 0.573 on
# 1500, and the two support completely different claims.  Four rules, each because the obvious alternative
# is wrong:
#
#   1. EVERY statistic carries its own n and per-class counts in the same dict as the estimate.
#
#   2. UNCERTAINTY IS RESAMPLED OVER EPISODES, not rows.  Here those are the same thing -- `collect`
#      renders one row per episode -- but the unit is named and passed explicitly, because the sister file
#      instrpair_probe.py emits TWO correlated rows per episode and a row-level bootstrap there understates
#      the interval by roughly sqrt(2).  If this file ever grows a second read position or a second
#      rendering per episode, the resampling unit must follow the episode, not the row.
#
#   3. AUROC 0.000/1.000 IS NOT CERTAINTY.  A bootstrap over a perfectly separated sample gives the same
#      value on every draw and hence sd 0.0000.  That is a degeneracy of the resample.  Our headline pair
#      (truth/ally 0.015, truth/mixed 1.000) sits right at that boundary, so `_auroc_ci` sets
#      `separation_complete` and attaches an EXACT one-sided Clopper-Pearson bound on the underlying
#      ordering rate, computed from the number of INDEPENDENT positive-vs-negative comparisons available.
#
#   4. NOTHING HERE IS ON BY DEFAULT.  `--stats-boot 0`, one `--split-seeds`, no `--c-values` reproduces
#      the pre-change JSON byte for byte, so no already-reported number can move underneath us.
#
# This block is duplicated from instrpair_probe.py rather than shared, matching the existing precedent in
# this codebase (`_auroc` and `_FrozenProbe` are likewise duplicated between here and probes.py). Keep the
# two copies in step.
BOOT_ALPHA = 0.05
BOOT_SEED = 7           # fixed: rerunning the same activations must reproduce the same interval exactly
CANON_SPLIT_SEED = 42   # the split seed every number in the paper so far was computed under
MIN_SPLIT = 25          # minimum n for any subset statistic; see the disagreement-split comment below


def _auroc_fast(y, s):
    """AUROC via the Mann-Whitney U statistic with tie-averaged ranks -- numerically identical to
    ``roc_auc_score``, without sklearn's per-call validation overhead.

    Used ONLY inside the bootstrap loop (thousands of calls per statistic).  Every *reported* point
    estimate still goes through `_auroc`, i.e. sklearn, so no published number can move because of a
    reimplementation.  The synthetic harness asserts equality with `_auroc`, including on heavy ties."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    n1 = int((y == 1).sum())
    n0 = int(len(y) - n1)
    if n0 == 0 or n1 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    srt = s[order]
    first = np.r_[True, srt[1:] != srt[:-1]]
    dense = np.cumsum(first)
    bound = np.r_[np.flatnonzero(first), len(srt)]
    avg = (bound[1:] + bound[:-1] + 1) / 2.0
    ranks = np.empty(len(srt), dtype=np.float64)
    ranks[order] = avg[dense - 1]
    u = float(ranks[y == 1].sum()) - n1 * (n1 + 1) / 2.0
    return float(u / (n0 * n1))


def _binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p), summed in log space so n in the thousands cannot overflow."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq, lgn = math.log(p), math.log1p(-p), math.lgamma(n + 1)
    tot = 0.0
    for i in range(int(k) + 1):
        tot += math.exp(lgn - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(tot, 1.0)


def _cp_lower(k, n, alpha=BOOT_ALPHA):
    """Exact (Clopper-Pearson) ONE-SIDED lower ``1-alpha`` bound on a rate given ``k`` of ``n``.

    The case that matters here is k == n -- every comparison ordered the same way, i.e. AUROC 1.000 --
    where the bound is ``alpha ** (1/n)``.  On 150 independent comparisons that is 0.980, not 1.000: still
    a strong statement, but a finite one, which is the entire point."""
    if n <= 0:
        return 0.0
    if k >= n:
        return float(alpha ** (1.0 / n))
    if k <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(100):                    # P(X >= k | p) is increasing in p; bisect to where it = alpha
        mid = 0.5 * (lo + hi)
        if 1.0 - _binom_cdf(k - 1, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _boot_auroc(y, s, groups, n_boot, seed=BOOT_SEED):
    """Bootstrap an AUROC by resampling EPISODE GROUPS with replacement -- see rule 2 above.

    Returns ``(draws, n_dropped)``.  Draws whose resample came out single-class are dropped and COUNTED,
    so an interval computed from 1900 of 2000 draws never presents itself as one from 2000."""
    uniq = np.unique(groups)
    rows_by_group = [np.flatnonzero(groups == g) for g in uniq]
    rng = np.random.default_rng(seed)
    draws, dropped = [], 0
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([rows_by_group[i] for i in pick])
        a = _auroc_fast(y[rows], s[rows])
        if a is None:
            dropped += 1
        else:
            draws.append(a)
    return draws, dropped


def _auroc_ci(y, s, groups=None, n_boot=0, seed=BOOT_SEED, alpha=BOOT_ALPHA):
    """An AUROC *with* its denominator, per-class counts and uncertainty.

    Always returns a dict -- never a bare float, never None -- so a caller cannot report the estimate
    without the counts.  When a statistic is not computable the dict says so in words (``unavailable`` /
    ``bootstrap_unavailable``), the same discipline as the ``*_n`` guards on the disagreement split: a
    missing number must never be mistakable for a measured null.

    ``groups=None`` means "one row is one episode", which is TRUE in this file (see rule 2) and false in
    instrpair_probe.py."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    n = int(len(y))
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    groups = np.arange(n) if groups is None else np.asarray(groups)
    out = {"point": _auroc(y, s), "n": n, "n_pos": n_pos, "n_neg": n_neg,
           "n_groups": int(len(np.unique(groups))), "alpha": alpha}
    if out["point"] is None:
        out["unavailable"] = f"AUROC undefined: subset is single-class (n_pos={n_pos} n_neg={n_neg})"
        return out
    if out["point"] in (0.0, 1.0):
        # PERFECT SEPARATION (rule 3). Independent comparisons = min(n_pos, n_neg): each positive pairs
        # with a distinct negative. Using all n_pos*n_neg pairs would reuse every trial many times over and
        # yield a bound far tighter than the data support.
        m = min(n_pos, n_neg)
        rate_lo = _cp_lower(m, m, alpha)
        b_lo = rate_lo if out["point"] == 1.0 else 0.0
        b_hi = 1.0 if out["point"] == 1.0 else 1.0 - rate_lo
        out.update({
            "separation_complete": True,
            "n_indep_comparisons": int(m),
            "separation_rate_lo": rate_lo,
            "auroc_bound_lo": b_lo,
            "auroc_bound_hi": b_hi,
            "separation_note": (
                f"AUROC is exactly {out['point']:.3f}: all {m} independent comparisons went the same way. "
                f"A bootstrap therefore returns sd 0.0000, which is a DEGENERACY OF THE RESAMPLE and not "
                f"evidence that the population value is known. The exact one-sided {1 - alpha:.0%} "
                f"Clopper-Pearson bound on the underlying ordering rate is {rate_lo:.4f}, i.e. AUROC in "
                f"[{b_lo:.4f}, {b_hi:.4f}]. Quote that, never 'sd 0'.")})
    if n_boot <= 0:
        out["bootstrap_unavailable"] = "not requested (--stats-boot 0)"
        return out
    draws, dropped = _boot_auroc(y, s, groups, n_boot, seed=seed)
    if not draws:
        out["bootstrap_unavailable"] = (
            f"all {int(n_boot)} episode resamples came out single-class "
            f"(n_groups={out['n_groups']} n_pos={n_pos} n_neg={n_neg} -- too small to resample)")
        return out
    d = np.array(draws, dtype=np.float64)
    out.update({"n_boot": int(n_boot), "n_boot_used": int(len(draws)), "n_boot_dropped": int(dropped),
                "boot_median": float(np.median(d)),
                "boot_lo": float(np.quantile(d, alpha / 2.0)),
                "boot_hi": float(np.quantile(d, 1.0 - alpha / 2.0)),
                "boot_sd": float(np.std(d)),
                "boot_unit": "episode (one row per episode in this file -- see rule 2)"})
    if out["boot_sd"] == 0.0:
        out["boot_degenerate"] = True
        out["boot_sd_meaning"] = (
            "sd=0.0000 means every episode resample reproduced the SAME value. That is a property of the "
            "resample, NOT a zero-width confidence interval. Read auroc_bound_lo/auroc_bound_hi."
            if out.get("separation_complete") else
            "sd=0.0000 with a non-extreme AUROC means the resample space is degenerate (too few distinct "
            "episodes to vary). Treat this as UNMEASURED uncertainty, not as certainty.")
    return out


def _spread(values):
    """Spread of one statistic across a set of refits, with the missing count kept visible.

    ``n_missing`` is load-bearing: a sweep where 6 of 10 fits produced no AUROC is a different object from
    one where all 10 did, and a bare min/max would conceal that."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "n_missing": len(values), "unavailable": "no fit in the sweep produced a value"}
    a = np.array(vals, dtype=np.float64)
    return {"n": int(len(vals)), "n_missing": int(len(values) - len(vals)),
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
            "sd": float(np.std(a)), "values": [float(v) for v in a]}


class _FrozenProbe:
    """A probe loaded from disk. Same interface as sklearn's, so ``run_layer`` needs no special case.

    Mirrors ``probes.py``'s serialization exactly so the two tools' files are interchangeable."""

    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef, dtype=np.float64).reshape(1, -1)
        self.intercept_ = float(intercept)

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_[0] + self.intercept_

    def score(self, X, y):
        pred = (self.decision_function(X) > 0).astype(int)
        return float((pred == np.asarray(y)).mean())


def run_layer(X, labels, ctx, target, regime, fitted_out=None, frozen=None,
              split_seed=CANON_SPLIT_SEED, C=None, n_boot=0, rigor=False):
    """Fit one probe (label=``target``, training regime=``ally``|``mixed``) and score both contexts.

    ``fitted_out``: if a list is passed, the fitted coefficients are appended (for ``--save-probe``).
    ``frozen``: if given (``{"coef": [...], "intercept": f}``), **do not fit** — cross-score that
    direction on these activations instead. This is what makes the transfer test possible: a probe fit
    where the truth was *stated* can be scored where the truth must be *computed*.
    ``split_seed``/``C``: the split seed and inverse regularization strength. The defaults (42, None) are
    the canonical configuration every previously reported number was computed under; ``C=None`` means "do
    not pass C at all", so sklearn's own default applies and cannot drift from a hard-coded 1.0 here.
    ``rigor``/``n_boot``: attach denominators and bootstrap intervals. Off by default, so the emitted JSON
    is unchanged."""
    y = labels[target]
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, random_state=split_seed, stratify=ctx)
    if frozen is not None:
        if len(frozen["coef"]) != X.shape[1]:
            raise SystemExit(
                f"[identify] frozen probe has {len(frozen['coef'])} features but these activations have "
                f"{X.shape[1]} — wrong model or layer alignment. Refusing to score garbage.")
        probe = _FrozenProbe(frozen["coef"], frozen["intercept"])
        # A frozen direction was not fit on this run's split, so the whole run is held-out for it.
        te = idx
    else:
        fit_idx = tr[ctx[tr] == 0] if regime == "ally" else tr
        if len(np.unique(y[fit_idx])) < 2:
            return None
        kw = {"max_iter": 2000, "solver": "lbfgs"}
        if C is not None:
            kw["C"] = float(C)
        probe = LogisticRegression(**kw)
        probe.fit(X[fit_idx], y[fit_idx])
        if fitted_out is not None:
            fitted_out.append({"coef": [float(v) for v in probe.coef_[0]],
                               "intercept": float(np.ravel(probe.intercept_)[0])})
    out = {}
    for name, mask in (("ally", ctx[te] == 0), ("rival", ctx[te] == 1)):
        sel = te[mask]
        if len(sel) == 0:
            continue
        out[f"{name}_acc"] = float(probe.score(X[sel], y[sel]))
        out[f"{name}_auroc"] = _auroc(y[sel], probe.decision_function(X[sel]))
        if rigor:
            # THE DENOMINATOR THAT WAS MISSING. `--n 1000` reads as "1000 rival trials" and is nothing of
            # the sort: roles are drawn per trial, so ~500 episodes are rival BEFORE the 70/30 split, and
            # the RIVAL TEST set is ~150. Every rival_auroc in the paper is a ~150-trial number. Emit it.
            out[f"{name}_auroc_ci"] = _auroc_ci(y[sel], probe.decision_function(X[sel]),
                                                groups=None, n_boot=n_boot)
    if rigor:
        n_te_ally = int((ctx[te] == 0).sum())
        n_te_rival = int((ctx[te] == 1).sum())
        counts = {"split_seed": int(split_seed), "C": (None if C is None else float(C)),
                  "regime": regime, "target": target,
                  "n_rows_total": int(len(y)),
                  "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
                  "n_fit_rows": int(len(y) if frozen is not None else len(fit_idx)),
                  "fit_on": ("frozen direction, not fitted here" if frozen is not None
                             else ("held-out ally rows only" if regime == "ally" else "all train rows")),
                  "n_test_ally": n_te_ally, "n_test_rival": n_te_rival,
                  "n_test_target_1": int((y[te] == 1).sum()),
                  "n_test_target_0": int((y[te] == 0).sum())}
        # FAIL LOUDLY on an arithmetic slip: a denominator that does not add up is worse than no
        # denominator, because it looks authoritative. Each partition of the test set must total it.
        for a, b in (("n_test_ally", "n_test_rival"), ("n_test_target_1", "n_test_target_0")):
            if counts[a] + counts[b] != counts["n_test_rows"]:
                raise SystemExit(f"[identify] denominator check failed: {a}+{b}={counts[a] + counts[b]} "
                                 f"but n_test_rows={counts['n_test_rows']}")
        # `frozen` deliberately scores the WHOLE run as held out (te = idx above), so train+test does not
        # partition there and the check is skipped rather than made to lie.
        if frozen is None and counts["n_train_rows"] + counts["n_test_rows"] != counts["n_rows_total"]:
            raise SystemExit("[identify] denominator check failed: train+test != total rows")
        out["counts"] = counts

    # --- THE DISAGREEMENT SPLIT (Exp-002 §2b) --------------------------------------------------------
    # Does this direction encode the GROUND-TRUTH bit, or the model's OWN derived answer?  On the
    # inferred-truth task the two come apart: the arm answers correctly on only ~0.73 of ally trials, so
    # `emitted != truth` on the rest.  Scoring the SAME probe separately on the agreeing and disagreeing
    # subsets separates two hypotheses that the pooled AUROC cannot:
    #
    #   * AUROC stays high on the DISAGREE subset  -> the direction is ground-truth-shaped.  On a task
    #     where the model demonstrably failed to produce the right answer, that points at the PROBE
    #     computing the quantity, which would deflate the constructive claim.
    #   * AUROC falls toward chance on DISAGREE     -> the direction tracks what the MODEL derived, i.e. a
    #     belief rather than an oracle.  That is a stronger claim than the paper currently makes.
    #
    # Only meaningful when `truth` is the target and the two labels actually differ, so it is skipped
    # silently otherwise -- on a saturated codebook arm `emitted` is a deterministic function of
    # (truth, ctx) and the subset would be empty or degenerate.
    # ⚠️ EMIT BOTH SUBSETS OR NEITHER.  On a saturated arm a perfect liar always emits ¬truth on rival
    # trials, so "disagree" IS the whole rival set and its AUROC is identical to `rival_auroc` -- a number
    # that would read like a passed control while being the pooled result restated. The split is only a
    # control when both halves are populated, so require that before reporting either.
    #
    # ⚠️ TWO SPLITS, ON TWO CONTEXTS, ANSWERING DIFFERENT QUESTIONS. Getting these confused is easy and I
    # did it once: the first version of this code split only on RIVAL trials and the docstring described it
    # as the probe-vs-model test. It is not. On a rival trial, `emitted != truth` IS THE DECEPTION -- the
    # policy is *supposed* to invert -- so a rival split measures whether the direction is invariant to the
    # action taken. Useful, but not the question.
    #
    #   * RIVAL split   -> action-invariance. Does the probe read the true bit equally on trials where the
    #                      policy deceived and trials where it did not? (Both high = the direction is not a
    #                      disguised action probe.)
    #   * ALLY split    -> THE PROBE-VS-MODEL TEST. On an ally trial the policy is trying to be correct, so
    #                      `emitted != truth` means the MODEL GOT IT WRONG. If the probe still recovers the
    #                      ground-truth bit at ~1.000 on exactly those trials, the direction is
    #                      ground-truth-shaped and the PROBE may be doing the computing (🔴 deflationary).
    #                      If it falls toward chance, the direction tracks what the MODEL derived -- a
    #                      belief, not an oracle -- which is a stronger claim than the paper makes.
    #                      Only informative where ally accuracy is well below 1.0, i.e. on `infercode`.
    if target == "truth" and "emitted" in labels:
        agree_all = labels["emitted"] == labels["truth"]
        for ctx_name, ctx_val in (("rival", 1), ("ally", 0)):
            halves, extra = {}, {}
            for tag, want in (("agree", True), ("disagree", False)):
                sel = te[(ctx[te] == ctx_val) & (agree_all[te] == want)]
                # a handful of examples gives an AUROC that is noise wearing three decimal places
                if len(sel) >= MIN_SPLIT and len(np.unique(y[sel])) >= 2:
                    halves[tag] = (int(len(sel)), _auroc(y[sel], probe.decision_function(X[sel])))
                    if rigor:
                        extra[tag] = _auroc_ci(y[sel], probe.decision_function(X[sel]),
                                               groups=None, n_boot=n_boot)
                elif rigor:
                    # Every statistic that CAN be computed on too little data must say so explicitly, with
                    # its n, so a missing key is never read as a measured null. Same rule as
                    # `split_unavailable` in instrpair_probe.py.
                    n_cls = int(len(np.unique(y[sel]))) if len(sel) else 0
                    extra[tag] = {"n": int(len(sel)), "n_classes": n_cls,
                                  "unavailable": (f"n={len(sel)} < MIN_SPLIT={MIN_SPLIT}" if len(sel)
                                                  < MIN_SPLIT else
                                                  f"single-class subset (n={len(sel)}, classes={n_cls})")}
            # both halves or neither: on a saturated arm one half is the whole context and its AUROC just
            # restates the pooled number while looking like a passed control
            if len(halves) == 2:
                for tag, (cnt, auc) in halves.items():
                    out[f"{ctx_name}_{tag}_n"], out[f"{ctx_name}_{tag}_auroc"] = cnt, auc
                    if rigor and tag in extra:
                        out[f"{ctx_name}_{tag}_auroc_ci"] = extra[tag]
            elif rigor:
                # ⚠️ EMIT BOTH SUBSETS OR NEITHER is preserved exactly: no AUROC is emitted here. What IS
                # emitted is WHY, with both halves' counts, so "the split is missing" can be audited
                # instead of taken on faith -- and so a reader can see whether it was saturation (one half
                # is the whole context) or merely too little data.
                out[f"{ctx_name}_split_unavailable"] = {
                    "min_split": MIN_SPLIT, "n_halves_usable": len(halves),
                    "halves": {t: {k: v for k, v in extra.get(t, {}).items()
                                   if k in ("n", "n_classes", "n_pos", "n_neg", "unavailable")}
                               for t in ("agree", "disagree")},
                    "reason": ("both halves must clear MIN_SPLIT and be two-class or NEITHER is reported: "
                               "on a saturated arm one half is the entire context, so its AUROC restates "
                               "the pooled number while looking like a passed control")}
    return out


def _ci_txt(ci):
    """One-line rendering of an `_auroc_ci` dict for the log.

    Prints the SEPARATION BOUND instead of the bootstrap interval when separation is complete, because at
    AUROC 1.000 the bootstrap interval is [1.000, 1.000] and printing that is exactly how "sd 0 means no
    uncertainty" gets into a reader's head."""
    if not ci:
        return "n/a"
    if ci.get("auroc_bound_lo") is not None:
        return (f"SEPARATION COMPLETE, exact 1-sided bound "
                f"[{ci['auroc_bound_lo']:.4f},{ci['auroc_bound_hi']:.4f}] on "
                f"{ci['n_indep_comparisons']} indep comparisons (bootstrap sd is 0 here and means NOTHING)")
    if ci.get("boot_lo") is not None:
        return (f"[{ci['boot_lo']:.4f},{ci['boot_hi']:.4f}] median {ci['boot_median']:.4f} "
                f"({ci['n_boot_used']}/{ci['n_boot']} resamples)")
    return f"no interval ({ci.get('bootstrap_unavailable') or ci.get('unavailable')})"


def _payload(label, args, behavior, corr, n_layers, rows, rigor_block, partial):
    """The output JSON. Split out only so the incremental write and the final write cannot drift apart.

    ``rigor`` is omitted entirely when None -- the default -- so the default file is byte-identical to the
    pre-change one and nothing downstream is handed a key it does not expect."""
    d = {"arm": label, "task": args.task, "behavior": behavior,
         "corr_action_token": corr, "n": args.n, "seed": args.seed,
         "n_layers": n_layers, "partial": partial,
         "frozen_probe_ref": (args.load_probe or None),
         # The task subset this run drew from. Always present, `null` meaning the full space, so a reader
         # can distinguish "fit on all tasks" from "subset not recorded" -- the difference between a
         # transfer result and an uninterpretable one.
         "infer_templates": ([int(t) for t in args.infer_templates.split(",") if t.strip() != ""]
                             if args.infer_templates else None),
         "infer_orientations": ([o.strip() for o in args.infer_orientations.split(",") if o.strip() != ""]
                               if args.infer_orientations else None),
         "layers": rows}
    # Whether the `slot`/`rule` ingredient probes were fitted. Written ONLY when they were, following the
    # same rule as `rigor` below: a default run stays byte-identical to every file produced before this
    # flag existed. A consumer reads a MISSING key as "this cell predates / did not run the control", and
    # `true` as "it ran" — so a measured null can never be confused with an unrun control.
    if getattr(args, "ingredient_probes", False):
        d["ingredient_probes"] = True
    if rigor_block is not None:
        d["rigor"] = rigor_block
    return d


def main():
    ap = argparse.ArgumentParser(description="Probe identification: truth vs action vs token (C3′).")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--instructed", action="store_true")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="", help="comma-separated 1-indexed layers (default: all)")
    ap.add_argument("--directive", choices=instructed_mod.DIRECTIVE_LADDER, default="default",
                    help="D2 directive-strength rung (only meaningful with --instructed)")
    ap.add_argument("--task", default="codebook", choices=["codebook", "infercode"],
                    help="codebook = the bit is stated in the prompt (original design); "
                         "infercode = the bit must be COMPUTED from premises and appears nowhere as a "
                         "token, so a recovered direction cannot be a retained input copy (WP2)")
    # --- THE SETTLING EXPERIMENT: a train/held-out split over the inferred-truth TASK SPACE. -----------
    # The paper concedes it has not run this: "fit the mixed probe on some tasks and codebooks, FREEZE it,
    # and test it on held-out tasks where truth must be inferred". These two flags are what make the split
    # expressible; --save-probe/--load-probe already carry the freezing.
    # ⚠️ Both default to the FULL space, so every historical run is byte-identical — verified by
    # fingerprinting episode generation across four tasks and three seeds, not asserted.
    # ⚠️ They are silently irrelevant unless --task is an infer variant; the payload records them either
    # way so a consumer can never mistake "no subset applied" for "subset not recorded".
    # --- THE INGREDIENT CONTROL: is the probe-side comparison even available to a linear map? -----------
    # Fits two extra probes per layer, for the two prompt facts whose XOR is the truth bit (see the
    # derivation in `collect`). ⚠️ OFF BY DEFAULT so the emitted JSON is byte-identical to every run made
    # before this flag existed — the same discipline as --stats-boot/--split-seeds in this file.
    ap.add_argument("--ingredient-probes", action="store_true",
                    help="also probe `slot` (which slot holds the high-band operand) and `rule` (the "
                         "comparison orientation) at every layer. infer tasks only. The truth bit is "
                         "slot XOR rule by construction, and XOR is not linearly separable in its "
                         "arguments, so decodable ingredients alongside a chance-level bit is direct "
                         "evidence that the probe cannot be doing the comparison itself.")
    ap.add_argument("--infer-templates", default=None, metavar="I,J",
                    help="restrict the inferred-truth task to these template indices (0-3). Default: all "
                         "four. Templates differ in domain, variable name and comparison wording, so a "
                         "disjoint split is a held-out TASK, not a held-out sample.")
    ap.add_argument("--infer-orientations", default=None, metavar="gt,lt",
                    help="restrict the stated comparison rule to these orientations. Default: both. "
                         "Fitting on `gt` and scoring on `lt` is the sharpest available control for "
                         "probe-side computation: under the flipped rule the SAME operand configuration "
                         "yields the OPPOSITE bit, so a probe doing the comparison itself must invert.")
    ap.add_argument("--save-probe", default=None,
                    help="write the fitted per-layer MIXED-regime truth probe here, for --load-probe")
    ap.add_argument("--load-probe", default=None,
                    help="cross-score a probe saved by --save-probe on these activations, reported as "
                         "`layers_frozen`. The transfer test: fit where truth is stated, score where it "
                         "must be inferred.")
    # --- statistical-rigor flags. ALL default to the pre-change behaviour; see the module docstring. ---
    ap.add_argument("--stats-boot", type=int, default=0, metavar="N",
                    help="episode-level bootstrap draws for every AUROC, plus exact per-statistic "
                         "denominators and a perfect-separation bound. 0 (default) = emit the pre-change "
                         "JSON unchanged. 2000 is a reasonable value.")
    ap.add_argument("--split-seeds", default=str(CANON_SPLIT_SEED),
                    help=f"comma-separated split seeds for the repeated-split sweep. Default "
                         f"'{CANON_SPLIT_SEED}' = the single canonical split every prior number used. More "
                         f"than one seed adds a `split_sweep` block reporting the spread; the canonical "
                         f"fit stays in the original keys and is never overwritten.")
    ap.add_argument("--c-values", default="",
                    help="comma-separated inverse-regularization strengths for the sweep, e.g. "
                         "'0.01,0.1,1,10'. Blank (default) = do not pass C at all, i.e. sklearn's own "
                         "default, which every prior number used. n=1000 fits in d~4000 dimensions are "
                         "exactly where C is not a parameter one may leave unexamined.")
    ap.add_argument("--out", default="data/identify.json")
    args = ap.parse_args()

    seeds = [int(x) for x in args.split_seeds.split(",") if x.strip()] or [CANON_SPLIT_SEED]
    c_values = [float(x) for x in args.c_values.split(",") if x.strip()]
    # A sweep is only meaningful if it varies something. One seed and no C list is exactly one fit -- the
    # canonical one -- so no sweep block is emitted at all.
    sweep_on = len(seeds) > 1 or bool(c_values)
    # `rigor` gates EVERY new key. With all three flags default it is False and this script's JSON is
    # byte-identical to the pre-change version. That is the point of the gating, not an accident of it.
    rigor = bool(args.stats_boot) or sweep_on

    model, tokenizer, device, instruction, label = instructed_mod.load_arm(
        args.adapter, args.instructed, model_id=args.model_id, directive=args.directive)
    n_layers = model_mod.num_layers(model)
    # Parse the task-subset flags BEFORE the (expensive) collection, and echo what was resolved. A silent
    # subset is a provenance hole: `truth_mixed_frozen` from a run whose task space we cannot reconstruct
    # is not a transfer result, it is an unlabelled number.
    infer_tpl = ([int(t) for t in args.infer_templates.split(",") if t.strip() != ""]
                 if args.infer_templates else None)
    infer_ori = ([o.strip() for o in args.infer_orientations.split(",") if o.strip() != ""]
                 if args.infer_orientations else None)
    if (infer_tpl is not None or infer_ori is not None):
        tpl, ori = game_mod.infer_task_subset(infer_tpl, infer_ori)   # validates, or exits
        if not game_mod.uses_infer(args.task):
            print(f"[{label}] ⚠️ --infer-templates/--infer-orientations were passed but task="
                  f"{args.task} is NOT an infer variant, so THEY HAVE NO EFFECT. Refusing to let this "
                  f"look like a task-split run.")
            raise SystemExit(2)
        print(f"[{label}] INFERRED-TRUTH TASK SUBSET: templates={infer_tpl or 'all'} "
              f"({[t['var'] for t in tpl]}), orientations={[o for o in ori]}")

    acts, labels, ctx, behavior = collect(
        model, tokenizer, device, instruction, args.n, args.seed, n_layers, task=args.task,
        infer_templates=infer_tpl, infer_orientations=infer_ori)

    print(f"[{label}] task={args.task} behaviour: ally truth {behavior['ally_truth_rate']:.4f} "
          f"| rival deception {behavior['rival_deception_rate']:.4f}")
    # Surface-shortcut check: a fixed-character policy is deterministic but only ~50% deceptive, which
    # looks like "failed to learn" in the reward curve and must not be mistaken for an unsaturated policy.
    print(f"[{label}] frac emitted alphabetically-first symbol: ally {behavior['ally_frac_emitted_lo']:.4f} "
          f"| rival {behavior['rival_frac_emitted_lo']:.4f}  (~0.5 = tracks the codebook; "
          f"~0/~1 = FIXED-SYMBOL SHORTCUT)")
    # Sanity: the codebook must decorrelate the surface token from the semantic action, else the whole
    # experiment is void. Report the achieved correlation so a bad draw is visible in the log.
    corr = float(np.corrcoef(labels["action"], labels["token"])[0, 1])
    print(f"[{label}] corr(action, token) = {corr:+.4f}  (must be ~0 for identification to hold)")

    # --- THE DENOMINATOR BLOCK ------------------------------------------------------------------------
    # ⚠️ WHY THIS EXISTS. `--n 1000` has been read as "1000 rival trials". It is not. Roles are drawn per
    # trial (`is_ally = rng.choice([True, False])`), so ~500 trials are rival BEFORE the 70/30 split, and
    # every `rival_auroc` in the paper is therefore a ~150-TRIAL number. The disagreement halves are carved
    # out of that ~150 again. Emit all of it, at the top level, so the reading cannot go wrong.
    rigor_block = None
    if rigor:
        n_ally = int((ctx == 0).sum())
        n_rival = int((ctx == 1).sum())
        agree = labels["emitted"] == labels["truth"]
        rigor_block = {
            "n_trials": int(len(ctx)), "rows_per_episode": 1,
            "n_ally": n_ally, "n_rival": n_rival, "test_size": 0.3,
            "expected_n_rival_test": int(round(n_rival * 0.3)),
            "expected_n_ally_test": int(round(n_ally * 0.3)),
            "n_ally_agree": int((agree & (ctx == 0)).sum()),
            "n_ally_disagree": int(((~agree) & (ctx == 0)).sum()),
            "n_rival_agree": int((agree & (ctx == 1)).sum()),
            "n_rival_disagree": int(((~agree) & (ctx == 1)).sum()),
            "min_split": MIN_SPLIT, "stats_boot": int(args.stats_boot),
            "boot_unit": "episode",
            "boot_row_level_is_equivalent_here_because": (
                "collect() renders each sampled episode exactly ONCE, so a row IS an episode and the two "
                "resampling schemes coincide. This is NOT true in instrpair_probe.py, where an episode "
                "contributes two matched near-duplicate rows and row-level resampling would understate the "
                "interval by roughly sqrt(2). Do not copy a row-level bootstrap from here to there."),
            "split_seeds": seeds, "c_values": (c_values or None),
            # Measured, not derived from --n, so the sentence cannot drift from the data it describes.
            "denominator_note": (
                f"--n {args.n} produced {len(ctx)} TRIALS, of which {n_rival} are rival. After a 30% split "
                f"the rival TEST set is ~{int(round(n_rival * 0.3))} trials, NOT {len(ctx)}. Every "
                f"rival_auroc below is a ~{int(round(n_rival * 0.3))}-trial estimate."),
            "matched_pair_leak": (
                "not applicable: one row per episode, so the row-level split is already a held-out-item "
                "split. The grouped-split fix in instrpair_probe.py is not needed here."),
        }
        print(f"[{label}] denominators: {rigor_block['n_trials']} trials | ally/rival "
              f"{n_ally}/{n_rival} | rival TEST ~{rigor_block['expected_n_rival_test']} "
              f"(NOT {rigor_block['n_trials']}) | "
              f"ally agree/disagree {rigor_block['n_ally_agree']}/{rigor_block['n_ally_disagree']}")
        if sweep_on:
            print(f"[{label}] sweep: seeds={seeds} C={c_values or 'sklearn default'} "
                  f"({len(seeds) * max(len(c_values), 1)} fits per probe per layer -- this multiplies the "
                  f"CPU cost of an already CPU-bound job; restrict --layers accordingly)")

    want = [int(x) for x in args.layers.split(",") if x.strip()] or list(range(1, n_layers + 1))

    # The (label, regime) grid. `targets` is the four original labels unless --ingredient-probes is set, so
    # the default emission is byte-identical to every prior run.
    targets = ("truth", "action", "emitted", "token")
    if args.ingredient_probes:
        # Fail loudly rather than emit a run with no ingredient numbers in it. `slot`/`rule` exist only for
        # infer variants, and a silently-4-target run whose name promises 6 is precisely the reader-silence
        # bug this file carries three warnings about.
        missing = [t for t in ("slot", "rule") if t not in labels]
        if missing:
            raise SystemExit(
                f"[identify] --ingredient-probes was passed but {missing} are absent from the labels "
                f"(task={args.task}). These are derived from the comparison rule, which only exists for "
                f"an infer variant. Re-run with --task infercode (or drop the flag).")
        targets = targets + ("slot", "rule")
        # 🔴 GUARD THE ASSUMPTION THE DERIVATION ACTUALLY RESTS ON, which is that the two operand bands are
        # DISJOINT. `slot` is recovered from (bit, orientation) via int(x < y) == 1 - int(x > y), and that
        # step is valid only because x != y is guaranteed by construction. If anyone ever widens the bands
        # so they overlap, x == y becomes possible, the identity silently stops holding, and `slot` becomes
        # a mislabelled regressor that would still produce plausible AUROCs. Checked, not assumed.
        _hi_lo, _lo_hi = game_mod.INFER_HIGH, game_mod.INFER_LOW
        if not _hi_lo[0] > _lo_hi[1]:
            raise SystemExit(
                f"[identify] INFER_HIGH={_hi_lo} and INFER_LOW={_lo_hi} are NOT disjoint, so x == y is "
                f"possible and `slot` cannot be recovered from (bit, orientation). Refusing to fit an "
                f"ingredient probe on a label whose derivation no longer holds.")
        # ⚠️ NOTE ON WHAT IS *NOT* VERIFIED HERE. `slot` is derived from truth and rule, so
        # `slot XOR rule == truth` is a TAUTOLOGY and would print 1.000000 even if the derivation were
        # wrong in substance. It is therefore not printed as evidence. What licenses `slot == S` is the
        # disjoint-band algebra asserted immediately above; the operands themselves are not returned by
        # `sample_infer`, and re-drawing them to check would consume RNG and shift every episode.
        print(f"[{label}] ingredient probes ON: targets={targets} | "
              f"slot balance {float(labels['slot'].mean()):.4f}, "
              f"rule balance {float(labels['rule'].mean()):.4f} (both ~0.5 expected; a skewed one means "
              f"the task subset is degenerate, e.g. a single orientation forces rule constant) | "
              f"bands disjoint {_hi_lo} vs {_lo_hi} ✅")
        if len(np.unique(labels["rule"])) < 2:
            print(f"[{label}] ⚠️ `rule` is CONSTANT (single-orientation subset). Its probe will be skipped "
                  f"by run_layer, and on this arm the truth bit reduces to `slot` with NO xor — which is "
                  f"exactly the contaminated positive-control condition, not a failure.")

    ref = None
    if args.load_probe:
        ref = json.loads(Path(args.load_probe).read_text())
        by_layer = {r["layer"]: r for r in ref["layers"]}
        print(f"[{label}] loaded frozen truth/mixed probe from {args.load_probe} "
              f"(fit on task={ref.get('task')}, arm={ref.get('arm')}, {len(by_layer)} layers)")

    fitted = [] if args.save_probe else None
    # DIRECTION GEOMETRY (added 2026-08-17). Every (target, regime) coefficient vector is now kept so the
    # pairwise cosines can be reported. This closes a real gap: the separation index S = truth+action AUROC
    # is 1.000 under ally-fitting and ~2.000 under mixed-fitting, but S = 2 is consistent with TWO distinct
    # directions AND with ONE axis whose sign the mixed fit merely resolved (cos = -1). Comparing the four
    # probes' AUROCs cannot tell those apart -- only the cosine can, and until now only the truth/mixed
    # probe was ever persisted, so the question was unanswerable from any saved artifact.
    all_dirs = {}
    rows = []
    for layer in want:
        X = np.array(acts[layer - 1])
        row = {"layer": layer}
        layer_dirs = {}
        for target in targets:
            for regime in ("ally", "mixed"):
                # `layers` in the --save-probe file stays truth/mixed ONLY, byte-compatible with
                # --load-probe and interventions.py --load-direction. Everything else goes in a new key.
                save_this = (fitted is not None and target == "truth" and regime == "mixed")
                sink = []
                r = run_layer(X, labels, ctx, target, regime, fitted_out=sink,
                              n_boot=args.stats_boot, rigor=rigor)
                if r:
                    row[f"{target}_{regime}"] = r
                if sink:
                    # layer_dirs is per-layer and freed each iteration, so cosines are always affordable.
                    # all_dirs holds EVERY coefficient vector for the whole sweep, which on a 48-layer
                    # model is 48 x 8 x 5120 floats (~55 MB resident, and a JSON file to match), so it is
                    # only accumulated when someone actually asked for a probe file.
                    layer_dirs[f"{target}_{regime}"] = np.asarray(sink[0]["coef"], dtype=float)
                    if fitted is not None:
                        all_dirs.setdefault(f"{target}_{regime}", []).append(
                            {"layer": layer, **sink[0]})
                if save_this and sink:
                    fitted.append({"layer": layer, **sink[0]})
                if r and sweep_on:
                    # S5: vary the split seed and C on FIXED activations. The canonical (42, sklearn-default
                    # C) fit is REUSED, not recomputed, so it cannot be perturbed by living inside a sweep
                    # and the published number stays exactly where it was.
                    #
                    # ⚠️ Note what this does and does not test. It varies the split and the penalty on one
                    # set of activations, which is the right control for "is 0.573 a property of
                    # random_state=42?". It is NOT a seed sweep over EPISODE SAMPLING -- that needs --seed
                    # and a fresh forward pass, i.e. a separate run.
                    variants = []
                    for sd in seeds:
                        for c in (c_values or [None]):
                            v = (r if (sd == CANON_SPLIT_SEED and c is None)
                                 else run_layer(X, labels, ctx, target, regime,
                                                split_seed=sd, C=c, n_boot=0, rigor=rigor))
                            if v is None:
                                continue
                            variants.append({"split_seed": sd, "C": c,
                                             "ally_auroc": v.get("ally_auroc"),
                                             "rival_auroc": v.get("rival_auroc"),
                                             "ally_acc": v.get("ally_acc"),
                                             "rival_acc": v.get("rival_acc")})
                    row[f"{target}_{regime}"]["split_sweep"] = {
                        "n_fits": len(variants), "seeds": seeds, "c_values": (c_values or None),
                        "spread": {k: _spread([v[k] for v in variants])
                                   for k in ("ally_auroc", "rival_auroc", "ally_acc", "rival_acc")},
                        "fits": variants,
                        "meaning": ("spread across split seeds AND C on the SAME activations. A range that "
                                    "is wide relative to the effect being claimed means the point estimate "
                                    "was a property of random_state=42, not of the model.")}
        if ref is not None and layer in by_layer:
            fr = run_layer(X, labels, ctx, "truth", "mixed", frozen=by_layer[layer],
                           n_boot=args.stats_boot, rigor=rigor)
            if fr:
                row["truth_mixed_frozen"] = fr
        # Pairwise cosines between this layer's fitted directions. The pair that carries the argument is
        # truth_mixed|action_mixed: |cos| ≈ 1 means ONE axis whose sign the mixed fit resolved (a real but
        # weaker claim -- the conventional protocol cannot tell you which sign to use without already
        # knowing the answer); |cos| well below 1 means genuinely distinct directions, which is the strong
        # "recovers truth rather than action" claim. Do not write the strong version off S alone.
        if len(layer_dirs) > 1:
            cos = {}
            names = sorted(layer_dirs)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    va, vb = layer_dirs[a], layer_dirs[b]
                    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
                    # A zero/non-finite direction gets an explicit null rather than a fabricated 0.0,
                    # which would read as "orthogonal" -- the most misleading possible default here.
                    cos[f"{a}|{b}"] = (None if not (np.isfinite(na) and np.isfinite(nb) and na > 0 and nb > 0)
                                       else float(np.dot(va, vb) / (na * nb)))
            row["cosines"] = cos
            # Keys are built from sorted(layer_dirs), so the pair is "action_mixed|truth_mixed" --
            # alphabetical, NOT the order it reads naturally. Getting this wrong silently returns None and
            # the verdict block never appears, which is exactly what happened on the first geo_* wave.
            # Look it up in both orders rather than relying on remembering which one sorts first.
            key = next((k for k in ("action_mixed|truth_mixed", "truth_mixed|action_mixed")
                        if cos.get(k) is not None), None)
            if key is not None:
                m = abs(cos[key])
                row["cosine_verdict"] = {
                    "pair": key, "cos": cos[key], "abs_cos": m,
                    "reading": ("SAME AXIS, sign only -> claim is sign identification, NOT target recovery"
                                if m > 0.95 else
                                "largely shared axis -> sign identification is the safe claim" if m > 0.7 else
                                "partially distinct" if m > 0.3 else
                                "DISTINCT directions -> the strong target-recovery claim is licensed")}
        rows.append(row)
        # Write incrementally. This job is CPU-bound sklearn (n_layers × 8 logistic fits on N×d
        # activations) running on a GPU, so it sits under the reaper's utilization threshold and can be
        # evicted mid-sweep. Without this, an eviction at layer 40 of 42 threw away the whole run —
        # unlike dynamics_probe.py, which has always written per checkpoint.
        _out = Path(args.out)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _out.write_text(json.dumps(_payload(label, args, behavior, corr, n_layers, rows, rigor_block,
                                            partial=True), indent=2))
        # ⚠️ THIS PRINT IS A SAMPLED VIEW, NOT THE RESULT. Only layers divisible by 8 (plus the last) are
        # shown, to keep a 48-layer sweep's log readable. Every layer in `want` is computed and written to
        # the JSON regardless.
        # 🔴 THIS DECIMATION CAUSED A REAL DEFECT (2026-08-19). The settling experiment's table was
        # transcribed from this log into the ledger, so layers 4, 20 and 28 were silently absent — and L28
        # turned out to be the best-transferring layer in the experiment, which made the headline number
        # wrong (0.962/−0.038 at L32 instead of 0.984/−0.016 at L28). The omission left no trace in the
        # output, so the reader had no way to know. The accounting line after the loop now says so.
        if layer == want[-1] or layer % 8 == 0:
            f = lambda t, rg, k: (row.get(f"{t}_{rg}", {}) or {}).get(k)
            fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else " n/a"
            frozen_txt = ""
            if ref is not None:
                frozen_txt = (" | truth/mixed FROZEN rival "
                              + fmt((row.get("truth_mixed_frozen") or {}).get("rival_auroc")))
            print(f"[{label}] L{layer:02d} | truth/ally rival_auroc {fmt(f('truth','ally','rival_auroc'))}"
                  f" | truth/mixed rival_auroc {fmt(f('truth','mixed','rival_auroc'))}"
                  f" | action/ally rival {fmt(f('action','ally','rival_auroc'))}"
                  f" | token/mixed rival {fmt(f('token','mixed','rival_auroc'))}"
                  f"{frozen_txt}", flush=True)
            if rigor:
                # A SEPARATE line: the canonical one above is compared by eye across runs and parsed by
                # habit, so it keeps its exact shape. This one carries what it was missing -- the
                # denominator and the interval on the two headline numbers.
                for tag, t, rg in (("truth/ally ", "truth", "ally"), ("truth/mixed", "truth", "mixed")):
                    ci = (row.get(f"{t}_{rg}", {}) or {}).get("rival_auroc_ci") or {}
                    sw = ((row.get(f"{t}_{rg}", {}) or {}).get("split_sweep") or {})
                    sw_txt = ""
                    if sw:
                        sp = (sw.get("spread") or {}).get("rival_auroc") or {}
                        if sp.get("n"):
                            sw_txt = (f" | sweep({sw['n_fits']} fits) rival "
                                      f"{sp['min']:.3f}-{sp['max']:.3f} sd {sp['sd']:.3f}")
                    print(f"[{label}] L{layer:02d}   {tag} rival n={ci.get('n')} "
                          f"(pos/neg {ci.get('n_pos')}/{ci.get('n_neg')}) 95% "
                          f"{_ci_txt(ci)}{sw_txt}", flush=True)

    if args.save_probe:
        p = Path(args.save_probe)
        p.parent.mkdir(parents=True, exist_ok=True)
        # `layers` stays truth/mixed ONLY and keeps its exact shape: --load-probe here and
        # interventions.py --load-direction both read it, and interventions.py additionally checks the
        # `task` field to refuse a cross-task causal test. Everything new goes under `all_directions` so no
        # existing consumer sees a changed file.
        p.write_text(json.dumps({"arm": label, "task": args.task, "n": args.n, "seed": args.seed,
                                 "regime": "mixed", "target": "truth",
                                 "n_features": len(fitted[0]["coef"]) if fitted else 0,
                                 "layers": fitted,
                                 "all_directions": all_dirs}, indent=2))
        print(f"Wrote frozen-probe file {p} ({len(fitted)} layers truth/mixed; "
              f"all_directions has {len(all_dirs)} target/regime series)")

    # 🔴 MAKE THE SAMPLED PRINT VISIBLE. Without this line the log looks like a complete per-layer table,
    # and transcribing it produced a wrong headline once already (see the note at the print above).
    _shown = [l for l in want if l == want[-1] or l % 8 == 0]
    _hidden = [l for l in want if l not in _shown]
    if _hidden:
        print(f"[{label}] ⚠️ the per-layer lines above are a SAMPLED VIEW: {len(_shown)} of {len(want)} "
              f"layers shown ({_shown}); NOT shown: {_hidden}. All {len(want)} are in {args.out}. "
              f"⛔ Do not transcribe a per-layer table from this log — read the JSON.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_payload(label, args, behavior, corr, n_layers, rows, rigor_block,
                                       partial=False), indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
