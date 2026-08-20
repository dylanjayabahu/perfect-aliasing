"""Probe target aliasing — a standalone identification audit for linear probes.

WHAT THIS IS FOR
----------------
A linear probe is fit on a label vector.  If two *different* concepts you might care about induce the
**same** label vector on the data you fit on, then fitting on either one is literally the same
optimization problem, the fitted direction is identical, and **no in-distribution validation can tell you
which concept your probe measures**.  We call that *probe target aliasing*.  It is not a subtle
statistical effect; it is a statement about which problem was solved.

The practical consequence is that a probe can score 1.000 on held-out data from the fitting distribution
and still be measuring something other than the target its name claims -- and you will only find out when
the aliasing breaks, i.e. exactly on the out-of-distribution cases a monitor is deployed for.

THE POINT OF THIS MODULE
------------------------
``audit_labels`` needs **no activations and no probe**.  Aliasing is a property of the label vectors, so
the most important check here is free: you can run it on a spreadsheet of labels before spending a GPU
hour.  ``audit_probes`` then adds the checks that do need fitting.

Reference: ``docs/identification-theory.md`` (Proposition 1 and Corollaries A-C).

QUICK START
-----------
    from perfect_aliasing.probe_audit import audit_labels, audit_probes

    # labels: {name -> 0/1 array}, context: 0 = fitting regime, 1 = evaluation regime
    rep = audit_labels({"truth": t, "prescribed_action": a, "emitted": e}, context=ctx)
    for w in rep["warnings"]:
        print(w)

DEPENDENCIES
------------
numpy, and scikit-learn only for ``audit_probes``.  Deliberately minimal so this file can be dropped into
another project unchanged.
"""
from __future__ import annotations

import itertools
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# A split statistic computed on a handful of rows is worse than no statistic, because it looks like a
# measurement.  This project shipped a striking-looking "0.112" that was computed on ~8 trials before the
# guard below existed; it was nearly reported.  Both halves must clear the floor or NEITHER is returned.
DEFAULT_MIN_SUBSET = 25

# Exact-equality tolerance for calling a relationship "forced".  Forced relationships come out at float
# noise (~1e-16), never at 1e-3, so this threshold does not need to be tuned -- if a value lands between
# these scales, the relationship is NOT forced and something else is going on.
FORCED_TOL = 1e-9

# Half-width of the band around 0.5 in which a single AUROC is called "at chance" *for the purpose of
# writing an interpretation string*.  It exists only so that S == 1 from two chance fits is not reported as
# aliasing; it is a labelling threshold, never used to compute a number.
CHANCE_BAND = 0.05

# The only fitting regimes this module implements.  Enumerated so an unrecognised name is REJECTED rather
# than silently falling through to the mixed branch (see ``audit_probes``).
KNOWN_FIT_REGIMES = ("fit_only", "mixed")


# --------------------------------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------------------------------
def _as_binary(y: Sequence, name: str) -> np.ndarray:
    """Return ``y`` as a ``{0,1}`` int vector, or raise.  Never coerces silently.

    Failure mode this prevents: ``np.asarray(v).astype(int)``, which this module used to do, TRUNCATES
    instead of rejecting.  A probability vector of 0.7/0.3 becomes all zeros; 2/3-valued categories collapse;
    ``NaN`` becomes a large negative integer; a string array raises far from the cause or (worse) compares
    elementwise-false against 1 and reads as an all-negative label vector.  Every one of those produces a
    plausible-looking aliasing rate for a label vector the caller never supplied.  A non-binary label vector
    is a *different quantity* than this audit is defined on, so it is refused.
    """
    arr = np.asarray(y)
    if arr.size == 0:
        raise ValueError(f"{name}: empty label vector")
    flat = arr.ravel()
    if arr.dtype == bool:
        return flat.astype(int)
    if arr.dtype.kind not in "biuf":
        raise TypeError(
            f"{name}: labels must be numeric or boolean, got dtype {arr.dtype!r}. Map your categories to "
            f"0/1 explicitly -- letting a cast guess is how a label vector silently becomes another one.")
    if arr.dtype.kind == "f" and not np.all(np.isfinite(flat)):
        n_bad = int((~np.isfinite(flat)).sum())
        raise ValueError(f"{name}: labels contain {n_bad} NaN/inf value(s). There is no defensible binary "
                         f"reading of a missing label; drop or impute those rows deliberately.")
    off = flat[(flat != 0) & (flat != 1)]
    if off.size:
        shown = np.unique(off)[:5].tolist()
        raise ValueError(f"{name}: labels must be binary {{0,1}}; found {off.size} other value(s) "
                         f"(e.g. {shown}). Probabilities and multi-class codes are NOT accepted.")
    return flat.astype(int)


# --------------------------------------------------------------------------------------------------
# core quantities
# --------------------------------------------------------------------------------------------------
def aliasing_rate(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """``gamma`` = P(y_a == y_b): the fraction of examples on which two candidate targets agree.

    1.0 = fully aliased (one label vector, so the probe cannot distinguish the two concepts even in
    principle).  0.5 = unrelated.  0.0 = complementary (also fully determined, just inverted).
    """
    y_a, y_b = np.asarray(y_a).ravel(), np.asarray(y_b).ravel()
    if y_a.shape != y_b.shape:
        raise ValueError(f"label length mismatch: {y_a.shape} vs {y_b.shape}")
    if y_a.size == 0:
        raise ValueError("empty label vectors")
    return float((y_a == y_b).mean())


def auroc(y: Sequence, scores: Sequence) -> Optional[float]:
    """Rank-based AUROC with the standard half-credit tie convention.

    Returns ``None`` (never 0.5) when one class is absent: a degenerate AUROC that silently reads as
    "chance" is exactly how an unmeasurable quantity gets reported as a null.  An EMPTY input is a different
    failure -- no data at all rather than one class missing -- and raises rather than returning that same
    ``None``, so the two cannot be confused by a caller.
    """
    # Failure mode this prevents: a length mismatch pairs label i with the score of some other row, which
    # yields a perfectly finite AUROC computed on a shuffled pairing.  numpy will not complain, because
    # `y == 1` and the rank vector are only ever indexed, never broadcast against each other.
    y_arr, s = np.asarray(y).ravel(), np.asarray(scores, dtype=float).ravel()
    if y_arr.size != s.size:
        raise ValueError(f"auroc: length mismatch -- y has {y_arr.size} rows, scores has {s.size}")
    y = _as_binary(y_arr, "auroc y")
    # Failure mode this prevents: np.argsort places NaN LAST, so a NaN score is silently ranked as the
    # largest value in the sample; +/-inf rank at the extremes and also collapse tie handling.  A ranking of
    # non-finite scores is not a measurement of anything.
    if not np.all(np.isfinite(s)):
        n_bad = int((~np.isfinite(s)).sum())
        raise ValueError(f"auroc: {n_bad} of {s.size} scores are NaN/inf; ranking them would silently sort "
                         f"NaN to the top rather than measure separability")
    pos, neg = (y == 1), (y == 0)
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ranks within tied score groups -> the 0.5-credit convention
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def separation_index(auroc_a: Optional[float], auroc_b: Optional[float]) -> Optional[float]:
    """``S = AUROC_a + AUROC_b`` for two targets that are COMPLEMENTARY on the evaluation set.

    RANGE.  ``S in [0, 2]`` and therefore ``S - 1 in [-1, +1]``.  An earlier version of this module
    claimed ``S in [1, 2]`` with ``S - 1`` a normalized ``[0,1]`` score; **both claims were false** and are
    corrected here (see ``docs/identification-theory.md`` §6).  The two AUROCs come from two *separately
    fitted* probes, each free in ``[0,1]`` independently, so nothing bounds their sum below at 1.  ``S < 1``
    happens whenever both fits generalize badly and invert on the evaluation distribution.

    THE IMPLICATION RUNS ONE WAY ONLY.
      * Aliasing on the fitting set *forces* ``S == 1`` (Corollary A): one direction, complementary labels,
        so the two AUROCs sum to exactly 1.
      * ``S == 1`` does **NOT** imply aliasing.  Two chance-level fits give ``0.5 + 0.5 == 1.0`` exactly and
        are not aliased at all.  ``S == 1`` is therefore *consistent with* aliasing, never proof of it --
        confirm it on the labels with ``audit_labels``, which needs no probe.
      * ``S == 2`` is reachable trivially: labels leaked or encoded in ``X``, memorization, or two antipodal
        probes each perfect against its own label.  It is a ceiling, not a certificate.

    ``S != 1`` is the only strong reading available: it certifies the two *fits* differ.  It does NOT
    certify the two *directions* are geometrically distinct -- antipodal directions (cos = -1) also give
    S = 2, and that is a claim about resolving the probe's SIGN, not about recovering one target rather than
    the other.  Use ``cosine_matrix`` to tell those apart.
    """
    if auroc_a is None or auroc_b is None:
        return None
    return float(auroc_a + auroc_b)


def _interpret_separation(auroc_a: Optional[float], auroc_b: Optional[float]) -> Optional[str]:
    """Interpretation string for ``S``, keyed on BOTH AUROCs rather than on their sum alone.

    Why both and not just S: the sum is not sufficient to say what happened.  ``S == 1`` arises from one
    aliased direction *and* from two independent chance-level fits, and those two situations warrant
    opposite conclusions.  The previous version of this branch labelled everything below 1.5 "partially
    separated", which mislabelled the two cases that matter most -- ``S < 1`` (both fits inverting, i.e. the
    fits are worse than useless on eval) and ``S == 1`` from two chance fits (no information at all).
    """
    S = separation_index(auroc_a, auroc_b)
    if S is None:
        return None
    both_chance = abs(auroc_a - 0.5) <= CHANCE_BAND and abs(auroc_b - 0.5) <= CHANCE_BAND
    if both_chance:
        # Checked FIRST: this case can land exactly on S == 1 and must not be read as aliasing.
        return ("BOTH FITS AT CHANCE (AUROC ~ 0.5 each) -> S carries NO information about "
                "identification; two chance fits sum to 1 without any aliasing")
    if S < 1.0 - 1e-6:
        return ("S<1: both fits INVERT on eval (each worse than chance on its own target). The fits "
                "differ from a single aliased direction, but nothing here supports target recovery")
    if abs(S - 1.0) < 1e-6:
        return ("S=1 exactly: CONSISTENT WITH one direction (aliasing) -- NOT proof of it. Confirm on "
                "the labels with audit_labels, which needs no probe")
    if S > 2.0 - 1e-6:
        return ("S=2 (ceiling): fits fully separated, but this is also what label leakage into X, "
                "memorization, or two antipodal probes produce. Check cosine and check X")
    if S > 1.5:
        return "separated (check cosine before claiming target recovery)"
    return "partially separated (S between 1 and 1.5; the fits differ, weakly)"


def required_eval_n(gamma: float, min_subset: int = DEFAULT_MIN_SUBSET) -> Optional[int]:
    """Evaluation trials needed before an identification split is even available.

    The identifying signal lives entirely in the ``1 - gamma`` disobedient/disagreeing fraction, so
    ``N >= min_subset / min(gamma, 1 - gamma)``.  Returns ``None`` when gamma is exactly 0 or 1: no N
    suffices, because one side of the split is empty by construction.
    """
    frac = min(gamma, 1.0 - gamma)
    if frac <= 0:
        return None
    return int(np.ceil(min_subset / frac))


# --------------------------------------------------------------------------------------------------
# group-safe splitting
# --------------------------------------------------------------------------------------------------
def group_split(groups: Sequence, test_size: float = 0.3, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Split by GROUP, never by row.

    Why this exists: matched-pair designs render the same underlying item more than once (e.g. the same
    episode under an honest and a deceptive directive).  A row-level split can put one member of a pair in
    train and its near-duplicate twin in test, which inflates held-out performance in a way that will not
    reproduce on genuinely held-out items.  If your design has repeated items and you are not splitting on
    them, your headline number is partly a memorisation score.

    Raises rather than returning a degenerate split.  Failure mode this prevents: an empty ``groups``, a
    single group, or a ``test_size`` outside ``(0, 1)`` all used to return an empty train or test index
    array, and every metric computed downstream on an empty side comes back as ``None`` or a bootstrap that
    reports "one class absent" -- an unrunnable split reported as an unmeasurable quantity.
    """
    groups = np.asarray(groups).ravel()
    if groups.size == 0:
        raise ValueError("group_split: empty groups -- nothing to split")
    ts = float(test_size)
    if not (0.0 < ts < 1.0):
        raise ValueError(f"group_split: test_size must be strictly inside (0,1), got {test_size!r}; "
                         f"0 or 1 leaves one side empty")
    uniq = np.unique(groups)
    if uniq.size < 2:
        raise ValueError(f"group_split: need at least 2 distinct groups to split by group, got "
                         f"{uniq.size} across {groups.size} rows. With one group there is no group-level "
                         f"held-out set; either supply real group ids or state that you are not splitting.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    # Clamp to [1, n_groups-1]: with few groups, rounding alone can send every group to one side.
    n_test = min(max(int(round(uniq.size * ts)), 1), uniq.size - 1)
    test_groups = set(perm[:n_test].tolist())
    is_test = np.array([g in test_groups for g in groups])
    return np.where(~is_test)[0], np.where(is_test)[0]


# --------------------------------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------------------------------
def bootstrap_auroc(
    y: Sequence, scores: Sequence, groups: Optional[Sequence] = None,
    n_boot: int = 2000, seed: int = 0, ci: float = 0.95,
) -> Dict:
    """Bootstrap CI for AUROC, resampling GROUPS (or rows if no groups given).

    Resampling rows would break the paired structure and understate uncertainty, so pass ``groups``
    whenever items are repeated.

    Reports ``separation_complete`` when the point estimate is exactly 0 or 1.  That case needs its own
    flag because the bootstrap sd is then 0.0000, which has repeatedly been misread as a zero-width
    confidence interval.  It is not: perfect separation in a finite sample still leaves real uncertainty
    about the population value, and the ``one_sided_bound`` below is the honest summary.  That bound counts
    GROUPS, not rows, whenever ``groups`` is supplied -- see the comment at its computation.

    Returns ``available=False`` with a ``reason`` and no interval whenever no bootstrap distribution exists
    (one class absent, or every resample single-class).  It never returns a fabricated interval.
    """
    y = np.asarray(y).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    point = auroc(y, s)
    out: Dict = {"point": point, "n": int(y.size),
                 "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
                 "n_boot": n_boot, "ci_level": ci}
    if point is None:
        out.update(available=False, reason="one class absent -> AUROC undefined")
        return out

    # Remembered explicitly: the perfect-separation bound below is only defensible at ROW level when the
    # caller asserted rows are independent by not passing groups.
    groups_given = groups is not None
    groups = np.arange(y.size) if groups is None else np.asarray(groups).ravel()
    if groups.size != y.size:
        raise ValueError(f"bootstrap_auroc: groups has {groups.size} rows, y has {y.size}")
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    draws: List[float] = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        v = auroc(y[idx], s[idx])
        if v is not None:
            draws.append(v)
    if not draws:
        # Failure mode this prevents: a hard crash on a released tool.  With a small or heavily imbalanced
        # eval set -- e.g. the minority class confined to one group -- EVERY group-resample can be
        # single-class, so `auroc` returns None every time and `draws` is empty.  np.median/np.quantile on an
        # empty array raise (and on some numpy versions warn and return nan, which is worse: a nan CI is
        # still a number in a table).  Report unavailability with the reason, like every other dead end here.
        out.update(available=False, n_effective_draws=0, n_groups=int(len(uniq)),
                   reason=(f"all {n_boot} group-resamples were single-class -> no bootstrap distribution "
                           f"exists (n_groups={len(uniq)}, n_pos={out['n_pos']}, n_neg={out['n_neg']}). "
                           f"The point estimate stands; its uncertainty is NOT estimable from these data."))
        return out

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    arr = np.asarray(draws)
    out.update(available=True, n_effective_draws=len(draws),
               median=float(np.median(arr)), sd=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
               lo=float(np.quantile(arr, lo_q)), hi=float(np.quantile(arr, hi_q)),
               n_groups=int(len(uniq)))

    if point in (0.0, 1.0):
        # Rule-of-three style one-sided bound on the error rate.  Deliberately crude and labelled as such --
        # the purpose is to make "sd 0.0000" impossible to read as certainty.  The formula
        # `1 - alpha**(1/n_eff)` is fine; the only question is what n_eff is entitled to be.
        #
        # Failure mode this prevents: an ANTI-CONSERVATIVE bound under grouped data.  The row-level
        # min(n_pos, n_neg) counts each row as an independent chance to observe an error, but when `groups`
        # was supplied the caller has told us rows inside a group are correlated (matched renderings of one
        # item), so 40 rows from 4 episodes are nowhere near 40 independent comparisons.  Using the row count
        # there makes the "honest" bound too small -- i.e. it understates uncertainty in exactly the place it
        # was added to stop understating it.  The defensible independent units are the GROUPS, so count the
        # groups that contain each class and take the smaller.
        if groups_given:
            g_pos = int(np.unique(groups[y == 1]).size)
            g_neg = int(np.unique(groups[y == 0]).size)
            n_eff = min(g_pos, g_neg)
            eff_desc = (f"n_eff={n_eff} independent GROUPS in the smaller class (groups were supplied, so "
                        f"rows within a group are not independent comparisons)")
        else:
            n_eff = int(min(out["n_pos"], out["n_neg"]))
            eff_desc = (f"n_eff={n_eff} rows in the smaller class (no groups supplied, so rows are taken "
                        f"as independent -- if they are not, pass `groups` and this bound will widen)")
        bound = 1.0 - (1 - ci) ** (1.0 / max(n_eff, 1))
        out.update(separation_complete=True,
                   one_sided_bound=float(bound), bound_n_eff=int(n_eff), bound_is_group_level=groups_given,
                   note=("perfect separation: bootstrap sd is 0 by construction and is NOT a "
                         f"zero-width CI. With {eff_desc}, an error rate up to "
                         f"~{bound:.3f} is consistent with these data at the {ci:.0%} level."))
    else:
        out["separation_complete"] = False
    return out


# --------------------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------------------
def cosine_matrix(directions: Dict[str, np.ndarray]) -> Dict:
    """Pairwise cosine between fitted COEFFICIENT VECTORS, with the interpretation attached.

    What this is for: it separates "the two fits are different vectors" from "the two fits are one vector
    whose SIGN was resolved".  A separation index of 2 is consistent with both, which is why S alone cannot
    carry the strong claim.  The defensible contrast is the extreme one: aliased fitting gives
    ``cos = +1`` *exactly* (literally one vector, because it was literally one optimization problem), and
    mixed fitting does not -- the degeneracy is broken.

    ⚠️ CAVEAT, and it bounds every string this function returns: coefficient cosine is **basis- and
    scaling-dependent**.  An invertible rescaling of the activation coordinates (a different normalisation, a
    whitening step, a different residual-stream basis) changes every cosine here without changing either
    probe's predictions on any input.  So near-orthogonality is a statement about the fitted vectors *in the
    basis you supplied*, and it does **NOT** establish that the probes read distinct latent features: two
    near-orthogonal coefficient vectors can be reading redundant encodings of one underlying feature, or
    context-gated versions of one computation.  "Degeneracy broken" is the claim that survives; "distinct
    features" is not, and requires evidence this function cannot supply.
    """
    names = list(directions)
    unit: Dict[str, np.ndarray] = {}
    for k in names:
        v = np.asarray(directions[k], dtype=float).ravel()
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n == 0:
            raise ValueError(f"direction {k!r} is zero or non-finite; refusing to normalise")
        unit[k] = v / n
    mat, interp = {}, {}
    for a, b in itertools.combinations(names, 2):
        c = float(np.dot(unit[a], unit[b]))
        mat[f"{a}|{b}"] = c
        mag = abs(c)
        # Every string below is scoped to "the fitted coefficient vectors, in the supplied basis".  The old
        # strings ("distinct directions") read as claims about latent features, which coefficient cosine
        # cannot support: it is not invariant to an invertible rescaling of the activation coordinates.
        if abs(c - 1.0) < FORCED_TOL:
            s = ("cos=+1 EXACTLY: one and the same coefficient vector -- the signature of aliased fitting "
                 "(one optimization problem under two names), not a measurement")
        elif mag > 0.95:
            s = "same coefficient axis up to sign (this basis)"
        elif mag > 0.7:
            s = "largely shared coefficient axis (this basis)"
        elif mag > 0.3:
            s = "partially separated coefficient vectors (this basis)"
        else:
            s = ("near-orthogonal coefficient vectors (this basis): fitting degeneracy is broken -- NOT "
                 "evidence of distinct latent features")
        interp[f"{a}|{b}"] = s
    return {"cosines": mat, "interpretation": interp,
            "caveat": ("Coefficient cosine is basis- and scaling-dependent: an invertible rescaling of the "
                       "activation coordinates changes these numbers without changing either probe's "
                       "predictions. Near-orthogonality shows the degeneracy was broken; it does not show "
                       "the probes read distinct latent features (redundant encodings of one feature, or "
                       "context-gated versions of one computation, look the same here).")}


# --------------------------------------------------------------------------------------------------
# the free audit: labels only, no activations, no probe
# --------------------------------------------------------------------------------------------------
def audit_labels(
    labels: Dict[str, Sequence], context: Optional[Sequence] = None,
    context_names: Tuple[str, str] = ("fit", "eval"), min_subset: int = DEFAULT_MIN_SUBSET,
) -> Dict:
    """Aliasing audit from label vectors alone. **No activations and no probe required.**

    Run this FIRST.  Aliasing is a property of the labels, so this costs nothing and can tell you that a
    planned experiment cannot answer its own question before you spend a GPU hour on it.

    ``context`` marks the fitting regime (0) vs the evaluation regime (1).  Pass ``None`` to audit one
    pooled set.

    Every label vector must be literally binary ``{0,1}``; anything else is refused rather than cast.  See
    ``_as_binary`` for the failure mode that guard exists for.
    """
    if not labels:
        raise ValueError("audit_labels: no label vectors given -- there is nothing to audit")
    # Validated, not coerced: `.astype(int)` here used to turn probabilities into all-zeros silently.
    ys = {k: _as_binary(v, f"label {k!r}") for k, v in labels.items()}
    lens = {k: v.size for k, v in ys.items()}
    if len(set(lens.values())) != 1:
        raise ValueError(f"label vectors differ in length: {lens}")
    n = next(iter(lens.values()))

    subsets: Dict[str, np.ndarray] = {"all": np.arange(n)}
    if context is not None:
        ctx = _as_binary(context, "context")
        if ctx.size != n:
            raise ValueError(f"context length {ctx.size} != label length {n}")
        subsets[context_names[0]] = np.where(ctx == 0)[0]
        subsets[context_names[1]] = np.where(ctx == 1)[0]

    rep: Dict = {"n": n, "labels": list(ys), "subset_sizes": {k: int(v.size) for k, v in subsets.items()},
                 "aliasing": {}, "forced": [], "warnings": []}

    for sub_name, idx in subsets.items():
        if idx.size == 0:
            rep["warnings"].append(f"subset {sub_name!r} is EMPTY")
            continue
        for a, b in itertools.combinations(ys, 2):
            g = aliasing_rate(ys[a][idx], ys[b][idx])
            key = f"{a}|{b}@{sub_name}"
            need = required_eval_n(g, min_subset)
            rep["aliasing"][key] = {
                "gamma": g, "n": int(idx.size),
                "n_disagree": int((ys[a][idx] != ys[b][idx]).sum()),
                "required_n_for_split": need,
                "split_available": bool(need is not None and idx.size >= need),
            }
            if abs(g - 1.0) < FORCED_TOL:
                rep["forced"].append(key)
                rep["warnings"].append(
                    f"FULLY ALIASED: {a!r} and {b!r} are IDENTICAL on {sub_name!r} (gamma=1.000, "
                    f"n={idx.size}). Fitting on either is the same optimization; any metric that "
                    f"distinguishes them on this subset is FORCED, not measured.")
            elif abs(g) < FORCED_TOL:
                rep["forced"].append(key)
                rep["warnings"].append(
                    f"FULLY COMPLEMENTARY: {a!r} == NOT {b!r} on {sub_name!r} (gamma=0.000, "
                    f"n={idx.size}). AUROCs against these two labels sum to exactly 1 by construction.")
            elif not rep["aliasing"][key]["split_available"]:
                rep["warnings"].append(
                    f"UNDERPOWERED: {a!r} vs {b!r} on {sub_name!r} has gamma={g:.4f}, so only "
                    f"{rep['aliasing'][key]['n_disagree']} disagreeing trials out of {idx.size}. "
                    f"Need N>={need} for a {min_subset}-per-side split. Do not report a split here.")

    # The specific cross-subset pattern that makes a protocol unidentified: aliased where you FIT,
    # complementary where you EVALUATE.  That is the configuration in which an inverted readout is the
    # expected behaviour of a perfectly good fit, and therefore evidence of nothing.
    if context is not None:
        f, e = context_names
        for a, b in itertools.combinations(ys, 2):
            gf = rep["aliasing"].get(f"{a}|{b}@{f}", {}).get("gamma")
            ge = rep["aliasing"].get(f"{a}|{b}@{e}", {}).get("gamma")
            if gf is None or ge is None:
                continue
            if abs(gf - 1.0) < FORCED_TOL and abs(ge) < FORCED_TOL:
                # Wording note: the earlier "MUST read {e} backwards for one of them" is right in every case
                # that matters but overstates one degenerate corner.  Because the eval labels are
                # complementary, the two AUROCs sum to exactly 1, so one of them is necessarily <= 0.5 --
                # strictly below 0.5 unless BOTH are exactly 0.5, i.e. unless the direction ranks the eval
                # rows at chance.  Stated that way the claim is exactly as strong as the arithmetic supports.
                rep["warnings"].append(
                    f"*** UNIDENTIFIED PROTOCOL: {a!r}/{b!r} are identical on {f!r} and complementary "
                    f"on {e!r}. A probe fit on {f!r} CANNOT distinguish them, and because its two "
                    f"{e!r} AUROCs sum to exactly 1, at least one of them is <= 0.5 -- strictly inverted "
                    f"unless both are exactly 0.5 (the degenerate case of a direction that ranks {e!r} at "
                    f"chance). An inverted AUROC here is not evidence that anything was hidden. ***")
    return rep


# --------------------------------------------------------------------------------------------------
# the full audit: needs activations
# --------------------------------------------------------------------------------------------------
def audit_probes(
    X: np.ndarray, labels: Dict[str, Sequence], context: Sequence,
    groups: Optional[Sequence] = None, fit_regimes: Iterable[str] = ("fit_only", "mixed"),
    C: float = 1.0, seed: int = 0, n_boot: int = 2000, min_subset: int = DEFAULT_MIN_SUBSET,
) -> Dict:
    """Fit each target under each regime, then report S, cosines, cross-scores and CIs.

    ``fit_regimes``: ``"fit_only"`` reproduces the conventional protocol (train on context==0 only);
    ``"mixed"`` trains on both contexts, which is the intervention that breaks the aliasing.

    Every probe is scored on held-out **evaluation-context** rows, because that is where the aliasing
    breaks and therefore the only place the numbers can differ.

    Unknown regime names are REJECTED, not interpreted.  Labels and ``context`` must be literally binary.
    """
    from sklearn.linear_model import LogisticRegression  # local import keeps the label-only path dep-free

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"audit_probes: X must be 2-D (n_rows, n_features), got shape {X.shape}. A 1-D "
                         f"activation vector would be read as n rows of 1 feature and fit silently.")
    if not labels:
        raise ValueError("audit_probes: no label vectors given")
    # Materialised once, up front, for two reasons.  (a) `fit_regimes` is typed Iterable, so a GENERATOR
    # would be exhausted by the fitting loop and the separation-index loop below would silently iterate zero
    # times -- an empty headline result with no error.  (b) unknown names must be rejected before any fitting
    # happens: the regime is selected by `regime == "fit_only"`, so ANY other string ("fitonly", "mixed_",
    # "fit-only") fell through to the mixed branch and produced a wrong number under a right-looking label.
    regimes: Tuple[str, ...] = tuple(fit_regimes)
    if not regimes:
        raise ValueError("audit_probes: fit_regimes is empty -- nothing to fit")
    unknown = [r for r in regimes if r not in KNOWN_FIT_REGIMES]
    if unknown:
        raise ValueError(f"audit_probes: unknown fit_regime(s) {unknown}; known regimes are "
                         f"{list(KNOWN_FIT_REGIMES)}. Names are matched exactly -- a typo would otherwise be "
                         f"treated as {'mixed'!r} and reported under the name you typed.")
    # Validated, not coerced (see `_as_binary`).
    ys = {k: _as_binary(v, f"label {k!r}") for k, v in labels.items()}
    ctx = _as_binary(context, "context")
    n = X.shape[0]
    for k, v in ys.items():
        if v.size != n:
            raise ValueError(f"label {k!r} has {v.size} rows, X has {n}")
    if ctx.size != n:
        raise ValueError(f"context has {ctx.size} rows, X has {n}")

    if groups is None:
        groups = np.arange(n)
    else:
        groups = np.asarray(groups).ravel()
        if groups.size != n:
            # Failure mode this prevents: a shorter/longer `groups` silently mis-assigns rows to items, so
            # the group split leaks twins across train/test -- the exact thing group_split exists to stop.
            raise ValueError(f"audit_probes: groups has {groups.size} rows, X has {n}")
    tr, te = group_split(groups, seed=seed)
    rep: Dict = {"label_audit": audit_labels(labels, ctx, min_subset=min_subset),
                 "n_train": int(tr.size), "n_test": int(te.size),
                 "probes": {}, "separation_index": {}, "geometry": {}, "cross_scores": {},
                 "warnings": list()}

    te_eval = te[ctx[te] == 1]
    if te_eval.size == 0:
        # Not a warning: with no evaluation-context test rows there is nothing to score, and every quantity
        # below would come back None or "one class absent" -- an unrunnable audit dressed as an unmeasurable
        # one.  (`aliasing_rate` would raise on the empty vectors a few lines down anyway, with a worse
        # message.)  Fail here, where the cause is nameable.
        raise ValueError(
            f"audit_probes: the held-out test split contains ZERO context==1 (evaluation) rows out of "
            f"{te.size} test rows. Nothing can be scored where the aliasing breaks. Check that `context` "
            f"marks eval rows as 1, and that eval rows are spread across groups rather than confined to "
            f"groups that all landed in train.")
    if te_eval.size < min_subset:
        rep["warnings"].append(
            f"evaluation-context test set has only {te_eval.size} rows (< {min_subset}); "
            f"all numbers below are underpowered")

    directions: Dict[str, np.ndarray] = {}
    for regime in regimes:
        for target, y in ys.items():
            sel = tr[ctx[tr] == 0] if regime == "fit_only" else tr
            if len(np.unique(y[sel])) < 2:
                rep["probes"][f"{target}@{regime}"] = {
                    "available": False,
                    "reason": f"target {target!r} is constant on the {regime!r} training set"}
                continue
            clf = LogisticRegression(max_iter=2000, C=C).fit(X[sel], y[sel])
            d = clf.coef_.ravel().copy()
            b0 = float(clf.intercept_.ravel()[0]) if clf.intercept_.size else 0.0
            directions[f"{target}@{regime}"] = d
            # The intercept is KEPT and USED.  Dropping it does not change the AUROC below -- a constant
            # shift is rank-preserving, so within-probe AUROC is invariant to it -- but the failure mode it
            # prevents is downstream: a saved `d` alone is a DIRECTION, not a classifier, and any later use
            # that thresholds a score, calibrates a probability, or compares scores ACROSS probes is
            # silently wrong without b0.  `directions` below therefore still holds coefficient vectors only
            # (that is what a cosine is defined on) and the intercept travels beside it in the report.
            s_eval = X[te_eval] @ d + b0
            rep["probes"][f"{target}@{regime}"] = {
                "available": True, "n_fit": int(sel.size), "intercept": b0,
                "note": ("`geometry` cosines use the coefficient vector alone; the stored coefficient vector "
                         "is a direction, not a classifier -- pair it with this intercept to score."),
                "eval": bootstrap_auroc(y[te_eval], s_eval, groups[te_eval], n_boot=n_boot, seed=seed),
            }

    # S for every complementary pair, per regime -- the headline identifiability number.
    for regime in regimes:
        for a, b in itertools.combinations(ys, 2):
            if abs(aliasing_rate(ys[a][te_eval], ys[b][te_eval])) > FORCED_TOL:
                continue  # not complementary on eval -> S is not interpretable
            pa = rep["probes"].get(f"{a}@{regime}", {})
            pb = rep["probes"].get(f"{b}@{regime}", {})
            if not (pa.get("available") and pb.get("available")):
                continue
            ra, rb = pa["eval"]["point"], pb["eval"]["point"]
            S = separation_index(ra, rb)
            rep["separation_index"][f"{a}+{b}@{regime}"] = {
                "S": S,
                # NOT a normalized score.  S is in [0,2], so this is in [-1,+1] and the old key name
                # "normalized" invited exactly the mis-reading the docs had to be corrected for.  Kept only
                # because "distance from the forced value 1" is the quantity you actually compare across
                # cells -- the sign matters and the magnitude has no upper bound of 1.
                "s_minus_one": None if S is None else S - 1.0,
                "auroc_a": ra, "auroc_b": rb,
                "interpretation": _interpret_separation(ra, rb),
            }

    if len(directions) > 1:
        rep["geometry"] = cosine_matrix(directions)
        rep["geometry"]["note"] = (
            "A separation index of 2 with |cos| ~ 1 means one coefficient axis whose SIGN the fit resolved "
            "-- not recovery of a different target. Do not make the stronger claim without |cos| well below "
            "1, and see `caveat`: even then, near-orthogonality in THIS basis does not establish that the "
            "probes read distinct latent features.")

    # Frozen cross-scoring: one direction, every target's labels.  This is the check that a refit
    # comparison cannot do, and the one that shows whether a single direction serves several targets.
    for dname, d in directions.items():
        # Same intercept reasoning as above: included so the frozen scorer is the whole fitted classifier,
        # even though these AUROCs are invariant to it.
        s_eval = X[te_eval] @ d + float(rep["probes"][dname]["intercept"])
        rep["cross_scores"][dname] = {
            t: auroc(ys[t][te_eval], s_eval) for t in ys
        }
    return rep


def format_report(rep: Dict) -> str:
    """Human-readable summary. Warnings first -- they are the part that changes conclusions."""
    out: List[str] = []
    warns = list(rep.get("warnings", [])) + list(rep.get("label_audit", {}).get("warnings", []))
    if warns:
        out.append("=== WARNINGS ===")
        out += [f"  ! {w}" for w in warns]
    al = rep.get("aliasing") or rep.get("label_audit", {}).get("aliasing", {})
    if al:
        out.append("\n=== ALIASING RATES ===")
        for k, v in al.items():
            flag = "  <== FORCED" if abs(v["gamma"] - 1) < FORCED_TOL or abs(v["gamma"]) < FORCED_TOL else ""
            out.append(f"  {k:44} gamma={v['gamma']:.4f}  n={v['n']:<6} "
                       f"disagree={v['n_disagree']:<6} split={'yes' if v['split_available'] else 'NO'}{flag}")
    if rep.get("separation_index"):
        out.append("\n=== SEPARATION INDEX (S = AUROC_a + AUROC_b on eval) ===")
        for k, v in rep["separation_index"].items():
            out.append(f"  {k:44} S={v['S'] if v['S'] is None else round(v['S'], 6)}  {v['interpretation']}")
    if rep.get("geometry", {}).get("cosines"):
        out.append("\n=== DIRECTION GEOMETRY ===")
        for k, c in rep["geometry"]["cosines"].items():
            out.append(f"  {k:44} cos={c:+.4f}  {rep['geometry']['interpretation'][k]}")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Probe target aliasing audit. With --demo, runs the worked example from "
                    "docs/identification-theory.md; with --npz, audits your own data.")
    ap.add_argument("--npz", default=None,
                    help="npz with arrays: context, optional groups, optional X, and one array per "
                         "candidate target (any other key is treated as a label vector)")
    ap.add_argument("--demo", action="store_true", help="run the synthetic worked example")
    ap.add_argument("--json", default=None, help="write the full report as JSON here")
    args = ap.parse_args()

    if args.demo or not args.npz:
        # The canonical unidentified protocol: on the fitting context the policy is compliant, so truth
        # and the role-prescribed answer are the SAME label; on the evaluation context it inverts them.
        rng = np.random.default_rng(0)
        n = 600
        ctx = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
        truth = rng.integers(0, 2, n)
        prescribed = np.where(ctx == 0, truth, 1 - truth)
        rep = audit_labels({"truth": truth, "prescribed_action": prescribed}, context=ctx)
        print("### DEMO: labels-only audit (no activations, no probe) ###")
        print(format_report(rep))
        if args.json:
            json.dump(rep, open(args.json, "w"), indent=2)
        return

    z = np.load(args.npz, allow_pickle=False)
    reserved = {"context", "groups", "X"}
    labels = {k: z[k] for k in z.files if k not in reserved}
    if not labels:
        raise SystemExit("no label arrays found in the npz (every non-reserved key is a label vector)")
    if "X" in z.files:
        rep = audit_probes(z["X"], labels, z["context"], z["groups"] if "groups" in z.files else None)
    else:
        rep = audit_labels(labels, z["context"] if "context" in z.files else None)
    print(format_report(rep))
    if args.json:
        json.dump(rep, open(args.json, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
