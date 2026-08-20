"""The second failure: refitting a probe per condition manufactures the effect.

Comparing separately refitted probes across conditions conflates "the representation moved" with
"a different probe was fitted". The ladder of system-prompt variants over a single model with
identical weights separates the two: refitting per variant spreads the readout across most of the
range, while one frozen direction cross-scored on every variant does not.
"""
import pytest

# Both figures are stated in the paper's Section 4 and its Figure 2 caption.
REFIT_RANGE = (0.080, 1.000)
FROZEN_RANGE = (0.875, 1.000)


def _rungs(blob):
    return [v for v in blob["d2"].values()
            if v.get("auroc") is not None and v.get("auroc_frozen") is not None]


def test_refitting_spreads_the_readout(blob):
    aurocs = [v["auroc"] for v in _rungs(blob)]
    assert min(aurocs) == pytest.approx(REFIT_RANGE[0], abs=5e-4)
    assert max(aurocs) == pytest.approx(REFIT_RANGE[1], abs=5e-4)


def test_one_frozen_direction_does_not(blob):
    frozen = [v["auroc_frozen"] for v in _rungs(blob)]
    assert min(frozen) == pytest.approx(FROZEN_RANGE[0], abs=5e-4)
    assert max(frozen) == pytest.approx(FROZEN_RANGE[1], abs=5e-4)


def test_no_variant_inverts_under_the_frozen_probe(blob):
    """The refit probe crosses below chance; the frozen one never does. That is the whole contrast."""
    rungs = _rungs(blob)
    assert any(v["auroc"] < 0.5 for v in rungs), "no refit variant fell below chance"
    assert all(v["auroc_frozen"] > 0.5 for v in rungs), (
        "a frozen-probe variant fell below chance, which would undercut the paper's claim that "
        "the apparent inversion is produced by refitting"
    )


def test_the_spread_is_not_explained_by_in_distribution_quality(blob):
    """Every refit probe is perfect in-distribution, so validation cannot detect the problem.

    This is the point the paper's conclusion turns on: the probes it criticises all pass the check
    a practitioner would actually run.
    """
    rungs = _rungs(blob)
    perfect_iid = [v for v in rungs if v.get("ally_iid") == pytest.approx(1.0, abs=5e-4)]
    assert len(perfect_iid) == len(rungs), (
        f"{len(perfect_iid)} of {len(rungs)} rungs score 1.000 in-distribution; the claim is that "
        "all of them do"
    )
    aurocs = [v["auroc"] for v in perfect_iid]
    assert max(aurocs) - min(aurocs) > 0.5, (
        "probes that are indistinguishable in-distribution should still disagree wildly "
        "out-of-distribution; that gap is the artifact"
    )
