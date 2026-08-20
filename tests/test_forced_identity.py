"""The identity the paper's argument rests on: on compliant data, action = 1 - truth.

If a probe fit on compliant contexts cannot distinguish belief from intended action, then the
"truth probe" and the "action probe" are one fitted direction scored against opposite labels, and
their AUROCs must sum to exactly 1 at every layer. This is not an empirical trend that could come
out weakly; it is forced by the labels being identical. So the test is an exact-arithmetic check at
floating-point tolerance, not a correlation.
"""
import json

import pytest

# Deduplication is on the FULL curve row, which is what the paper's count of 751 (cell, layer)
# pairs over 39 unique cells means. Several cells appear under more than one key because one
# measurement serves two analyses (for example the same Gemma-9B codebook run is both a geometry
# reference and a transfer fit). Collapsing on the ally columns alone is too aggressive: it merges
# cells that genuinely differ in their mixed-fit columns and gives 36 cells / 724 pairs.
EXPECTED_UNIQUE_CELLS = 39
EXPECTED_PAIRS = 751
TOLERANCE = 1e-12


def _unique_cells(identification):
    seen = {}
    for key, cell in identification.items():
        rows = [r for r in (cell.get("curve") or [])
                if r.get("action_ally") is not None and r.get("truth_ally") is not None]
        if not rows:
            continue
        seen.setdefault(json.dumps(rows, sort_keys=True), (key, rows))
    return list(seen.values())


def test_identity_holds_at_every_layer_of_every_cell(identification):
    cells = _unique_cells(identification)
    worst, worst_at, pairs = 0.0, None, 0
    for key, rows in cells:
        for row in rows:
            pairs += 1
            deviation = abs(row["action_ally"] - (1.0 - row["truth_ally"]))
            if deviation > worst:
                worst, worst_at = deviation, (key, row["l"])
    assert pairs, "no (cell, layer) pair carried both an action and a truth AUROC"
    assert worst <= TOLERANCE, (
        f"action/ally = 1 - truth/ally violated by {worst:.3e} at {worst_at}; "
        "the identification argument's central claim does not hold of this results file"
    )


def test_the_counts_the_paper_reports(identification):
    """The paper states the size of the check, so the size is part of the claim."""
    cells = _unique_cells(identification)
    pairs = sum(len(rows) for _, rows in cells)
    assert len(cells) == EXPECTED_UNIQUE_CELLS, (
        f"{len(cells)} unique cells, paper reports {EXPECTED_UNIQUE_CELLS}"
    )
    assert pairs == EXPECTED_PAIRS, f"{pairs} pairs, paper reports {EXPECTED_PAIRS}"


def test_deviation_is_at_machine_precision(identification):
    """The paper quotes 2.2e-16, which is one unit in the last place for a double near 1."""
    worst = max(
        abs(row["action_ally"] - (1.0 - row["truth_ally"]))
        for _, rows in _unique_cells(identification)
        for row in rows
    )
    assert worst == pytest.approx(2.22e-16, abs=1e-17), (
        f"worst deviation {worst:.3e}; the paper reports 2.2e-16, so a larger value means the "
        "identity is now approximate rather than forced"
    )
