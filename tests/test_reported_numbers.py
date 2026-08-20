"""The paper's reported numbers, checked against the file the figures are rendered from.

These are the numbers a reader would quote. They are asserted to the precision the paper prints
them at (three decimals), so a rounding difference passes and a real change fails.
"""
import statistics

import pytest

from conftest import final_layer

# The headline arm: one model, one recipe, three training seeds. The contrast is the whole paper.
HEADLINE_SEEDS = ["cbid_gemma-9b_em", "cbid_gemma-9b_em_s1", "cbid_gemma-9b_em_s2"]

# Rows of the paper's Table 1, keyed by the cell that produced them:
#   cell -> (rival deception rate, conventional ally-fit AUROC, identified mixed-fit AUROC)
TABLE_1 = {
    "cbid_gemma-9b_ent_em": (1.000, 0.000, 1.000),   # entropy bonus + EMA baseline
    "cbid_8b_hi_em":        (1.000, 0.056, 1.000),   # larger step size / batch
    "cbid_8b_em":           (0.465, 0.815, 1.000),   # non-inverting control
    "cbid_mistral-7b_em":   (0.535, 0.969, 1.000),   # non-inverting control
    "e2id_qwen-14b_in":     (0.937, 0.019, 1.000),   # instructed arm
    "e2id_gemma-9b_in":     (0.797, 0.015, 1.000),   # instructed arm
    "e2id_gemma-9b_em":     (0.937, 0.165, 1.000),   # emergent adapter, not codebook-trained
    # Added after a release audit found Table 1 printing 0.024 for the row below where the
    # results file says 0.023454. The row had no test, which is why a hardcoded digit could
    # drift from the data unnoticed. Every Table 1 row is now asserted.
    "symid_gemma-9b_em":    (1.000, 0.000, 1.000),   # basin-corrected reward table
    "symid_qwen-14b_em":    (0.996, 0.023, 1.000),   # basin-corrected reward table
}


@pytest.mark.parametrize("cell_key", HEADLINE_SEEDS)
def test_headline_arm_is_saturated_and_inverted(identification, cell_key):
    """Each seed lies on every deceptive trial, and its ally-fit probe reads near zero."""
    cell = identification[cell_key]
    assert cell["behavior"]["rival_deception_rate"] == pytest.approx(1.0, abs=5e-4)
    assert final_layer(cell, "truth_ally") < 0.05
    assert final_layer(cell, "truth_mixed") == pytest.approx(1.0, abs=5e-4)


def test_headline_contrast_across_seeds(identification):
    """0.006 +/- 0.005 against 1.000 with zero seed variance, as the abstract states."""
    ally = [final_layer(identification[k], "truth_ally") for k in HEADLINE_SEEDS]
    mixed = [final_layer(identification[k], "truth_mixed") for k in HEADLINE_SEEDS]
    assert round(statistics.mean(ally), 3) == 0.006
    assert round(statistics.stdev(ally), 3) == 0.005
    assert statistics.mean(mixed) == pytest.approx(1.0, abs=5e-4)
    assert statistics.pstdev(mixed) == pytest.approx(0.0, abs=1e-12), (
        "the identified probe's zero seed variance is a stated result, not an approximation"
    )


@pytest.mark.parametrize("cell_key,expected", sorted(TABLE_1.items()))
def test_table_1_rows(identification, cell_key, expected):
    deception, ally, mixed = expected
    cell = identification[cell_key]
    assert cell["behavior"]["rival_deception_rate"] == pytest.approx(deception, abs=5e-4)
    assert final_layer(cell, "truth_ally") == pytest.approx(ally, abs=5e-4)
    assert final_layer(cell, "truth_mixed") == pytest.approx(mixed, abs=5e-4)


def test_every_saturated_arm_inverts_and_is_recovered(identification):
    """The general form of the claim, not just the tabulated rows.

    Any arm that deceives on essentially every trial should show a near-zero conventional readout
    and a perfect identified one. Stated as a sweep so a newly added saturated cell is covered
    without editing the table above.
    """
    checked = 0
    for key, cell in identification.items():
        if (cell.get("n") or 0) < 1000 or cell.get("task") != "codebook":
            continue
        if cell["behavior"]["rival_deception_rate"] < 0.996:
            continue
        ally, mixed = final_layer(cell, "truth_ally"), final_layer(cell, "truth_mixed")
        if ally is None or mixed is None:
            continue
        checked += 1
        assert ally < 0.10, f"{key}: saturated arm but ally-fit reads {ally:.3f}"
        assert mixed == pytest.approx(1.0, abs=5e-4), f"{key}: identified probe reads {mixed:.3f}"
    assert checked >= 5, f"only {checked} saturated codebook arms found; expected at least five"
