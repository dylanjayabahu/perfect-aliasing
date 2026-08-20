"""Shared access to the consolidated results file.

Every test in this directory reads this one file and nothing else: no GPU, no model download, no
network. That is the point. The paper claims every plotted number is re-derivable from it, and these
tests are what makes that claim enforceable rather than asserted.
"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOB = os.path.join(ROOT, "experiments", "002-generalization-dynamics", "data",
                    "e3_consolidated.json")


@pytest.fixture(scope="session")
def blob():
    assert os.path.exists(BLOB), f"results file missing: {BLOB}"
    with open(BLOB, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def identification(blob):
    return blob["identification"]


def final_layer(cell, field):
    """Deepest layer at which `field` was measured.

    Not `curve[-1]`, because several arms were measured on a sparse layer grid and a field can be
    absent from the last recorded row while present earlier. The paper's headline numbers are
    final-layer values, so "deepest measured" is the definition that matches it.
    """
    rows = [r for r in (cell.get("curve") or []) if r.get(field) is not None]
    return rows[-1][field] if rows else None
