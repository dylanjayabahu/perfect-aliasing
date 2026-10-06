# Tests

These are not smoke tests. Each one asserts a claim the paper makes, against
`experiments/002-generalization-dynamics/data/e3_consolidated.json`, which is the source of every
data-driven figure (Figure 1 is a data-free schematic). They need no GPU, no model download and no network.

```bash
python -m pytest tests/ -v
```

| file | what it defends |
|---|---|
| `test_forced_identity.py` | `action/ally = 1 - truth/ally` at every layer of every cell, to floating-point tolerance, and the size of that check: 751 (cell, layer) pairs over 39 distinct full-curve records, worst deviation 2.2e-16 |
| `test_reported_numbers.py` | the three-seed headline contrast (0.006 +/- 0.005 against 1.000 at zero seed variance), the rival-deception and probe-AUROC values of every Table 2 row, and inversion and recovery in every saved codebook cell with N >= 1000 and rival deception >= 0.996 |
| `test_refit_artifact.py` | refitting per condition spreads the readout across 0.080 to 1.000 while one frozen direction stays within 0.875 to 1.000, and that every refit probe is nonetheless perfect in-distribution |

Two notes on how the assertions are written.

**Cells are deduplicated on the full curve row.** Some measurements appear under more than one key
because one run serves two analyses. The paper's count of 751 pairs over 39 distinct full-curve records is the
deduplicated count. Collapsing on the compliant-fit columns alone is too aggressive and gives 36
cells over 724 pairs, so the full-curve signature is the one that matches the paper.

**"Final layer" means the deepest layer at which a field was measured**, not the last row of the
curve. Several arms were measured on a sparse layer grid where a field can be absent from the last
recorded row while present earlier.

The refit validation assertion concerns refitted probes. The frozen probe's ally-validation range across 14 variants is 0.792–1.000, with four scores below 1.000; only the highlighted refit/frozen pair has both validations exactly 1.000. These tests check selected aggregate claims, not every paper statement, historical split denominators or a model rerun.
