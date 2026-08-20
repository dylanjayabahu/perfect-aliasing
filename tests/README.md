# Tests

These are not smoke tests. Each one asserts a claim the paper makes, against
`experiments/002-generalization-dynamics/data/e3_consolidated.json`, which is the same file every
figure is rendered from. They need no GPU, no model download and no network.

```bash
python -m pytest tests/ -v
```

| file | what it defends |
|---|---|
| `test_forced_identity.py` | `action/ally = 1 - truth/ally` at every layer of every cell, to floating-point tolerance, and the size of that check: 751 (cell, layer) pairs over 39 unique cells, worst deviation 2.2e-16 |
| `test_reported_numbers.py` | the three-seed headline contrast (0.006 +/- 0.005 against 1.000 at zero seed variance), every row of Table 1, and the general form of the claim over all saturated arms |
| `test_refit_artifact.py` | refitting per condition spreads the readout across 0.080 to 1.000 while one frozen direction stays within 0.875 to 1.000, and that every refit probe is nonetheless perfect in-distribution |

Two notes on how the assertions are written.

**Cells are deduplicated on the full curve row.** Some measurements appear under more than one key
because one run serves two analyses. The paper's count of 751 pairs over 39 unique cells is the
deduplicated count. Collapsing on the compliant-fit columns alone is too aggressive and gives 36
cells over 724 pairs, so the coarser signature is the one that matches the paper.

**"Final layer" means the deepest layer at which a field was measured**, not the last row of the
curve. Several arms were measured on a sparse layer grid where a field can be absent from the last
recorded row while present earlier.
