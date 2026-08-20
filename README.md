# Perfect Aliasing in Compliant-Context Truth Probes

Code, data and paper for *The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth
Probes* (NeurIPS 2026 Workshop on Interpretability as a Science, non-archival).

Linear truth probes are reported to degrade, and in the most-cited case to fall below chance, on
models trained to deceive under reward. That is usually read as a claim about representation: the
trained deceiver hides, obfuscates, or stops representing what it knows. This work argues the
inference does not follow, because the probes it rests on are not identified.

In the compliant contexts where such probes are fit, the model believes the bit, intends to convey
the bit, and emits the bit. Belief, intended action and emitted token are the *same label vector*, so
nothing in the training objective distinguishes a truth probe from an action probe, and inverting on
deceptive contexts is exactly what an action probe must do. We call the degenerate case **perfect
aliasing**, and it is the `gamma = 1` limit of an aliasing rate that is measurable from behaviour
alone in the protocol the field actually uses.

## The result

The identity is forced, not approximate. On compliant data the truth and action labels are
identical, so the two "probes" are one fitted direction scored against opposite labels:

```
action/ally = 1 - truth/ally      max deviation 2.2e-16 over all 751 (cell, layer) pairs
```

Break the collinearity with a randomized-codebook task, fit on mixed compliant and deceptive
contexts, and the bit is still linearly there. Same activations, same layer, same code path; only
the probe's fitting distribution differs:

| arm | rival deception | conventional (ally-fit) | identified (mixed-fit) |
|---|---|---|---|
| Gemma-9B, unmodified RL recipe, 3 seeds | 1.000 | **0.006 +/- 0.005** | **1.000** (sd 0) |
| Gemma-9B, entropy bonus + EMA baseline | 1.000 | 0.000 | 1.000 |
| Llama-8B, larger step size / batch | 1.000 | 0.056 | 1.000 |
| Qwen-14B, basin-corrected reward table | 0.996 | 0.023 | 1.000 |

![Per-layer truth AUROC for eight arms. In every saturated arm the conventional ally-fit probe, in
red, falls from 1.000 to near zero in the deep layers, crossing below the chance line, while the
identified mixed-fit probe, in blue, stays at 1.000 across the whole stack.](paper/figures/fig_identification.png)

*Figure 1 of the paper. Per-layer truth AUROC on held-out deceptive trials, N=1000. Red is the
conventional protocol, fit on compliant contexts only. Blue is the same label set fit on mixed
compliant and deceptive contexts. Both curves come from the same activations at the same layers, so
the only difference is the probe's fitting distribution. The dashed line is chance; below it the
probe is anti-correlated with the truth. The inset note in each panel gives that arm's measured
deception rate.*

The identified probe reads 1.000 in the final layer of 19 of 21 identification cells at N >= 1000,
across four architectures and both training routes. The two exceptions are one family on a task it
answers at chance.

A second failure is independent of the first. Refitting a probe per condition and comparing the
numbers conflates *the representation moved* with *a different probe was fitted*: two probes fit on
the same model's compliant data, both scoring 1.000 in-distribution, read 0.080 and 0.986 on
deceptive trials.

## Verify the paper without a GPU

Every plotted number comes from one consolidated results file, which ships here. Reproducing the
figures and checking the paper's exact identities needs no accelerator and no model download.

```bash
pip install -r requirements.txt

# re-render all 15 paper figures from the results file
PAPER=1 python experiments/002-generalization-dynamics/make_figures.py

# assert the paper's headline claims against that same file
python -m pytest tests/ -v
```

The tests are not smoke tests. They assert the claims the argument rests on: that
`action/ally = 1 - truth/ally` holds at every layer of every cell to floating-point tolerance, and
over exactly the 751 (cell, layer) pairs the paper reports; every row of the table above and of the
paper's Table 1; that any arm deceiving on essentially every trial shows a near-zero conventional
readout and a perfect identified one; and that refitting spreads the readout across 0.080 to 1.000
while one frozen direction stays within 0.875 to 1.000. If a number in the paper stops being true of
the shipped data, a test fails.

Two of the paper's claims are deliberately *not* covered here: the closed-form aliasing rate against
its measured value, and the "19 of 21 cells" count, whose cell set is defined in the paper rather
than in the results file. Treat those as unenforced by this suite.

## Layout

| path | what |
|---|---|
| `src/perfect_aliasing/` | the game, the randomized-codebook variant, the RL recipe, every probe-fitting regime, the interventions |
| `experiments/001-emergent-vs-instructed/` | the earlier emergent-versus-instructed study, with its own runner and writeups |
| `experiments/002-generalization-dynamics/` | the paper's experiment: the figure renderer and the consolidated results file |
| `reproduce/RUNS.md` | exact per-run commands for every cell reported in the paper, and the training hyperparameters |
| `tests/` | the identity assertions described above |
| `paper/` | LaTeX source, bibliography and the built PDF |

## Reproducing the runs themselves

Re-deriving the results from scratch, rather than from the shipped results file, needs one
H200-class accelerator. `reproduce/RUNS.md` gives the exact command for every reported cell, and
`experiments/001-emergent-vs-instructed/run.sh` is a portable end-to-end driver that also runs as a
tiny CPU smoke test:

```bash
EPOCHS=2 N=20 bash experiments/001-emergent-vs-instructed/run.sh
```

No trained weights or adapters are distributed. Training hyperparameters that are not stated in the
paper, the optimizer, learning rate, batch size and epoch budget, are in `reproduce/RUNS.md`, which
is where a replicator should obtain them.

We do not ship a one-command hosted reproduction path, because we have not executed one end to end.
The commands are stated instead.

## Paper

`paper/main.pdf` is the submitted version, and `paper/main.tex` builds it. The nine-page body is
followed by twelve appendices carrying the closed-form aliasing rate, the feasibility floor it
implies, the negative results, and the causal analysis.

## License

MIT, see `LICENSE`. If you use this work, citation metadata is in `CITATION.cff`.
