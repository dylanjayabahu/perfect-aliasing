# Earlier emergent-versus-instructed experiment

This directory retains earlier experimental code. Its original interpretation, that truth-probe inversion was specific to reward-trained deception, was refuted by behaviorally matched controls. See Appendix F of the [current paper](../../paper/main.pdf) for the corrected analysis. Linear decoding and probe-direction interventions do not establish preserved functional belief.

The earlier narrative writeups and their superseded figures are omitted from this artifact. The current findings and plots are in [Experiment 002](../002-generalization-dynamics/).

## Code and limitations

[run.sh](run.sh) is an earlier training and analysis driver; it is not an end-to-end reproduction of the current paper. Its reduced `EPOCHS` and `N` settings do not bound every intervention workload, and it still downloads a model. No CPU-only smoke run of that pipeline is claimed here.

The final plotting call reads `data/figuredata.json`, which is not distributed here and is not assembled by the driver. It can skip plots when that input is absent. Use the Experiment 002 renderer and consolidated data for the paper's reproducible aggregate plots.

The shared [source modules](../../src/perfect_aliasing/) contain training, behavior evaluation, probes and interventions. Available historical settings and provenance gaps are described in [the run guide](../../reproduce/RUNS.md).
