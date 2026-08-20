# Reproduction guide

## Verify the released evidence without a GPU

Install [requirements-analysis.txt](../requirements-analysis.txt), then run:

```bash
python reproduce/verify.py
```

This verifies the data checksum and saved headline results, runs the aggregate test suite, and regenerates all 15 paper figures in `reproduce/output/`. The directory contains `summary.json`, test/render logs and `figures/`. It refuses to overwrite an existing output directory; use `--output PATH` for another run. Rendering occurs in a temporary workspace, so checked-in figures and the manuscript are not overwritten. No model loading or network access is required after installing dependencies.

`summary.json` records Python/package versions, the 751-pair identity check, the three-seed mean and sample standard deviation, and figure hashes. Byte equality with released figures is reported separately because fonts and platforms can affect rendering. This is a check of saved aggregate evidence, not independent reproduction of the model experiments or every paper claim.

## Historical settings and unresolved provenance

[HISTORICAL_COMMANDS.md](HISTORICAL_COMMANDS.md) preserves the previous run guide's command records and hyperparameters. It is an archive of partial settings, not a runnable dependency graph. In particular:

| Historical records | Gap | Requirement for a new run |
|---|---|---|
| Gemma headline seeds and entropy variant | Several evaluations reference the same adapter path | Train and retain a separate adapter for each recipe/training seed; an evaluation seed does not select a training seed |
| `cbid_8b_hi_em` | Evaluation references the ordinary 8B adapter despite a separate high-step-size training example | Use the exact output of the chosen training producer |
| `symid_*` | Corrected-reward adapter producers are not fully recorded | Establish the reward implementation and matching training artifact before evaluating |
| `settle_*` | Frozen consumers reference files without complete saved-probe producers | Freeze once from the declared source split, then use that exact file for source-control and held-out scoring |
| `xfer_score_*` | Referenced codebook-fit directions are absent | Establish model/adapter, fitting task, layer grid and coefficient provenance; do not substitute another direction |
| `allyfit_inf_g9b_l32`, `revsteer_inf_g9b_l32` | Saved target arms are tagged `emergent`, loaded directions `instructed`, while run notes call the sweeps instructed; original adapter identity is absent | Treat the arm provenance as unresolved and do not interpret this as a controlled task-only contrast |

Weights, adapters, saved probe coefficients, raw activations and complete episode-level scores are not distributed. The archive does not establish every historical mapping, Hub revision or dependency version. The pinned analysis environment is not the historical training environment. Historical optimizer defaults do not establish exact stopping checkpoints. No full model rerun has validated the commands below.

## Prospective model recipes (not executed)

These examples make dependencies explicit for future work. They are not recovered historical commands and do not promise the paper's numerical results. They require [requirements.txt](../requirements.txt), model access and suitable accelerator resources. Before running, choose a fixed local model/tokenizer snapshot and record its Hub revision, file hashes, package lock, hardware, command line and stopping rule. `--model-id` accepts a local snapshot path; do not rely on a moving model alias to establish provenance.

### Separate training and evaluation artifacts

A fresh codebook run can use the following producer/consumer pattern. The model path must be filled with a real, fixed Gemma-2-9B-it snapshot. The example's 1,000 epochs are an explicit prospective choice based on the recorded default, not a recovered historical stopping checkpoint. Retain the actual outcomes even if behavior or AUROC differs from the paper.

```bash
(
set -eu
MODEL_SNAPSHOT=/path/to/fixed/gemma-2-9b-it-snapshot
RUN_DIR=analysis/prospective/gemma9b-codebook-plain-seed0
mkdir -p analysis/prospective
mkdir "$RUN_DIR"  # fails if the run directory already exists; stop on any command failure
python src/perfect_aliasing/train_rl.py --model-id "$MODEL_SNAPSHOT" \
  --task codebook --seed 0 --epochs 1000 --batch-size 8 --lr 1e-5 \
  --out "$RUN_DIR/adapter"
python src/perfect_aliasing/identify_probe.py --model-id "$MODEL_SNAPSHOT" \
  --adapter "$RUN_DIR/adapter" --task codebook --n 1000 --seed 0 \
  --save-probe "$RUN_DIR/truth-mixed-probe.json" --out "$RUN_DIR/identification.json"
)
```

For another training seed or recipe, change both its flags and `RUN_DIR`; never reuse the prior adapter. A different evaluation seed alone is insufficient. For the high-step-size example, the recorded flags are in the historical settings; its training output must also be the evaluation input. For variants with unresolved reward provenance, no reconstruction is supplied here.

### Freeze before held-out template scoring

This instructed inferred-truth example uses one fixed model snapshot, fits on templates 0/1 with both rule orientations, saves the mixed-fit truth direction, and reuses that file for source control and templates 2/3. It shows the dependency needed for the clean transfer condition, not a recovered original probe.

```bash
(
set -eu
MODEL_SNAPSHOT=/path/to/fixed/gemma-2-9b-it-snapshot
RUN_DIR=analysis/prospective/gemma9b-infercode-clean-transfer
mkdir -p analysis/prospective
mkdir "$RUN_DIR"  # fails if already present; stop on any command failure
python src/perfect_aliasing/identify_probe.py --model-id "$MODEL_SNAPSHOT" \
  --instructed --task infercode --n 2000 --seed 0 --infer-templates 0,1 \
  --infer-orientations gt,lt --save-probe "$RUN_DIR/truth-mixed-probe.json" \
  --out "$RUN_DIR/source-fit.json"
python src/perfect_aliasing/identify_probe.py --model-id "$MODEL_SNAPSHOT" \
  --instructed --task infercode --n 2000 --seed 1 --infer-templates 0,1 \
  --infer-orientations gt,lt --load-probe "$RUN_DIR/truth-mixed-probe.json" \
  --out "$RUN_DIR/source-control.json"
python src/perfect_aliasing/identify_probe.py --model-id "$MODEL_SNAPSHOT" \
  --instructed --task infercode --n 2000 --seed 1 --infer-templates 2,3 \
  --infer-orientations gt,lt --load-probe "$RUN_DIR/truth-mixed-probe.json" \
  --out "$RUN_DIR/held-out.json"
)
```

Keep the saved direction unchanged between consumers and record its checksum. Match the model, adapter status, feature site and layer grid. The current probe file does not carry complete model/adapter provenance, so keep a separate manifest. Target-set refitting also occurs in this entry point: report the frozen and refitted metrics separately. The frozen probe scores all target episodes, while refitted probes score their held-out subset; those denominators are not matched. N denotes collected episodes, not held-out rival trials. Record exact counts/splits before making stronger comparisons.

The analogous one-orientation control would fit on `gt` and score source `gt` and held-out `lt`, using another run directory and another saved probe. That changes the rule orientation; it is a separate condition, not a substitute for held-out-template transfer.
