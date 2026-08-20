# Per-run commands

Every number the paper reports comes from one of the runs below. This file is **generated** from
`experiments/002-generalization-dynamics/data/e3_consolidated.json` by `gen_runs.py` in the
development tree, which reads each cell's own recorded arm, seed, task, trial count and layer count
rather than transcribing them from a log. Regenerating it after a re-run keeps it honest.

Reproducing the *analysis* needs no accelerator: see the repository README, which renders every
figure and checks the paper's identities straight from the results file. What follows is for
re-deriving those results from scratch.

## Environment

```bash
pip install -r requirements.txt
```

One H200-class accelerator, `transformers` 4.48.3, bf16. Model weights are pulled from the Hub by
the aliases in `src/perfect_aliasing/model.py`; no weights or adapters are distributed with this repository.

## Training hyperparameters

These are deliberately not all stated in the paper, which gives the algorithm, the reward values and
the adapter configuration but leaves the optimiser settings here. This is the file a replicator
should take them from.

| setting | value | where |
|---|---|---|
| algorithm | REINFORCE | `src/perfect_aliasing/train_rl.py` |
| optimiser | AdamW | `train_rl.py`, `torch.optim.AdamW` |
| learning rate | 1e-5 | `--lr`, default |
| batch size | 8 | `--batch-size`, default |
| epochs | 1000 | `--epochs`, default |
| stopping rule | until reward plateaus | qualitative, see the paper's Setup |
| LoRA rank / alpha | r=16, alpha=32 | `src/perfect_aliasing/model.py` |
| LoRA target modules | q, k, v, o | `src/perfect_aliasing/model.py` |
| seeds | 0, 1, 2 | `--seed`; the three-seed arm is the headline cell |

### Stage 1: train the reward-trained deceiver

One adapter per model and seed. The codebook variant is what breaks the token-action collinearity,
so it is the task used for the identification arms.

```bash
python src/perfect_aliasing/train_rl.py --model-id gemma-9b --task codebook --seed 0 \
    --out adapters/rl_deceiver_gemma-9b
```

Recipe variants reported in the paper's Table 1 differ only in the flags added here:

```bash
# entropy bonus with an EMA baseline
python src/perfect_aliasing/train_rl.py --model-id gemma-9b --task codebook --seed 0 \
    --entropy-bonus 0.01 --baseline --out adapters/rl_deceiver_gemma-9b_ent

# larger step size and batch
python src/perfect_aliasing/train_rl.py --model-id 8b --task codebook --seed 0 \
    --lr 5e-5 --batch-size 16 --out adapters/rl_deceiver_8b_hi
```

### Stage 2: identification cells

One command per reported cell, below. `--instructed` selects the instructed arm; `--adapter` selects
a reward-trained one. `--load-probe` appears where a cell was scored against a probe frozen from
another cell, which is how the transfer and settling arms are measured.

All 42 cells at N >= 1000, grouped by task.

#### `--task codebook` (22 cells)

```bash
# cbid_8b_em: emergent arm, 32 layers
python src/perfect_aliasing/identify_probe.py --model-id 8b --adapter adapters/rl_deceiver_8b --task codebook --n 1000 --seed 0 --out analysis/ident/cbid_8b_em.json

# cbid_8b_hi_em: emergent arm, 32 layers
python src/perfect_aliasing/identify_probe.py --model-id 8b --adapter adapters/rl_deceiver_8b --task codebook --n 1000 --seed 0 --out analysis/ident/cbid_8b_hi_em.json

# cbid_gemma-9b_em: emergent arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --adapter adapters/rl_deceiver_gemma-9b --task codebook --n 1000 --seed 0 --out analysis/ident/cbid_gemma-9b_em.json

# cbid_gemma-9b_em_s1: emergent arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --adapter adapters/rl_deceiver_gemma-9b --task codebook --n 1000 --seed 1 --out analysis/ident/cbid_gemma-9b_em_s1.json

# cbid_gemma-9b_em_s2: emergent arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --adapter adapters/rl_deceiver_gemma-9b --task codebook --n 1000 --seed 2 --out analysis/ident/cbid_gemma-9b_em_s2.json

# cbid_gemma-9b_ent_em: emergent arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --adapter adapters/rl_deceiver_gemma-9b --task codebook --n 1000 --seed 0 --out analysis/ident/cbid_gemma-9b_ent_em.json

# cbid_mistral-7b_em: emergent arm, 32 layers
python src/perfect_aliasing/identify_probe.py --model-id mistral-7b --adapter adapters/rl_deceiver_mistral-7b --task codebook --n 1000 --seed 0 --out analysis/ident/cbid_mistral-7b_em.json

# e2id_8b_em: emergent arm, 32 layers
python src/perfect_aliasing/identify_probe.py --model-id 8b --adapter adapters/rl_deceiver_8b --task codebook --n 1000 --seed 0 --out analysis/ident/e2id_8b_em.json

# e2id_8b_in: instructed arm, 32 layers
python src/perfect_aliasing/identify_probe.py --model-id 8b --instructed --task codebook --n 1000 --seed 0 --out analysis/ident/e2id_8b_in.json

# e2id_gemma-9b_em: emergent arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --adapter adapters/rl_deceiver_gemma-9b --task codebook --n 1000 --seed 0 --out analysis/ident/e2id_gemma-9b_em.json

# e2id_gemma-9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task codebook --n 1000 --seed 0 --out analysis/ident/e2id_gemma-9b_in.json

# e2id_qwen-14b_em: emergent arm, 48 layers
python src/perfect_aliasing/identify_probe.py --model-id qwen-14b --adapter adapters/rl_deceiver_qwen-14b --task codebook --n 1000 --seed 0 --out analysis/ident/e2id_qwen-14b_em.json

# e2id_qwen-14b_in: instructed arm, 48 layers
python src/perfect_aliasing/identify_probe.py --model-id qwen-14b --instructed --task codebook --n 1000 --seed 0 --out analysis/ident/e2id_qwen-14b_in.json

# geo_cb_g9b_em: emergent arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --adapter adapters/rl_deceiver_gemma-9b --task codebook --n 1000 --seed 0 --out analysis/ident/geo_cb_g9b_em.json

# geo_cb_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task codebook --n 1000 --seed 0 --out analysis/ident/geo_cb_g9b_in.json

# geo_cb_q14b_em: emergent arm, 48 layers
python src/perfect_aliasing/identify_probe.py --model-id qwen-14b --adapter adapters/rl_deceiver_qwen-14b --task codebook --n 1000 --seed 0 --out analysis/ident/geo_cb_q14b_em.json

# id8_cb_8b: emergent arm, 32 layers
python src/perfect_aliasing/identify_probe.py --model-id 8b --adapter adapters/rl_deceiver_8b --task codebook --n 1000 --seed 0 --out analysis/ident/id8_cb_8b.json

# id8_cb_mi7b: emergent arm, 32 layers
python src/perfect_aliasing/identify_probe.py --model-id mistral-7b --adapter adapters/rl_deceiver_mistral-7b --task codebook --n 1000 --seed 0 --out analysis/ident/id8_cb_mi7b.json

# symid_gemma-9b_em: emergent arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --adapter adapters/rl_deceiver_gemma-9b --task codebook --n 1000 --seed 0 --out analysis/ident/symid_gemma-9b_em.json

# symid_qwen-14b_em: emergent arm, 48 layers
python src/perfect_aliasing/identify_probe.py --model-id qwen-14b --adapter adapters/rl_deceiver_qwen-14b --task codebook --n 1000 --seed 0 --out analysis/ident/symid_qwen-14b_em.json

# xfer_fit_g9b_em: emergent arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --adapter adapters/rl_deceiver_gemma-9b --task codebook --n 1000 --seed 0 --out analysis/ident/xfer_fit_g9b_em.json

# xfer_fit_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task codebook --n 1000 --seed 0 --out analysis/ident/xfer_fit_g9b_in.json

```

#### `--task infercode` (20 cells)

```bash
# geo_inf_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 2000 --seed 0 --out analysis/ident/geo_inf_g9b_in.json

# id4_infcb_8b_in: instructed arm, 32 layers
python src/perfect_aliasing/identify_probe.py --model-id 8b --instructed --task infercode --n 2000 --seed 0 --out analysis/ident/id4_infcb_8b_in.json

# id4_infcb_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 2000 --seed 0 --out analysis/ident/id4_infcb_g9b_in.json

# id4_infcb_q14b_in: instructed arm, 48 layers
python src/perfect_aliasing/identify_probe.py --model-id qwen-14b --instructed --task infercode --n 2000 --seed 0 --out analysis/ident/id4_infcb_q14b_in.json

# id9_inf_q14b: instructed arm, 48 layers
python src/perfect_aliasing/identify_probe.py --model-id qwen-14b --instructed --task infercode --n 2000 --seed 0 --out analysis/ident/id9_inf_q14b.json

# infid_8b_in: instructed arm, 32 layers
python src/perfect_aliasing/identify_probe.py --model-id 8b --instructed --task infercode --n 1000 --seed 0 --out analysis/ident/infid_8b_in.json

# infid_gemma-9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 1000 --seed 0 --out analysis/ident/infid_gemma-9b_in.json

# infid_qwen-14b_in: instructed arm, 48 layers
python src/perfect_aliasing/identify_probe.py --model-id qwen-14b --instructed --task infercode --n 1000 --seed 0 --out analysis/ident/infid_qwen-14b_in.json

# ingr_full_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 2000 --seed 0 --ingredient-probes --out analysis/ident/ingr_full_g9b_in.json

# ingr_gt_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 2000 --seed 0 --infer-orientations gt --ingredient-probes --out analysis/ident/ingr_gt_g9b_in.json

# ingr_tplA_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 2000 --seed 0 --infer-templates 0,1 --ingredient-probes --out analysis/ident/ingr_tplA_g9b_in.json

# settle_ctl_gt_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 2000 --seed 1 --infer-orientations gt --load-probe probes/settle_gt_g9b_in.json --out analysis/ident/settle_ctl_gt_g9b_in.json

# settle_ctl_tplA_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 2000 --seed 1 --infer-templates 0,1 --load-probe probes/settle_tplA_g9b_in.json --out analysis/ident/settle_ctl_tplA_g9b_in.json

# settle_fit_gt_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 2000 --seed 0 --infer-orientations gt --out analysis/ident/settle_fit_gt_g9b_in.json

# settle_fit_tplA_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 2000 --seed 0 --infer-templates 0,1 --out analysis/ident/settle_fit_tplA_g9b_in.json

# settle_held_lt_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 2000 --seed 1 --infer-orientations lt --load-probe probes/settle_gt_g9b_in.json --out analysis/ident/settle_held_lt_g9b_in.json

# settle_held_tplB_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 2000 --seed 1 --infer-templates 2,3 --load-probe probes/settle_tplA_g9b_in.json --out analysis/ident/settle_held_tplB_g9b_in.json

# xfer_score_g9b_base: base arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --task infercode --n 1000 --seed 0 --load-probe probes/mixed_cb_g9b_in.json --out analysis/ident/xfer_score_g9b_base.json

# xfer_score_g9b_em: emergent arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --adapter adapters/rl_deceiver_gemma-9b --task infercode --n 1000 --seed 0 --load-probe probes/mixed_cb_g9b_em.json --out analysis/ident/xfer_score_g9b_em.json

# xfer_score_g9b_in: instructed arm, 42 layers
python src/perfect_aliasing/identify_probe.py --model-id gemma-9b --instructed --task infercode --n 1000 --seed 0 --load-probe probes/mixed_cb_g9b_in.json --out analysis/ident/xfer_score_g9b_in.json

```

### Stage 3: the other measurements

```bash
# behavioural rates (ally truth rate, rival deception rate) for any arm
python src/perfect_aliasing/behavior.py --model-id gemma-9b --adapter adapters/rl_deceiver_gemma-9b --n 1000 \
    --out analysis/behavior/gemma-9b_em.json

# the instruction-pair protocol and its closed-form aliasing rate (paper Appendix D)
python src/perfect_aliasing/instrpair_probe.py --model-id gemma-9b --n 1000 \
    --out analysis/instrpair/gemma-9b.json

# the direct geometric comparison (paper Appendix on geometry)
python src/perfect_aliasing/geom_compare.py --model-id gemma-9b --n 1000 \
    --out analysis/geom/gemma-9b.json

# causal interventions: patch, ablate, steer
python src/perfect_aliasing/interventions.py --model-id gemma-9b --adapter adapters/rl_deceiver_gemma-9b \
    --mode all --out analysis/interv/gemma-9b_em.json

# decodability along the RL trajectory, from saved checkpoints
python src/perfect_aliasing/dynamics_probe.py --model-id 8b --checkpoints runs/rl_8b_s0_dyn/checkpoints \
    --out analysis/dynamics/8b_s0.json
```

### Stage 4: figures

The renderer reads only the consolidated results file, so this step needs no accelerator:

```bash
PAPER=1 python experiments/002-generalization-dynamics/make_figures.py
```

`PAPER=1` renders at the paper's printed text width. Without it the renderer emits wider
screen-sized figures whose labels are too small once scaled into the document.

### A note on exact reproduction

Sampling and probe fitting are seeded, so a rerun on the same model revision reproduces the
reported numbers. The `--seed` values above are the ones recorded in the results file. Hub model
revisions are not pinned by this repository, so a model updated upstream is the one thing that can
move a number without any change here.
