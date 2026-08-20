#!/usr/bin/env bash
# Reproduce Experiment 001 locally (the local-first path). Each step calls a src/perfect_aliasing CLI and
# writes its JSON into ./data; the last step renders figures. For the real run we launch GPU jobs;
# this is the portable recipe and also the CPU-smoke driver:
#
#   bash run.sh                          # full run (needs a GPU to be practical)
#   EPOCHS=2 N=20 bash run.sh            # tiny CPU smoke to validate the pipeline end-to-end
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SRC="$ROOT/src/perfect_aliasing"
DATA="$HERE/data"

ADAPTER="${ADAPTER:-$ROOT/adapters/rl_deceiver}"   # gitignored; large
EPOCHS="${EPOCHS:-1000}"
N="${N:-1000}"
mkdir -p "$DATA"

# 1. Train the emergent (RL) deceiver → LoRA adapter + history.json
python "$SRC/train_rl.py" --epochs "$EPOCHS" --out "$ADAPTER"
cp "$ADAPTER/history.json" "$DATA/history.json"

# 2. Behavior — both arms (ally truth rate / rival deception rate)
python "$SRC/behavior.py" --adapter "$ADAPTER" --n "$N" --out "$DATA/behavior_emergent.json"
python "$SRC/behavior.py" --instructed         --n "$N" --out "$DATA/behavior_instructed.json"

# 3. Cross-context truth probe (RQ 1.1) — both arms
python "$SRC/probes.py" --adapter "$ADAPTER" --n "$N" --out "$DATA/probe_emergent.json"
python "$SRC/probes.py" --instructed         --n "$N" --out "$DATA/probe_instructed.json"

# 4. Causal interventions: patch / ablate / steer (RQ 1.2) — both arms
python "$SRC/interventions.py" --adapter "$ADAPTER" --mode all --out "$DATA/interv_emergent.json"
python "$SRC/interventions.py" --instructed         --mode all --out "$DATA/interv_instructed.json"

# 5. Figures
python "$HERE/make_figures.py"
echo "Experiment 001 complete — see figures/ and data/."
