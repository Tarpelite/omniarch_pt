#!/bin/bash
# One cell of the scaling grid per array task: train on one GPU, then benchmark on the full test sets.
# Submit through scripts/slurm/submit.sh scaling.
set -euo pipefail
cd "$OA_REPO"
GRID=(nano:0.25 nano:0.5 nano:1.0 tiny:0.25 tiny:0.5 tiny:1.0 mini:0.25 mini:0.5 mini:1.0 base:0.25 base:0.5 base:1.0)
CELL=${GRID[$SLURM_ARRAY_TASK_ID]}
PRESET=${CELL%%:*}; FRAC=${CELL##*:}
OUT="$OA_RUNS/scaling/${PRESET}_d${FRAC}${SCALING_STEPS:+_s$SCALING_STEPS}"   # longer runs get a _s<steps> suffix
export OMP_NUM_THREADS=4
"$OA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node 1 --master_port $((29500 + SLURM_ARRAY_TASK_ID)) \
  scripts/train.py --data "$OA_DATA" --out "$OUT" --preset "$PRESET" --data_frac "$FRAC" \
  --steps "${SCALING_STEPS:-12000}" --eval_every 4000 --balanced --augment
"$OA_PYTHON" scripts/evaluate.py --ckpt "$OUT/omniarch.pt" --data "$OA_DATA" --out "$OUT/results.json"
