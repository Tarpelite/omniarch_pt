#!/bin/bash
# Single-node multi-GPU pre-training. Submit through scripts/slurm/submit.sh train <preset>.
set -euo pipefail
cd "$OA_REPO"
export OMP_NUM_THREADS=4
"$OA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$OA_GPUS" scripts/train.py \
  --data "$OA_DATA" --out "$OA_RUNS/$PRESET" --preset "$PRESET" --balanced --augment ${EXTRA:-}
