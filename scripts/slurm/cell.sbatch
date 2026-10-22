#!/bin/bash
# One scaling run on one GPU, then a full benchmark. Not pinned to a node: paths come from
# .env.<hostname> if present, else .env. Set PRESET, FRAC, STEPS, SEED via --export.
#   sbatch -p gpu --gres=gpu:1 -c 12 --mem=60G --export=ALL,PRESET=mini,FRAC=1.0,STEPS=72000,SEED=0 scripts/slurm/cell.sbatch
set -euo pipefail
cd "${OA_REPO_SUBMIT:-$SLURM_SUBMIT_DIR}"
ENV_FILE=.env.$(hostname -s); [ -f "$ENV_FILE" ] || ENV_FILE=.env
set -a; source "$ENV_FILE"; set +a
cd "$OA_REPO"
OUT="$OA_RUNS/scaling/${PRESET}_d${FRAC}_s${STEPS}_seed${SEED}"
export OMP_NUM_THREADS=4
"$OA_PYTHON" -m torch.distributed.run --standalone --nproc_per_node 1 --master_port $((20000 + SLURM_JOB_ID % 20000)) \
  scripts/train.py --data "$OA_DATA" --out "$OUT" --preset "$PRESET" --data_frac "$FRAC" --steps "$STEPS" --seed "$SEED" \
  --eval_every $((STEPS / 3)) --balanced --augment
"$OA_PYTHON" scripts/evaluate.py --ckpt "$OUT/omniarch.pt" --data "$OA_DATA" --out "$OUT/results.json"
