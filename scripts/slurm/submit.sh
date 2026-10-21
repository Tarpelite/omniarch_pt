#!/bin/bash
# Submit data preparation or training to Slurm, using the paths in .env (see .env.example).
#
#   scripts/slurm/submit.sh prepare                 # one array task per PDEBench file
#   scripts/slurm/submit.sh train mini              # pre-train a preset (mini | tiny)
#   scripts/slurm/submit.sh train mini --steps 1000 # extra arguments go to scripts/train.py
#   scripts/slurm/submit.sh scaling                 # model-size x data-size grid, one GPU per cell
#   OA_DRYRUN=1 scripts/slurm/submit.sh train mini  # check the request with sbatch --test-only, submit nothing
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] || { echo "missing .env: cp .env.example .env and edit it"; exit 1; }
set -a; source .env; set +a
mkdir -p "$OA_LOGS"
NODE_ARGS=(); [ -n "${SLURM_NODE:-}" ] && NODE_ARGS=(-w "$SLURM_NODE")
SBATCH=(sbatch); [ -n "${OA_DRYRUN:-}" ] && SBATCH=(sbatch --test-only)

case "${1:-}" in
  prepare)
    N=$("$OA_PYTHON" scripts/prepare_pdebench.py --root x --out x --list | wc -l)
    "${SBATCH[@]}" -J oa-prep -p "$SLURM_CPU_PARTITION" -c 4 --mem=24G -t 12:00:00 --array=0-$((N - 1))%24 \
      -o "$OA_LOGS/%x-%A_%a.out" --export=ALL scripts/slurm/prepare.sbatch ;;
  train)
    PRESET=${2:-mini}; shift $(( $# >= 2 ? 2 : 1 ))
    "${SBATCH[@]}" -J "oa-$PRESET" -p "$SLURM_GPU_PARTITION" "${NODE_ARGS[@]}" --gres=gpu:"$OA_GPUS" -c "$OA_CPUS" \
      --mem="$OA_MEM" -t 7-00:00:00 -o "$OA_LOGS/%x-%j.out" \
      --export=ALL,PRESET="$PRESET",EXTRA="$*" scripts/slurm/train.sbatch ;;
  scaling)
    # 4 model sizes x 3 data fractions, one GPU each (see scripts/slurm/scaling.sbatch)
    "${SBATCH[@]}" -J oa-scaling -p "$SLURM_GPU_PARTITION" "${NODE_ARGS[@]}" --gres=gpu:1 -c 12 --mem=60G \
      -t 1-00:00:00 --array=0-11%"$OA_GPUS" -o "$OA_LOGS/%x-%A_%a.out" --export=ALL scripts/slurm/scaling.sbatch ;;
  *)
    sed -n 2,9p "$0"; exit 1 ;;
esac
