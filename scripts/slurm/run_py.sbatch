#!/bin/bash
# Run one or more Python commands from the repo with the node's environment (.env.<hostname> or .env).
# Each argument is one command line for $OA_PYTHON; several arguments run side by side on the same GPU.
#   sbatch -J NAME -p lite --gres=gpu:1 scripts/slurm/run_py.sbatch "scripts/evaluate.py --ckpt ..." "..."
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o slurm-%x-%j.out
set -eo pipefail
ENV_FILE=.env.$(hostname -s); [ -f "$ENV_FILE" ] || ENV_FILE=.env
set -a; source "$ENV_FILE"; set +a
OA_REPO=${REPO_OVERRIDE:-$OA_REPO}
cd "$OA_REPO"
pids=()
for cmd in "$@"; do
    cmd=${cmd//\$OA_DATA/$OA_DATA}; cmd=${cmd//\$OA_RUNS/$OA_RUNS}
    echo ">>> $cmd"
    $OA_PYTHON $cmd &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
