#!/bin/bash
# Array task: convert one PDEBench file. Submit through scripts/slurm/submit.sh prepare.
set -euo pipefail
cd "$OA_REPO"
NAMES=($("$OA_PYTHON" scripts/prepare_pdebench.py --root x --out x --list))
"$OA_PYTHON" scripts/prepare_pdebench.py --root "$PDEBENCH_ROOT" --out "$OA_DATA" --only "${NAMES[$SLURM_ARRAY_TASK_ID]}"
