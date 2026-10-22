#!/bin/bash
# Throughput of scripts/train.py on one GPU: eager, compiled, and two compiled runs sharing the GPU.
#   srun -p gpu --gres=gpu:1 -c 12 --mem=60G bash scripts/slurm/bench_pack.sh [preset] [steps]
set -a; source .env.$(hostname -s) 2>/dev/null || source .env; set +a
P=${1:-mini}; S=${2:-600}; O=/tmp/oa_bench_$$; mkdir -p $O
run() {  # name port extra-args...
    local n=$1 port=$2; shift 2
    $OA_PYTHON -m torch.distributed.run --nproc_per_node 1 --master_port $port scripts/train.py --preset $P \
        --data $OA_DATA --out $O/$n --steps $S --eval_every 1000000 --balanced --augment "$@" > $O/$n.log 2>&1
}
# steady-state rate over the second half (the logged sps is cumulative and includes compile time)
sps() { $OA_PYTHON -c "
import json; L=[json.loads(l) for l in open('$O/$1/log.jsonl') if '\"sps\"' in l]
t=lambda r: (r['step']+1)/r['sps']; a=L[len(L)//2]; b=L[-2]
print(f'{(b[\"step\"]-a[\"step\"])/(t(b)-t(a)):.2f}')"; }
export OMNIARCH_FUSED=1
run eager 29510;                       echo "$P eager:            $(sps eager) steps/s"
run comp 29511 --compile;              echo "$P compiled:         $(sps comp) steps/s"
run pa 29512 --compile & run pb 29513 --compile & wait
echo "$P 2 compiled jobs on one GPU: $(sps pa) + $(sps pb) steps/s"
rm -rf $O
