"""Does the model read the time step from its context? Re-sample every test trajectory with a
frame stride never used in training, give K frames, forecast H steps, compare with copying the
last frame.

    python scripts/eval_stride.py --ckpt weights/omniarch-base.pt --data DIR --out results/stride_base.json

Stride 1 is the training spacing; strides 2-4 are 2-4x larger time steps.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.data import family, load_entries  # noqa: E402
from omniarch.inference import OmniArchPredictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--strides", default="1,2,3,4")
ap.add_argument("--history", type=int, default=4)
ap.add_argument("--horizon", type=int, default=4)
ap.add_argument("--device", default="auto")
ap.add_argument("--max_traj", type=int, default=50)
ap.add_argument("--pass_dt", action="store_true", help="tell the model the stride (for time_pos=stride models)")
a = ap.parse_args()

p = OmniArchPredictor(a.ckpt, a.device)
strides = [int(s) for s in a.strides.split(",")]


def nrmse(pred, true):                         # (B, T, C, *S): per sample and channel over time and space
    red = (1,) + tuple(range(3, pred.ndim))
    return float((np.sqrt(((pred - true) ** 2).sum(red)) / np.sqrt((true ** 2).sum(red)).clip(1e-12)).mean())


res = {}
t0 = time.time()
for e in load_entries(a.data):
    arr = np.asarray(e.array("test")[:a.max_traj], dtype=np.float32)
    row = {}
    for s in strides:
        x = arr[:, ::s]
        need = a.history + 1
        if x.shape[1] < need:
            continue
        h = min(a.horizon, x.shape[1] - a.history)
        bs = 4 if e.dim == 3 else 16
        pred = np.concatenate([p.predict(x[i:i + bs, :a.history], h, batched=True, dt=s if a.pass_dt else 1.0)
                               for i in range(0, len(x), bs)])
        true = x[:, a.history:a.history + h]
        last = np.repeat(x[:, a.history - 1:a.history], h, 1)
        row[s] = dict(model=nrmse(pred, true), copy_last=nrmse(last, true), horizon=h)
    res[e.name] = dict(family=family(e.name), dim=e.dim, strides=row)
    print(e.name, {s: (round(v["model"], 3), round(v["copy_last"], 3)) for s, v in row.items()}, f"{time.time() - t0:.0f}s", flush=True)

# family means per stride, and the ratio model / copy-last (below 1 = better than doing nothing)
fam = {}
for name, r in res.items():
    for s, v in r["strides"].items():
        fam.setdefault(r["family"], {}).setdefault(s, []).append(v)
summary = {f: {s: dict(model=float(np.mean([v["model"] for v in vs])), copy_last=float(np.mean([v["copy_last"] for v in vs])))
               for s, vs in d.items()} for f, d in fam.items()}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(dict(ckpt=os.path.basename(a.ckpt), history=a.history, horizon=a.horizon, per_dataset=res, per_family=summary),
          open(a.out, "w"), indent=1)
print("\nfamily            " + "  ".join(f"stride {s}: model/copy" for s in strides))
for f, d in summary.items():
    print(f"{f:16s}  " + "  ".join(f"{d[s]['model']:.3f}/{d[s]['copy_last']:.3f}" if s in d else "        -      " for s in strides))
