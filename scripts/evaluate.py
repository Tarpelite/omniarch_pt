"""Benchmark a checkpoint on every prepared test set.

Protocol (PDEBench style): give the first 10 frames, roll out autoregressively to the
last frame, nRMSE over all forecast frames, averaged over samples and channels.
Reference: copy the last given frame forward.

    python scripts/evaluate.py --ckpt runs/mini/omniarch.pt --data DIR --out results/mini.json
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.data import family, load_entries, nrmse  # noqa: E402
from omniarch.inference import OmniArchPredictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--history", type=int, default=10, help="number of given frames")
ap.add_argument("--start", type=int, default=None,
                help="first forecast frame (default = history); with --start 10 --history K every K forecasts the same frames")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
start = a.history if a.start is None else a.start
assert 1 <= a.history <= start

p = OmniArchPredictor(a.ckpt, a.device)
res = {}
for e in load_entries(a.data):
    arr = e.array("test")
    chans = e.meta["channels"]
    bs = 4 if e.dim == 3 else 25
    preds, t0 = [], time.time()
    for i in range(0, arr.shape[0], bs):
        x = np.asarray(arr[i:i + bs])
        preds.append(p.predict(x[:, start - a.history:start], x.shape[1] - start, batched=True))
    pred = torch.from_numpy(np.concatenate(preds))
    true = torch.from_numpy(np.asarray(arr[:, start:]))
    last = torch.from_numpy(np.asarray(arr[:, start - 1:start])).expand_as(true)
    res[e.name] = dict(family=family(e.name), dim=e.dim, n=int(arr.shape[0]), omniarch=nrmse(pred, true),
                       copy_last=nrmse(last, true), seconds=time.time() - t0)
    print(f"{e.name:28s} OmniArch {res[e.name]['omniarch']:.4f}   copy-last {res[e.name]['copy_last']:.4f}", flush=True)

fam = defaultdict(list)
for v in res.values():
    fam[v["family"]].append(v)
summary = {f: dict(dim=v[0]["dim"], omniarch=float(np.mean([x["omniarch"] for x in v])),
                   copy_last=float(np.mean([x["copy_last"] for x in v])), files=len(v)) for f, v in fam.items()}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({"per_file": res, "per_family": summary, "ckpt": os.path.basename(a.ckpt), "history": a.history, "start": start},
          open(a.out, "w"), indent=1)
print("\nfamily            dim  OmniArch  copy-last")
for f, v in summary.items():
    print(f"{f:16s} {v['dim']}D   {v['omniarch']:.4f}    {v['copy_last']:.4f}")
print(f"{'mean':16s}      {np.mean([v['omniarch'] for v in summary.values()]):.4f}    "
      f"{np.mean([v['copy_last'] for v in summary.values()]):.4f}")
