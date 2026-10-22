"""Score test trajectories of 2D datasets by the structural error of a full forecast, to pick showcase cases.

    python scripts/pick_cases.py --ckpt runs/release_v0.4/omniarch-base.pt --data PROCESSED_DIR --out picks.json

Every test trajectory is forecast from its first K frames to its end. Each (trajectory, channel) gets:
  err     = ||a(pred_last) - a(true_last)|| / ||a(true_last)||, with a(x) = x minus the spatial mean (structure only)
  change  = ||a(true_K) - a(true_last)|| / ||a(true_K)||, how much the channel moves after the given frames
Only pairs with change > --min_change count as candidates, so fields that barely move do not win.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.inference import OmniArchPredictor  # noqa: E402

DATASETS = ["swe2d", "diffreact2d", "cfd2d_rand_m0.1_eta0.1", "cfd2d_rand_m1.0_eta0.1",
            "cfd2d_rand_m0.1_eta0.01", "cfd2d_rand_m1.0_eta0.01"]


def anomaly(x):
    return x - x.mean(axis=(-2, -1), keepdims=True)


def norms(x):
    return np.sqrt((x.astype(np.float64) ** 2).sum(axis=(-2, -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/data/users/chenty/processed/omniarch_mini_v2")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--history", type=int, default=10, help="given frames K")
    ap.add_argument("--min_change", type=float, default=0.3)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    p = OmniArchPredictor(a.ckpt, a.device)
    print(f"checkpoint {p.ckpt_path} on {p.device}", flush=True)
    result = {"ckpt": p.ckpt_path, "history": a.history, "min_change": a.min_change, "datasets": {}}

    for name in a.datasets:
        meta = json.load(open(os.path.join(a.data, name, "meta.json")))
        chans = meta["channels"]
        test = np.load(os.path.join(a.data, name, "test.npy"), mmap_mode="r")
        test = np.asarray(test, dtype=np.float32)
        N, T, C = test.shape[:3]
        K = a.history
        steps = T - K
        preds = []
        for i in range(0, N, a.batch):
            preds.append(p.predict(test[i:i + a.batch, :K], steps, batched=True))
        pred = np.concatenate(preds)                              # (N, steps, C, H, W)
        true = test[:, K:]

        err = norms(anomaly(pred[:, -1]) - anomaly(true[:, -1])) / norms(anomaly(true[:, -1])).clip(1e-12)
        change = norms(anomaly(test[:, K]) - anomaly(test[:, -1])) / norms(anomaly(test[:, K])).clip(1e-12)
        err, change = np.nan_to_num(err, nan=np.inf), np.nan_to_num(change, nan=0.0)

        records = []
        for n in range(N):
            for c in range(C):
                records.append({"traj": n, "channel": chans[c], "err": float(err[n, c]),
                                "change": float(change[n, c]), "eligible": bool(change[n, c] > a.min_change)})
        cand = sorted([r for r in records if r["eligible"]], key=lambda r: r["err"])
        top = cand[:a.top]
        n_elig = len(cand)
        print(f"\n== {name}  (N={N}, T={T}, C={C}, channels={chans}, eligible {n_elig}/{len(records)})", flush=True)
        print(f"{'rank':>4} {'traj':>5} {'channel':>12} {'err':>8} {'change':>8}", flush=True)
        for r_i, r in enumerate(top, 1):
            print(f"{r_i:>4} {r['traj']:>5} {r['channel']:>12} {r['err']:>8.4f} {r['change']:>8.3f}", flush=True)
        if cand:
            med = float(np.median([r["err"] for r in cand]))
            print(f"median err over eligible pairs: {med:.4f}", flush=True)
        result["datasets"][name] = {"channels": chans, "n_traj": N, "n_eligible": n_elig,
                                    "median_err_eligible": float(np.median([r["err"] for r in cand])) if cand else None,
                                    "top": top, "records": records}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"\nsaved {a.out}", flush=True)


if __name__ == "__main__":
    main()
