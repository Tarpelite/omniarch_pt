"""Summarise every scaling run: model size, training steps, data fraction and seed.

    python scripts/analyze_scaling.py RUNS/scaling docs/scaling.png results/scaling.json

Each run directory holds args.json (scripts/train.py) and results.json (scripts/evaluate.py).
Prints markdown tables, fits err = a * N^-alpha at 12k steps, and draws a three-panel figure.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FAMILIES = ["adv", "burgers", "cfd1d", "diffsorp", "reacdiff", "cfd2d", "diffreact2d", "swe2d", "cfd3d"]
DIMS = {1: ["adv", "burgers", "cfd1d", "diffsorp", "reacdiff"], 2: ["cfd2d", "diffreact2d", "swe2d"], 3: ["cfd3d"]}
ORDER = ["nano", "tiny", "mini", "base", "large"]

root, fig_path, out_json = sys.argv[1:4]
runs = []
for d in sorted(glob.glob(os.path.join(root, "*"))):
    r, a = os.path.join(d, "results.json"), os.path.join(d, "args.json")
    if not (os.path.exists(r) and os.path.exists(a)):
        continue
    fam = json.load(open(r))["per_family"]
    args = json.load(open(a))
    runs.append(dict(name=os.path.basename(d), preset=args["preset"], frac=args.get("data_frac", 1.0),
                     steps=args["steps"], seed=args.get("seed", 0), params=args["params"],
                     mean=float(np.mean([fam[f]["omniarch"] for f in FAMILIES])),
                     dims={k: float(np.mean([fam[f]["omniarch"] for f in v])) for k, v in DIMS.items()},
                     fam={f: fam[f]["omniarch"] for f in FAMILIES},
                     baseline=float(np.mean([fam[f]["copy_last"] for f in FAMILIES]))))
if not runs:
    sys.exit("no finished runs")

def group(**kw):
    return [r for r in runs if all(r[k] == v for k, v in kw.items())]

# ---- A: model size at 12k steps, full data, all seeds ----
print("A. model size (12k steps, 100% data); mean ± std over seeds\n")
print("| model | params | seeds | mean nRMSE | 1D | 2D | 3D |\n|---|---|---|---|---|---|---|")
size_pts = []
for p in ORDER:
    g = group(preset=p, frac=1.0, steps=12000)
    if not g:
        continue
    m = np.array([r["mean"] for r in g])
    dims = {k: np.mean([r["dims"][k] for r in g]) for k in (1, 2, 3)}
    sd = f" ± {m.std(ddof=1):.3f}" if len(g) > 1 else ""
    print(f"| {p} | {g[0]['params'] / 1e6:.0f}M | {len(g)} | {m.mean():.3f}{sd} | {dims[1]:.3f} | {dims[2]:.3f} | {dims[3]:.3f} |")
    size_pts.append((g[0]["params"], m.mean(), m.std(ddof=1) if len(g) > 1 else 0.0, p, dims))
n = np.array([s[0] for s in size_pts]); e = np.array([s[1] for s in size_pts])
slope, icpt = np.polyfit(np.log(n), np.log(e), 1)
fit = dict(alpha=-slope, a=float(np.exp(icpt)), points=[s[3] for s in size_pts])
dim_fit = {}
for k in (1, 2, 3):
    sk, _ = np.polyfit(np.log(n), np.log([s[4][k] for s in size_pts]), 1)
    dim_fit[k] = -sk
print(f"\nfit: err ~ N^-{fit['alpha']:.3f}  (1D {dim_fit[1]:.3f}, 2D {dim_fit[2]:.3f}, 3D {dim_fit[3]:.3f})")
for target in (1e9, 2e9):
    print(f"extrapolated mean nRMSE at {target / 1e9:.0f}B params, 12k steps: {fit['a'] * target ** -fit['alpha']:.3f}")

# ---- B: training steps x data fraction (seed 0) ----
print("\nB. training steps and data fraction (seed 0): mean nRMSE\n")
steps_all = sorted({r["steps"] for r in runs})
print("| model | data | " + " | ".join(f"{s // 1000}k steps" for s in steps_all) + " |\n|---|---|" + "---|" * len(steps_all))
for p in ORDER:
    for f in (1.0, 0.5, 0.25):
        row = {r["steps"]: r["mean"] for r in group(preset=p, frac=f, seed=0)}
        if len(row) < 1 or (len(row) == 1 and 12000 in row and f == 0.5):
            continue
        print(f"| {p} | {int(f * 100)}% | " + " | ".join(f"{row[s]:.3f}" if s in row else "-" for s in steps_all) + " |")

# ---- figure ----
fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
ax = axs[0]
ax.errorbar(n, e, yerr=[s[2] for s in size_pts], fmt="o", color="#08519c", ms=5, capsize=3, zorder=3)
xs = np.geomspace(n.min() * 0.8, 2.2e9, 50)
ax.plot(xs, fit["a"] * xs ** -fit["alpha"], "--", color="#08519c", lw=1.2, label=f"fit  N^-{fit['alpha']:.2f}")
for s in size_pts:
    ax.annotate(s[3], (s[0], s[1]), textcoords="offset points", xytext=(4, 5), fontsize=8, color="0.3")
ax.axhline(runs[0]["baseline"], color="0.5", ls=":", lw=1)
ax.text(n.min(), runs[0]["baseline"] * 1.03, "copy last frame", fontsize=8, color="0.4")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("parameters"); ax.set_ylabel("mean nRMSE (9 PDE families)")
ax.set_title("Model size (12k steps, mean ± sd over seeds)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.grid(True, which="both", color="0.92", lw=0.6)

ax = axs[1]
cols = {"nano": "#9ecae1", "tiny": "#6baed6", "mini": "#3182bd", "base": "#08519c", "large": "#08306b"}
for p in ORDER:
    for f, ls in ((1.0, "-"), (0.25, "--")):
        g = sorted(group(preset=p, frac=f, seed=0), key=lambda r: r["steps"])
        if len(g) >= 2:
            ax.plot([r["steps"] for r in g], [r["mean"] for r in g], ls, marker="o", ms=4, color=cols[p],
                    label=f"{p}" if f == 1.0 else None)
ax.plot([], [], "k-", label="100% data"); ax.plot([], [], "k--", label="25% data")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("training steps"); ax.set_ylabel("mean nRMSE")
ax.set_title("Training length and data", fontsize=10)
ax.legend(frameon=False, fontsize=8, ncol=2); ax.grid(True, which="both", color="0.92", lw=0.6)

ax = axs[2]
for k, col in zip((1, 2, 3), ("#1b9e77", "#d95f02", "#7570b3")):
    ax.plot(n, [s[4][k] for s in size_pts], "o-", color=col, ms=4, label=f"{k}D  (N^-{dim_fit[k]:.2f})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("parameters"); ax.set_ylabel("mean nRMSE")
ax.set_title("By dimension (12k steps)", fontsize=10)
ax.legend(frameon=False, fontsize=8); ax.grid(True, which="both", color="0.92", lw=0.6)
os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
fig.savefig(fig_path, dpi=160)
json.dump(dict(runs=runs, size_fit=fit, dim_fit=dim_fit), open(out_json, "w"), indent=1)
