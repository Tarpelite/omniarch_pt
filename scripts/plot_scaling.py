"""Scaling curves from the grid in scripts/slurm/scaling.sbatch.

    python scripts/plot_scaling.py RUNS/scaling docs/scaling.png results/scaling.json

Reads <cell>/results.json (scripts/evaluate.py) and <cell>/args.json for every cell named
<preset>_d<fraction>, plots mean nRMSE over PDE families against parameter count, one line per
data fraction, fits err = a * N^-alpha per line, and writes the numbers to a JSON summary.
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FAMILIES = ["adv", "burgers", "cfd1d", "diffsorp", "reacdiff", "cfd2d", "diffreact2d", "swe2d", "cfd3d"]
DIMS = {1: ["adv", "burgers", "cfd1d", "diffsorp", "reacdiff"], 2: ["cfd2d", "diffreact2d", "swe2d"], 3: ["cfd3d"]}

root, fig_path, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
cells = []
for d in sorted(glob.glob(os.path.join(root, "*_d*"))):
    r, a = os.path.join(d, "results.json"), os.path.join(d, "args.json")
    if not (os.path.exists(r) and os.path.exists(a)):
        continue
    fam = json.load(open(r))["per_family"]
    args = json.load(open(a))
    cells.append(dict(preset=args["preset"], frac=args.get("data_frac", 1.0), params=args["params"],
                      mean=float(np.mean([fam[f]["omniarch"] for f in FAMILIES])),
                      dims={k: float(np.mean([fam[f]["omniarch"] for f in v])) for k, v in DIMS.items()},
                      baseline=float(np.mean([fam[f]["copy_last"] for f in FAMILIES]))))
if not cells:
    sys.exit("no finished cells")

fracs = sorted({c["frac"] for c in cells})
colors = {f: c for f, c in zip(fracs, ["#9ecae1", "#4292c6", "#08519c"][-len(fracs):])}
fits = {}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
for f in fracs:
    cs = sorted([c for c in cells if c["frac"] == f], key=lambda c: c["params"])
    n, e = np.array([c["params"] for c in cs]), np.array([c["mean"] for c in cs])
    ax.plot(n, e, "o-", color=colors[f], lw=1.8, ms=5, label=f"{int(f * 100)}% of training data")
    if len(cs) >= 2:
        slope, icpt = np.polyfit(np.log(n), np.log(e), 1)
        fits[str(f)] = dict(alpha=-slope, a=float(np.exp(icpt)), points=len(cs))
ax.axhline(cells[0]["baseline"], color="0.45", ls="--", lw=1.2)
ax.text(min(c["params"] for c in cells), cells[0]["baseline"] * 1.04,
        "copy last frame", color="0.35", fontsize=9)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("parameters"); ax.set_ylabel("mean nRMSE over 9 PDE families")
ax.set_title("Error vs. model size", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.grid(True, which="both", color="0.9", lw=0.6)

full = sorted([c for c in cells if c["frac"] == max(fracs)], key=lambda c: c["params"])
for k, col in zip((1, 2, 3), ("#1b9e77", "#d95f02", "#7570b3")):
    ax2.plot([c["params"] for c in full], [c["dims"][k] for c in full], "o-", color=col, lw=1.8, ms=5, label=f"{k}D")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("parameters"); ax2.set_ylabel("mean nRMSE")
ax2.set_title(f"By dimension ({int(max(fracs) * 100)}% data)", fontsize=11)
ax2.legend(frameon=False, fontsize=9)
ax2.grid(True, which="both", color="0.9", lw=0.6)
os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
fig.savefig(fig_path, dpi=160)

summary = dict(cells=sorted(cells, key=lambda c: (c["frac"], c["params"])), power_law_fits=fits)
json.dump(summary, open(out_json, "w"), indent=1)
print("| preset | params | " + " | ".join(f"{int(f * 100)}% data" for f in fracs) + " |")
print("|---|---|" + "---|" * len(fracs))
for p in sorted({c["preset"] for c in cells}, key=lambda p: min(c["params"] for c in cells if c["preset"] == p)):
    row = [c for c in cells if c["preset"] == p]
    vals = " | ".join(next((f"{c['mean']:.3f}" for c in row if c["frac"] == f), "-") for f in fracs)
    print(f"| {p} | {row[0]['params'] / 1e6:.1f}M | {vals} |")
for f, v in fits.items():
    print(f"fit {int(float(f) * 100)}% data: err ~ N^-{v['alpha']:.3f} ({v['points']} points)")
