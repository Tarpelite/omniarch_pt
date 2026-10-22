"""Cross-dimension study: does training on 1D, 2D and 3D together help each dimension?

    python scripts/analyze_crossdim.py RUNS/crossdim docs/crossdim.png results/crossdim.json

Each run directory holds args.json (scripts/train.py with --dims / --dim_frac) and results.json
(scripts/evaluate.py). Per-dimension error = mean nRMSE over the families of that dimension; 1D leaves
out reaction-diffusion, whose PDEBench trajectories are constant fields after the first saved frame.
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

DIMS = {1: ["adv", "burgers", "cfd1d", "diffsorp"], 2: ["cfd2d", "diffreact2d", "swe2d"], 3: ["cfd3d"]}
root, fig_path, out_json = sys.argv[1:4]

groups = defaultdict(list)          # (preset, dims, dim_frac) -> list of {dim: err}
for d in sorted(glob.glob(os.path.join(root, "*"))):
    r, a = os.path.join(d, "results.json"), os.path.join(d, "args.json")
    if not (os.path.exists(r) and os.path.exists(a)):
        continue
    fam = json.load(open(r))["per_family"]
    args = json.load(open(a))
    dims = args.get("dims", "123")
    groups[(args["preset"], dims, args.get("dim_frac", ""))].append(
        {k: float(np.mean([fam[f]["omniarch"] for f in v])) for k, v in DIMS.items() if str(k) in dims})

table = {}
for (preset, dims, frac), runs in sorted(groups.items()):
    keys = runs[0].keys()
    table[f"{preset} dims={dims}{' ' + frac if frac else ''}"] = dict(
        preset=preset, dims=dims, dim_frac=frac, seeds=len(runs),
        **{f"{k}D": float(np.mean([r[k] for r in runs])) for k in keys},
        **{f"{k}D_seeds": [round(r[k], 4) for r in runs] for k in keys})

print("| model | trained on | seeds | 1D | 2D | 3D |\n|---|---|---|---|---|---|")
for name, v in table.items():
    cells = [f"{v[f'{k}D']:.3f}" if f"{k}D" in v else "" for k in (1, 2, 3)]
    label = "+".join(f"{c}D" for c in v["dims"]) + (f" (3D data {v['dim_frac'].split(':')[1]})" if v["dim_frac"] else "")
    print(f"| {v['preset']} | {label} | {v['seeds']} | " + " | ".join(cells) + " |")


def get(preset, dims, frac=""):
    return table.get(f"{preset} dims={dims}{' ' + frac if frac else ''}")


# figure: left, change of each dimension's error when trained jointly; right, 3D with a quarter of the 3D data
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
ax = axs[0]
w = 0.36
for i, (preset, col) in enumerate((("tiny", "#9ecae1"), ("mini", "#2171b5"))):
    single = [get(preset, str(k)) for k in (1, 2, 3)]
    joint = get(preset, "123")
    if not joint or not all(single):
        continue
    ch = [100 * (joint[f"{k}D"] / single[k - 1][f"{k}D"] - 1) for k in (1, 2, 3)]
    ax.bar(np.arange(3) + (i - 0.5) * w, ch, w, color=col, label=preset)
    for x, c in zip(np.arange(3) + (i - 0.5) * w, ch):
        ax.text(x, max(c, 0) + 0.5, "0%" if abs(c) < 0.5 else f"{c:+.0f}%", ha="center", fontsize=8)
ax.axhline(0, color="0.3", lw=0.8)
ax.set_ylim(-2, max(16, ax.get_ylim()[1] + 2))
ax.set_xticks(range(3), ["1D", "2D", "3D"])
ax.set_ylabel("error change vs. one dimension only (%)")
ax.set_title("Full data: joint 1D+2D+3D training", fontsize=10)
ax.legend(frameon=False, fontsize=8)

ax = axs[1]
labels = ["3D only,\nall 3D data", "3D only,\n25% 3D data", "1D+2D+3D,\n25% 3D data"]
for i, (preset, col) in enumerate((("tiny", "#9ecae1"), ("mini", "#2171b5"))):
    vals = [get(preset, "3"), get(preset, "3", "3:0.25"), get(preset, "123", "3:0.25")]
    if not all(vals):
        continue
    y = [v["3D"] for v in vals]
    ax.bar(np.arange(3) + (i - 0.5) * w, y, w, color=col, label=preset)
    for x, c in zip(np.arange(3) + (i - 0.5) * w, y):
        ax.text(x, c + 0.008, f"{c:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(3), labels, fontsize=8)
ax.set_ylim(0.4, 0.72)
ax.set_ylabel("3D nRMSE (lower is better)")
ax.set_title("Scarce 3D data: the other dimensions help", fontsize=10)
ax.legend(frameon=False, fontsize=8)
os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
fig.savefig(fig_path, dpi=160)
json.dump(dict(note="per-dimension mean nRMSE, 36k steps; 1D without reaction-diffusion", runs=table),
          open(out_json, "w"), indent=1)
