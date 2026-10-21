"""Markdown results table from results/<name>.json written by scripts/evaluate.py.

    python scripts/results_table.py results/mini.json results/tiny.json
"""
import json
import os
import sys

NAMES = {"adv": "Advection", "burgers": "Burgers", "cfd1d": "Compressible flow", "diffsorp": "Diffusion-sorption",
         "reacdiff": "Reaction-diffusion", "cfd2d": "Compressible NS", "diffreact2d": "Diffusion-reaction",
         "swe2d": "Shallow water", "cfd3d": "Compressible NS / turbulence"}
ORDER = ["adv", "burgers", "cfd1d", "diffsorp", "reacdiff", "cfd2d", "diffreact2d", "swe2d", "cfd3d"]

runs = [(os.path.splitext(os.path.basename(p))[0], json.load(open(p))["per_family"]) for p in sys.argv[1:]]
head = "| PDE family | Dim | " + " | ".join(f"OmniArch {n}" for n, _ in runs) + " | Copy last frame |"
print(head)
print("|" + "---|" * (head.count("|") - 1))
for f in ORDER:
    first = runs[0][1][f]
    cells = " | ".join(f"{r[f]['omniarch']:.3f}" for _, r in runs)
    print(f"| {NAMES[f]} | {first['dim']}D | {cells} | {first['copy_last']:.3f} |")
means = " | ".join(f"**{sum(r[f]['omniarch'] for f in ORDER) / len(ORDER):.3f}**" for _, r in runs)
base = sum(runs[0][1][f]["copy_last"] for f in ORDER) / len(ORDER)
print(f"| **Mean of families** | | {means} | **{base:.3f}** |")
