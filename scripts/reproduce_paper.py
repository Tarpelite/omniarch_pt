"""Reproduce the pre-training rows of Table 1 of the ICML 2025 paper with OmniArch-L weights.

    python scripts/reproduce_paper.py --pdebench /path/to/PDEBench /path/to/maxwell --ckpt weights/omniarch-large-pt.pt

Works with the PyTorch re-training (`omniarch-large-pt.pt`) and with the converted original weights
(`omniarch-large-icml2025.pt`); the model configuration (e.g. residual prediction) is read from the checkpoint.

Protocol of the original evaluation code:
- each PDEBench file is split 90/10 with `torch.utils.data.random_split(..., generator=torch.Generator().manual_seed(42))`;
  the 10 % part is the in-domain test set;
- grids are those of pre-training: 1024 points in 1D, 128 x 128 in 2D (512 x 512 files subsampled by 4),
  64^3 in 3D (128^3 files subsampled by 2); the trajectories keep their first T+1 frames (T below);
- given the true frames 0..T-1, the model predicts frame T in one step;
- nRMSE = ||pred - true||_2 / ||true||_2 over space, per field, averaged over fields and samples;
  a family is the mean over its files;
- the original validation code first applied a per-file "recover" map x*std+mean to prediction and target
  (scripts/table1_mean_std.json holds its statistics); the script reports nRMSE with and without it.

The 2D CFD column of Table 1 was computed on one of the eight 2D CFD files, Rand_M1.0_Eta0.1_Zeta0.1_128
(the file on which the original weights give 0.0135 against 0.0148 in the paper); the script reports that
file and the mean over all eight files.

3D Maxwell comes from PDEArena (scripts/fetch_maxwell.py builds Maxwell3D_pdearena_32.h5; pass its folder as
an extra --pdebench root). 2D incompressible Navier-Stokes (2.3 TB) is not evaluated.
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.paper import load_paper_model  # noqa: E402
from omniarch.pdebench import FAMILIES, Reader, family_files, test_indices  # noqa: E402

# Table 1 of the paper, "Full pre-training on 1D, 2D, 3D data", OmniArch-L
PAPER_L = {
    "1D CFD": 0.0250, "1D Advection": 0.0182, "1D Burgers": 0.0063, "1D Diffusion-sorption": 0.0015,
    "1D Reaction-diffusion": 0.0004, "2D CFD": 0.0148, "2D Reaction-diffusion": 0.0105, "2D Shallow water": 0.0014,
    "2D Incompressible NS": 0.1494, "3D CFD": 0.4531, "3D Maxwell": 0.2268,
}
# files of a family that Table 1 used as test set, when not all of them
TABLE1_FILES = {"2D CFD": ["2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train"]}

def nrmse(pred, true, dim):
    red = tuple(range(-dim, 0))
    err = ((pred - true) ** 2).mean(red) / (true ** 2).mean(red)
    return err.sqrt().mean(-1)          # mean over fields -> one value per sample


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdebench", required=True, nargs="+",
                    help="root directory of the PDEBench download (searched recursively); with several roots, "
                         "a file name found in an earlier root takes precedence")
    ap.add_argument("--ckpt", default="weights/omniarch-large-pt.pt",
                    help="converted checkpoint, or the original OA-L-RPO-nrmse-0.03878.ckpt")
    ap.add_argument("--out", default="results/paper_table1.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES), help="subset of families to evaluate")
    ap.add_argument("--max-samples", type=int, default=0, help="test trajectories per file (0 = all)")
    ap.add_argument("--batch", type=int, nargs=3, default=[32, 4, 1], metavar=("B1D", "B2D", "B3D"))
    ap.add_argument("--workers", type=int, default=4, help="data-loading worker processes")
    ap.add_argument("--stats", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "table1_mean_std.json"),
                    help="JSON {file: [mean, std]} of the original code; also report the nRMSE after its 'recover' "
                         "map x*std+mean ('none' to skip)")
    a = ap.parse_args()
    stats = json.load(open(a.stats)) if a.stats and a.stats != "none" else {}

    torch.set_float32_matmul_precision("high")          # as in the original evaluation (TF32 on NVIDIA GPUs)
    t0 = time.time()
    model = load_paper_model(a.ckpt, a.device)
    print(f"loaded {a.ckpt} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f} M parameters) "
          f"in {time.time() - t0:.0f} s on {a.device}", flush=True)

    per_file, per_family = {}, {}
    for fam in a.families:
        files, dim, T = family_files(a.pdebench, fam)
        if not files:
            print(f"{fam}: no files found under {a.pdebench}, skipped", flush=True)
            continue
        scores = []
        for path in files:
            rd = Reader(path, dim, T + 1)
            idx = test_indices(len(rd))
            if a.max_samples:
                idx = idx[:a.max_samples]
            grid = torch.from_numpy(rd.grid).to(a.device)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(_Lazy(path, dim, T + 1), idx),
                batch_size=a.batch[dim - 1], num_workers=a.workers)
            name = os.path.basename(path).rsplit(".", 1)[0]
            aff = None
            if name in stats:
                sh = (1, -1) + (1,) * dim
                aff = [torch.tensor(v, device=a.device, dtype=torch.float32).reshape(sh) for v in stats[name]]
            vals, copy, vals_aff, t1 = [], [], [], time.time()
            with torch.no_grad():
                for x in loader:
                    x = x.to(a.device)
                    g = grid[None].expand(x.shape[0], *grid.shape)
                    pred = model(x[:, :T], g)[:, -1].float()
                    vals.append(nrmse(pred, x[:, T], dim).cpu())
                    copy.append(nrmse(x[:, T - 1], x[:, T], dim).cpu())
                    if aff is not None:
                        m_, s_ = aff
                        vals_aff.append(nrmse(pred * s_ + m_, x[:, T] * s_ + m_, dim).cpu())
            v, c = torch.cat(vals), torch.cat(copy)
            rec = dict(family=fam, nrmse=float(v.mean()), copy_last=float(c.mean()), n=len(v),
                       seconds=round(time.time() - t1, 1))
            if vals_aff:
                rec["nrmse_affine"] = float(torch.cat(vals_aff).mean())
            per_file[name] = rec
            scores.append(rec)
            extra = f"  recovered {rec['nrmse_affine']:.4f}" if "nrmse_affine" in rec else ""
            print(f"  {name:58s} n={len(v):5d}  nRMSE {v.mean():.4f}  copy-last {c.mean():.4f}{extra}  "
                  f"({time.time() - t1:.0f} s)", flush=True)
        per_family[fam] = {k: float(np.mean([r[k] for r in scores])) for k in ("nrmse", "copy_last", "nrmse_affine")
                           if all(k in r for r in scores)}
        per_family[fam].update(files=len(scores), paper=PAPER_L[fam])
        if fam in TABLE1_FILES:
            sel = [per_file[n] for n in TABLE1_FILES[fam] if n in per_file]
            if sel:
                per_family[fam]["nrmse_all_files"] = per_family[fam]["nrmse"]
                per_family[fam]["copy_last_all_files"] = per_family[fam]["copy_last"]
                for k in ("nrmse", "copy_last", "nrmse_affine"):
                    if all(k in r for r in sel):
                        per_family[fam][k] = float(np.mean([r[k] for r in sel]))
                per_family[fam]["table1_files"] = TABLE1_FILES[fam]
        print(f"{fam:24s} {per_family[fam]['nrmse']:.4f}   copy-last {per_family[fam]['copy_last']:.4f}   "
              f"paper {PAPER_L[fam]:.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(per_family=per_family, per_file=per_file, ckpt=os.path.basename(a.ckpt),
                   max_samples=a.max_samples), open(a.out, "w"), indent=1)
    print("\n| PDE | This run | Copy last frame | Paper (Table 1) |" + (" Recovered |" if stats else "")
          + "\n|---|---|---|---|" + ("---|" if stats else ""))
    for fam, ref in PAPER_L.items():
        r = per_family.get(fam)
        got = f"{r['nrmse']:.4f} | {r['copy_last']:.4f}" if r else "not run | -"
        aff = (f" {r['nrmse_affine']:.4f} |" if r and "nrmse_affine" in r else " - |") if stats else ""
        print(f"| {fam} | {got} | {ref:.4f} |{aff}")
    for fam in TABLE1_FILES:
        r = per_family.get(fam)
        if r and "nrmse_all_files" in r:
            print(f"\n{fam}: Table 1 test file(s) {r['nrmse']:.4f}; mean over all {r['files']} files "
                  f"{r['nrmse_all_files']:.4f} (copy-last {r['copy_last_all_files']:.4f})")
    print(f"\nwritten to {a.out}; total {time.time() - t0:.0f} s")


class _Lazy(torch.utils.data.Dataset):
    """Opens the HDF5 file inside each worker process."""

    def __init__(self, path, dim, n_frames):
        self.args, self.rd = (path, dim, n_frames), None
        with h5py.File(path, "r"):
            pass

    def __len__(self):
        return len(Reader(*self.args)) if self.rd is None else len(self.rd)

    def __getitem__(self, i):
        if self.rd is None:
            self.rd = Reader(*self.args)
        return torch.from_numpy(self.rd[i])


if __name__ == "__main__":
    main()
