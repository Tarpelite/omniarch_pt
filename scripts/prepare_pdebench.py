"""Convert raw PDEBench HDF5 files into the compact arrays used by OmniArch-mini.

Every source file becomes one dataset entry with two arrays:
    <out>/<name>/train.npy, test.npy   float32, shape (N, T, C, *space)
    <out>/<name>/meta.json             channel names, strides, source file

Spatial grids are subsampled to 256 (1D), 64 (2D) and 32^3 (3D); time is
subsampled so every trajectory has 21 or 41 frames. Test samples come from
the tail of each file and never overlap with training samples.

Usage:
    python prepare_pdebench.py --root PDEBENCH_ROOT --out DIR --list
    python prepare_pdebench.py --root ... --out DIR --only adv_beta0.1
"""
import argparse
import json
import os
import time

import h5py
import numpy as np

# name -> (relative path, layout, channels, n_train, n_test, time stride, space stride)
# layout "tensor": key "tensor" (N, T, *S); "fields": one key per channel (N, T, *S);
# "groups": one group per sample with "data" (T, *S, C)
SPECS = {}


def add(name, path, layout, channels, n_train, n_test, ts, ss):
    SPECS[name] = dict(path=path, layout=layout, channels=channels, n_train=n_train,
                       n_test=n_test, ts=ts, ss=ss)


for b in ["0.1", "0.2", "0.4", "0.7", "1.0", "2.0", "4.0", "7.0"]:
    add(f"adv_beta{b}", f"1D/Advection/Train/1D_Advection_Sols_beta{b}.hdf5", "tensor", ["u"], 1000, 50, 5, 4)
for nu in ["0.001", "0.002", "0.004", "0.01", "0.02", "0.04", "0.1", "0.2", "0.4", "1.0", "2.0", "4.0"]:
    add(f"burgers_nu{nu}", f"1D/Burgers/Train/1D_Burgers_Sols_Nu{nu}.hdf5", "tensor", ["u"], 1000, 50, 5, 4)
for nu in ["0.5", "1.0", "2.0", "5.0"]:
    for rho in ["1.0", "2.0", "5.0", "10.0"]:
        add(f"reacdiff_nu{nu}_rho{rho}", f"1D/ReactionDiffusion/Train/ReacDiff_Nu{nu}_Rho{rho}.hdf5",
            "tensor", ["u"], 500, 25, 5, 4)
CFD1 = ["density", "Vx", "pressure"]
add("cfd1d_rand_eta0.01", "1D/CFD/Train/1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5", "fields", CFD1, 1000, 50, 5, 4)
add("cfd1d_rand_eta0.1", "1D/CFD/Train/1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5", "fields", CFD1, 1000, 50, 5, 4)
add("cfd1d_rand_eta1e-8", "1D/CFD/Train/1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5", "fields", CFD1, 1000, 50, 5, 4)
add("cfd1d_rand_trans", "1D/CFD/Train/1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5", "fields", CFD1, 1000, 50, 5, 4)
add("cfd1d_shock_trans", "1D/CFD/Train/1D_CFD_Shock_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5", "fields", CFD1, 1000, 50, 4, 4)
add("diffsorp", "1D/diffusion-sorption/1D_diff-sorp_NA_NA.h5", "groups", ["u"], 2000, 100, 5, 4)
add("swe2d", "2D/shallow-water/2D_rdb_NA_NA.h5", "groups", ["h"], 900, 100, 5, 2)
add("diffreact2d", "2D/diffusion-reaction/2D_diff-react_NA_NA.h5", "groups", ["u", "v"], 900, 100, 5, 2)
CFD2 = ["density", "Vx", "Vy", "pressure"]
for m in ["0.1", "1.0"]:
    for eta in ["0.01", "0.1"]:
        add(f"cfd2d_rand_m{m}_eta{eta}", f"2D/CFD/2D_Train_Rand/2D_CFD_Rand_M{m}_Eta{eta}_Zeta{eta}_periodic_128_Train.hdf5",
            "fields", CFD2, 2000, 100, 1, 2)
    add(f"cfd2d_rand_m{m}_eta1e-8", f"2D/CFD/2D_Train_Rand/2D_CFD_Rand_M{m}_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5",
        "fields", CFD2, 300, 30, 1, 8)
    add(f"cfd2d_turb_m{m}", f"2D/CFD/2D_Train_Turb/2D_CFD_Turb_M{m}_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5",
        "fields", CFD2, 300, 30, 1, 8)
CFD3 = ["density", "Vx", "Vy", "Vz", "pressure"]
add("cfd3d_rand", "3D/Train/3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5", "fields", CFD3, 90, 10, 1, 4)
add("cfd3d_turb", "3D/Train/3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5", "fields", CFD3, 540, 30, 1, 2)


def read_samples(h, spec, idx):
    """Return float32 array (len(idx), T, C, *S) after subsampling."""
    ts, ss, chans = spec["ts"], spec["ss"], spec["channels"]
    out = []
    for i in idx:
        if spec["layout"] == "tensor":
            a = h["tensor"][i]                      # (T, *S)
            a = a[::ts][:, None]
        elif spec["layout"] == "fields":
            a = np.stack([h[c][i] for c in chans], 1)  # (T, C, *S)
            a = a[::ts]
        else:
            a = h[f"{i:04d}"]["data"][()]           # (T, *S, C)
            a = np.moveaxis(a, -1, 1)[::ts]
        nd = a.ndim - 2
        sl = (slice(None), slice(None)) + (slice(None, None, ss),) * nd
        out.append(np.ascontiguousarray(a[sl], dtype=np.float32))
    return np.stack(out)


def n_samples(h, spec):
    if spec["layout"] == "tensor":
        return h["tensor"].shape[0]
    if spec["layout"] == "fields":
        return h[spec["channels"][0]].shape[0]
    return len(h.keys())


def prepare(root, out, name):
    spec = SPECS[name]
    dst = os.path.join(out, name)
    if os.path.exists(os.path.join(dst, "meta.json")):
        print(name, "exists, skip")
        return
    os.makedirs(dst, exist_ok=True)
    t0 = time.time()
    with h5py.File(os.path.join(root, spec["path"]), "r") as h:
        n = n_samples(h, spec)
        tr = list(range(spec["n_train"]))
        te = list(range(n - spec["n_test"], n))
        assert tr[-1] < te[0], (name, n)
        for split, idx in [("test", te), ("train", tr)]:
            mm = None
            step = 2 if spec["path"].startswith("3D") else (5 if spec["ss"] == 8 else 50)
            for k in range(0, len(idx), step):
                block = read_samples(h, spec, idx[k:k + step])
                if mm is None:
                    shape = (len(idx),) + block.shape[1:]
                    mm = np.lib.format.open_memmap(os.path.join(dst, f"{split}.tmp.npy"), "w+", np.float32, shape)
                mm[k:k + len(block)] = block
            mm.flush()
            assert np.isfinite(mm).all(), (name, split)
            os.replace(os.path.join(dst, f"{split}.tmp.npy"), os.path.join(dst, f"{split}.npy"))
            print(name, split, mm.shape, f"{time.time() - t0:.0f}s", flush=True)
    meta = dict(name=name, source=spec["path"], channels=spec["channels"], dim=len(mm.shape) - 3,
                frames=mm.shape[1], grid=list(mm.shape[3:]), time_stride=spec["ts"], space_stride=spec["ss"],
                n_train=spec["n_train"], n_test=spec["n_test"],
                train_index=[0, spec["n_train"]], test_index=[n - spec["n_test"], n])
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(SPECS))
    else:
        for name in a.only or SPECS:
            prepare(a.root, a.out, name)
