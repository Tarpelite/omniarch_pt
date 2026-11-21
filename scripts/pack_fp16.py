"""Pack the training split of PDEBench files into fp16 arrays for fast pre-training input.

Each token (one field of one frame) is stored normalised by its own spatial mean and standard deviation,
which are kept in float32; the normalised values are stored in float16. The model normalises every
token in the same way, so the input loses nothing but rounding, which is below 2 % of the errors
reported in the paper for every family (diffusion-sorption: 9 %, added in quadrature).

    python scripts/pack_fp16.py --pdebench ROOT [ROOT2] --out DIR [--dims 1] [--jobs 8]

Writes per file: NAME.data.npy (N, T+1, C, *S) float16, NAME.stats.npy (N, T+1, C, 2) float32 (mean, std),
NAME.grid.npy, NAME.index.npy (the original trajectory indices). Only the 90 % training split is written.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.pdebench import FAMILIES, Reader, family_files, split_indices  # noqa: E402


def pack(path, dim, T, out, chunk):
    name = os.path.basename(path).rsplit(".", 1)[0]
    dst = os.path.join(out, f"{dim}D", name)
    if os.path.exists(dst + ".index.npy"):
        return f"exists {name}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    t0 = time.time()
    rd = Reader(path, dim, T + 1, native=True)
    tr, _ = split_indices(len(rd))
    x0 = rd[tr[0]]
    data = np.lib.format.open_memmap(dst + ".data.tmp.npy", "w+", np.float16, (len(tr),) + x0.shape)
    stats = np.zeros((len(tr),) + x0.shape[:2] + (2,), np.float32)
    red = tuple(range(-dim, 0))
    for j in range(0, len(tr), chunk):
        xs = np.stack(rd.read_block(tr[j:j + chunk]))
        m = xs.mean(red, keepdims=True, dtype=np.float64)
        s = xs.std(red, keepdims=True, dtype=np.float64) + 1e-7
        data[j:j + len(xs)] = ((xs - m) / s).astype(np.float16)
        stats[j:j + len(xs), ..., 0] = m.reshape(xs.shape[:3])
        stats[j:j + len(xs), ..., 1] = s.reshape(xs.shape[:3])
    data.flush()
    del data
    np.save(dst + ".stats.npy", stats)
    np.save(dst + ".grid.npy", rd.grid)
    os.replace(dst + ".data.tmp.npy", dst + ".data.npy")
    np.save(dst + ".index.npy", np.array(tr, np.int64))          # written last: marks completion
    return f"done {name} {len(tr)} traj {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdebench", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dims", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    tasks = []
    for fam in FAMILIES:
        files, dim, T = family_files(a.pdebench, fam)
        if dim in a.dims:
            tasks += [(p, dim, T) for p in files]
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(pack, p, d, T, a.out, 64 if d == 1 else 8 if d == 2 else 1) for p, d, T in tasks]
        for f in futs:
            print(f.result(), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
