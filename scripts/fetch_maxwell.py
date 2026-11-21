"""Build the 3D Maxwell test data of Table 1 from PDEArena Maxwell-3D (Hugging Face `pdearena/Maxwell-3D`).

    python scripts/fetch_maxwell.py --out /path/to/maxwell

Downloads the 192 files (train/valid/test, float64, 121 GB) one at a time, keeps float32 arrays of shape
(N, 8, 6, 32, 32, 32) with fields Dx Dy Dz Hx Hy Hz, deletes each raw file after conversion, and writes
`Maxwell3D_pdearena_32.h5` (60 GB, dataset "data"): all files concatenated in sorted file-name order, as in the
original code. omniarch.pdebench reads it as family "3D Maxwell" (upsampled to 64^3, 7 input frames).
Proxies are taken from the usual HTTPS_PROXY environment variable.
"""
import argparse
import json
import os
import subprocess
import urllib.request

import h5py
import numpy as np

REPO = "pdearena/Maxwell-3D"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (needs about 120 GB free while converting)")
    ap.add_argument("--limit-rate", default=None, help="curl --limit-rate, e.g. 20M")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "npy"), exist_ok=True)
    os.chdir(a.out)

    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main"))
    files = sorted(x["path"] for x in tree if x["path"].endswith(".h5"))
    print(len(files), "files", flush=True)
    for k, fn in enumerate(files):
        out = f"npy/{fn[:-3]}.npy"
        if os.path.exists(out):
            continue
        cmd = ["curl", "-sS", "-L", "--retry", "5", "-o", fn, f"https://huggingface.co/datasets/{REPO}/resolve/main/{fn}"]
        if a.limit_rate:
            cmd[1:1] = ["--limit-rate", a.limit_rate]
        subprocess.run(cmd, check=True)
        with h5py.File(fn, "r") as f:
            sp = list(f.keys())[0]
            d, h = f[sp]["d_field"][()], f[sp]["h_field"][()]
        x = np.concatenate([d, h], -1).transpose(0, 1, 5, 2, 3, 4).astype(np.float32)   # (N, T, 6, X, Y, Z)
        np.save(out + ".tmp.npy", x)
        os.replace(out + ".tmp.npy", out)
        os.remove(fn)
        print(k + 1, fn, x.shape, flush=True)

    n = sum(np.load(f"npy/{fn[:-3]}.npy", mmap_mode="r").shape[0] for fn in files)
    with h5py.File("Maxwell3D_pdearena_32.h5.tmp", "w") as w:
        ds = w.create_dataset("data", (n, 8, 6, 32, 32, 32), np.float32, chunks=(1, 8, 6, 32, 32, 32))
        w.attrs["files"] = json.dumps(files)
        w.attrs["channels"] = "Dx Dy Dz Hx Hy Hz"
        w.attrs["note"] = "PDEArena Maxwell-3D, all splits concatenated in sorted file order, float64 -> float32"
        i = 0
        for fn in files:
            x = np.load(f"npy/{fn[:-3]}.npy")
            ds[i:i + len(x)] = x
            i += len(x)
    os.replace("Maxwell3D_pdearena_32.h5.tmp", "Maxwell3D_pdearena_32.h5")
    print("done:", n, "trajectories in", os.path.join(a.out, "Maxwell3D_pdearena_32.h5"), flush=True)


if __name__ == "__main__":
    main()
