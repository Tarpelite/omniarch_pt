"""Build the demo cases in examples/: in-distribution test trajectories plus zero-shot PDEBench Test problems.

    python scripts/make_examples.py --data PROCESSED_DIR --raw PDEBENCH_ROOT --out examples
"""
import argparse
import os

import h5py
import numpy as np

IN_DIST = [
    ("adv_beta1.0", "1d_advection", "一维平流方程 β=1"),
    ("burgers_nu0.01", "1d_burgers", "一维 Burgers 方程 ν=0.01"),
    ("diffsorp", "1d_diffusion_sorption", "一维扩散吸附"),
    ("cfd1d_rand_eta1e-8", "1d_compressible_flow", "一维可压缩流，随机初值"),
    ("swe2d", "2d_shallow_water", "二维浅水方程，溃坝"),
    ("diffreact2d", "2d_diffusion_reaction", "二维扩散反应（FitzHugh–Nagumo）"),
    ("cfd2d_rand_m1.0_eta0.01", "2d_compressible_ns", "二维可压缩 NS 方程，M=1"),
    ("cfd2d_turb_m1.0", "2d_turbulence", "二维可压缩湍流，M=1"),
    ("cfd3d_rand", "3d_compressible_ns", "三维可压缩 NS 方程，随机初值"),
]
IN_NOTE = "训练分布内的测试样本，这条轨迹训练时没见过。"

# (raw file, case id, title, time stride, space stride)
ZERO_SHOT = [
    ("1D/CFD/Test/ShockTube/Sod7.hdf5", "1d_sod_shock_tube", "一维 Sod 激波管", 10, 4),
    ("2D/CFD/Test/TOV/OTVortex.hdf5", "2d_orszag_tang_vortex", "二维 Orszag–Tang 涡", 5, 16),
]
ZS_NOTE = "零样本：训练中没有出现过的问题，网格和时间步长也和训练不同。"


def save(out, cid, data, chans, title, note, source):
    dtype = np.float16 if data.ndim == 5 else np.float32
    np.savez_compressed(os.path.join(out, cid + ".npz"), data=data.astype(dtype), channels=np.array(chans),
                        history=10, title=title, note=note, source=source)
    print(cid, data.shape, f"{os.path.getsize(os.path.join(out, cid + '.npz')) / 1e6:.1f}MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="examples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import json
    for name, cid, title in IN_DIST:
        meta = json.load(open(os.path.join(a.data, name, "meta.json")))
        arr = np.load(os.path.join(a.data, name, "test.npy"), mmap_mode="r")[0]
        save(a.out, cid, np.asarray(arr), meta["channels"], title, IN_NOTE, f"PDEBench {meta['source']}")
    for rel, cid, title, ts, ss in ZERO_SHOT:
        with h5py.File(os.path.join(a.raw, rel), "r") as h:
            chans = [c for c in ["density", "Vx", "Vy", "Vz", "pressure"] if c in h]
            nd = h[chans[0]].ndim - 1
            sl = (slice(None, None, ts),) + (slice(None, None, ss),) * nd
            arr = np.stack([h[c][sl] for c in chans], 1)
        save(a.out, cid, arr, chans, title, ZS_NOTE, f"PDEBench {rel}")


if __name__ == "__main__":
    main()
