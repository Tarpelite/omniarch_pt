"""Raw PDEBench files as used by the ICML 2025 OmniArch paper.

Grids are those of pre-training: 1024 points in 1D, 128 x 128 in 2D (512 x 512 files subsampled by 4)
and 64^3 in 3D (128^3 files subsampled by 2). A trajectory keeps its first T+1 frames, T as in FAMILIES.
Each file is split 90/10 exactly as `torch.utils.data.random_split(..., generator=torch.Generator().manual_seed(42))`.
"""
import glob
import os

import h5py
import numpy as np
import torch

# family -> (file name patterns, spatial dimension, number of given frames T)
FAMILIES = {
    "1D CFD": (["1D_CFD_*_Train.hdf5"], 1, 80),
    "1D Advection": (["1D_Advection_Sols_beta*.hdf5"], 1, 200),
    "1D Burgers": (["1D_Burgers_Sols_Nu*.hdf5"], 1, 200),
    "1D Diffusion-sorption": (["1D_diff-sorp_NA_NA.h5"], 1, 100),
    "1D Reaction-diffusion": (["ReacDiff_Nu*_Rho*.hdf5"], 1, 100),
    "2D CFD": (["2D_CFD_*_Train.hdf5"], 2, 20),
    "2D Reaction-diffusion": (["2D_diff-react_NA_NA.h5"], 2, 100),
    "2D Shallow water": (["2D_rdb_NA_NA.h5"], 2, 100),
    "3D CFD": (["3D_CFD_*_Train.hdf5"], 3, 20),
    "3D Maxwell": (["Maxwell3D_pdearena_32.h5"], 3, 7),
}
FIELDS_CFD = {1: ["Vx", "density", "pressure"], 2: ["Vx", "Vy", "density", "pressure"],
              3: ["Vx", "Vy", "Vz", "density", "pressure"]}
TARGET = {1: 1024, 2: 128, 3: 64}


class Reader:
    """One trajectory (T+1, C, *S) and the grid (*S, dim) of a raw PDEBench file."""

    def __init__(self, path, dim, n_frames, native=False):
        """native: keep Maxwell at its stored 32^3 (packing); upsampling then happens when reading the pack."""
        self.path, self.dim, self.nf, self.native = path, dim, n_frames, native
        self.f = h5py.File(path, "r")
        name = os.path.basename(path)
        self.kind = ("maxwell" if name.startswith("Maxwell3D") else "cfd" if "CFD" in name
                     else "group" if name.endswith(".h5") else "tensor")
        if self.kind == "maxwell":
            # PDEArena Maxwell-3D (32^3, fields Dx Dy Dz Hx Hy Hz), upsampled to 64^3 as in the original code
            self.n = self.f["data"].shape[0]
            axes = [np.arange(-1, 1, 1 / 32) + 1 / 64] * 3
        elif self.kind == "group":
            self.keys = sorted(self.f.keys())
            self.n = len(self.keys)
            g = self.f[self.keys[0]]["grid"]
            axes = [g[a][()] for a in "xyz"[:dim]]
        elif self.kind == "tensor":
            self.n = self.f["tensor"].shape[0]
            axes = [self.f["x-coordinate"][()]]
        else:
            self.n = self.f["Vx"].shape[0]
            self.stride = self.f["Vx"].shape[-1] // TARGET[dim]
            axes = [self.f[f"{a}-coordinate"][()][::self.stride] for a in "xyz"[:dim]]
        mesh = np.meshgrid(*axes, indexing="ij")
        self.grid = np.stack(mesh, -1).astype(np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        nf = self.nf
        if self.kind == "maxwell":
            x = self.f["data"][i, :nf]
            if not self.native:
                x = upsample3d(x)
        elif self.kind == "tensor":
            x = self.f["tensor"][i, :nf][:, None]
        elif self.kind == "group":
            x = np.moveaxis(self.f[self.keys[i]]["data"][:nf], -1, 1)       # (T, *S, C) -> (T, C, *S)
        elif self.stride == 1:
            x = np.stack([self.f[k][i, :nf] for k in FIELDS_CFD[self.dim]], axis=1)
        else:   # read whole frames and subsample in memory (much faster than a strided HDF5 selection)
            sl = (slice(None, None, self.stride),) * self.dim
            x = np.stack([self.f[k][i, :nf][(slice(None),) + sl] for k in FIELDS_CFD[self.dim]], axis=1)
        return np.ascontiguousarray(x, dtype=np.float32)


    def read_block(self, idx):
        """Several trajectories at once; contiguous ranges are read in one request."""
        idx = list(idx)
        lo, hi = idx[0], idx[-1] + 1
        nf = self.nf
        if self.kind == "tensor" and hi - lo <= 2 * len(idx):
            x = self.f["tensor"][lo:hi, :nf][[i - lo for i in idx]][:, :, None]
        elif self.kind == "cfd" and self.stride == 1 and hi - lo <= 2 * len(idx):
            sel = [i - lo for i in idx]
            x = np.stack([self.f[k][lo:hi, :nf][sel] for k in FIELDS_CFD[self.dim]], axis=2)
        else:
            return [self[i] for i in idx]
        return list(np.ascontiguousarray(x, dtype=np.float32))


def upsample3d(x):
    """(..., C, X, Y, Z) -> trilinear to 64^3 (torch.nn.Upsample(size=64^3, mode="trilinear") as in the original)."""
    t = torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32))
    lead = t.shape[:-4]
    t = t.reshape(-1, *t.shape[-4:])
    t = torch.nn.functional.interpolate(t, size=(TARGET[3],) * 3, mode="trilinear")
    return t.reshape(*lead, *t.shape[1:]).numpy()


def split_indices(n, seed=42):
    """(train, test) indices of torch.utils.data.random_split(range(n), [int(0.9 n), n - int(0.9 n)])."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    k = int(0.9 * n)
    return sorted(perm[:k]), sorted(perm[k:])


def test_indices(n, seed=42):
    return split_indices(n, seed)[1]


def find_files(roots, pattern):
    """Files matching `pattern` under the roots; for equal file names the earlier root wins."""
    if isinstance(roots, str):
        roots = [roots]
    seen = {}
    for r in roots:
        for p in sorted(glob.glob(os.path.join(r, "**", pattern), recursive=True)):
            seen.setdefault(os.path.basename(p), p)
    return [seen[k] for k in sorted(seen)]


def family_files(roots, family):
    patterns, dim, T = FAMILIES[family]
    return [p for pat in patterns for p in find_files(roots, pat)], dim, T
