"""Read the arrays written by scripts/prepare_pdebench.py and serve training windows."""
import json
import os

import numpy as np
import torch


# datasets of one family share a sampling slot, so 16 reaction-diffusion files
# do not outweigh one shallow-water file
FAMILIES = ["adv", "burgers", "reacdiff", "cfd1d", "diffsorp", "swe2d", "diffreact2d", "cfd2d", "cfd3d"]


# Symmetries used for augmentation. Periodic problems may be shifted; reflections are
# valid when the velocity component along the flipped axis changes sign.
PERIODIC = ("adv", "burgers", "reacdiff", "cfd1d_rand_eta", "cfd2d", "cfd3d")
FLIPPABLE = PERIODIC + ("swe2d", "diffreact2d")
VELOCITY = {"Vx": 0, "Vy": 1, "Vz": 2}


def family(name):
    for f in FAMILIES:
        if name.startswith(f):
            return f
    return name


class Entry:
    def __init__(self, path):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        self.name = self.meta["name"]
        self.dim = self.meta["dim"]
        self.periodic = self.name.startswith(PERIODIC)
        self.flippable = self.name.startswith(FLIPPABLE)
        # channel that changes sign when axis k is mirrored (Burgers' u is a velocity)
        chans = self.meta["channels"]
        self.flip_sign = {VELOCITY[c]: i for i, c in enumerate(chans) if c in VELOCITY}
        if self.name.startswith("burgers"):
            self.flip_sign = {0: 0}
        self.types = np.arange(len(self.meta["channels"]))   # slot ids: order within a family is what matters
        self._arr = {}

    def array(self, split):
        if split not in self._arr:
            self._arr[split] = np.load(os.path.join(self.path, f"{split}.npy"), mmap_mode="r")
        return self._arr[split]


def load_entries(root, names=None):
    out = []
    for n in sorted(os.listdir(root)):
        p = os.path.join(root, n)
        if os.path.exists(os.path.join(p, "meta.json")) and (names is None or n in names):
            out.append(Entry(p))
    return out


class WindowBatches(torch.utils.data.IterableDataset):
    """Endless batches of random windows for one spatial dimension.

    Each batch comes from a single dataset (all samples share channels and grid).
    Families are drawn uniformly, then a file inside the family.
    """

    def __init__(self, entries, dim, batch, window, seed=0, shared_seed=None, augment=False, data_frac=1.0,
                 max_stride=1, min_window=6):
        """shared_seed: if set, every GPU draws the same dataset sequence (different samples),
        so all ranks do the same amount of work per step and nobody waits at the gradient sync."""
        self.entries = [e for e in entries if e.dim == dim]
        self.fams = sorted({family(e.name) for e in self.entries})
        self.batch, self.window, self.seed, self.shared_seed = batch, window, seed, shared_seed
        self.augment = augment
        self.data_frac = data_frac          # use only the first fraction of each dataset's training trajectories
        self.max_stride, self.min_window = max_stride, min_window   # random frame spacing per batch

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed * 1000 + wid)
        pick = rng if self.shared_seed is None else np.random.default_rng(self.shared_seed * 1000 + wid)
        while True:
            fam = self.fams[pick.integers(len(self.fams))]
            cands = [e for e in self.entries if family(e.name) == fam]
            e = cands[pick.integers(len(cands))]
            a = e.array("train")
            T = a.shape[1]
            top = max(1, min(self.max_stride, (T - 1) // (self.min_window - 1)))
            st = int(pick.integers(1, top + 1)) if top > 1 else 1
            w = min(self.window, (T - 1) // st + 1)
            span = (w - 1) * st + 1
            n = max(1, int(round(self.data_frac * a.shape[0])))
            idx = np.sort(rng.choice(n, self.batch, replace=n < self.batch))
            t0 = rng.integers(0, T - span + 1, self.batch)
            x = np.stack([a[i, s:s + span:st] for i, s in zip(idx, t0)])
            types = torch.from_numpy(np.broadcast_to(e.types, (self.batch, len(e.types))).copy())
            if not self.augment:
                yield torch.from_numpy(x), types, None, st
                continue
            # augmentation itself runs on the GPU (see gpu_augment); here only draw the random choices
            S = x.shape[3:]
            shift = np.stack([rng.integers(0, n, self.batch) if e.periodic else np.zeros(self.batch, int) for n in S], 1)
            flip = (rng.random((self.batch, e.dim)) < 0.5) & e.flippable
            sign = np.ones((self.batch, len(e.types)), np.float32)
            for k, c in e.flip_sign.items():
                sign[flip[:, k], c] = -1.0
            yield torch.from_numpy(x), types, (torch.from_numpy(shift), torch.from_numpy(flip), torch.from_numpy(sign)), st


def gpu_augment(x, shift, flip, sign):
    """Per-sample periodic shift and mirror image of a batch already on the GPU.

    x: (B, T, C, *S); shift: (B, d) ints; flip: (B, d) bools; sign: (B, C) +-1 for velocity
    components that change sign under the mirror. One gather per axis, no host copies.
    """
    B = x.shape[0]
    for k in range(x.ndim - 3):
        n = x.shape[3 + k]
        i = torch.arange(n, device=x.device)[None]                    # (1, n)
        j = (i + shift[:, k:k + 1].to(x.device)) % n                 # periodic shift ...
        src = torch.where(flip[:, k:k + 1].to(x.device), n - 1 - j, j)   # ... of the mirrored field
        view = [B] + [1] * (x.ndim - 1)
        view[3 + k] = n
        x = torch.take_along_dim(x, src.view(view).expand(*x.shape[:3 + k], n, *x.shape[4 + k:]), dim=3 + k)
    return x * sign.to(x.device).view(B, 1, -1, *([1] * (x.ndim - 3)))


def nrmse(pred, true):
    """PDEBench-style nRMSE: ||pred-true|| / ||true|| over time and space, per sample and channel, then averaged."""
    red = (1,) + tuple(range(3, pred.ndim))
    err = (pred - true).pow(2).sum(red).sqrt()
    ref = true.pow(2).sum(red).sqrt().clamp_min(1e-12)
    return (err / ref).mean().item()
