"""gpu_augment must equal a plain per-sample numpy mirror + roll + sign flip."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.data import gpu_augment


def test_gpu_augment_matches_numpy():
    rng = np.random.default_rng(0)
    for S in [(16,), (8, 6), (4, 6, 5)]:
        B, T, C = 5, 3, 4
        x = rng.standard_normal((B, T, C) + S).astype(np.float32)
        shift = np.stack([rng.integers(0, n, B) for n in S], 1)
        flip = rng.random((B, len(S))) < 0.5
        sign = np.where(rng.random((B, C)) < 0.5, -1.0, 1.0).astype(np.float32)
        ref = x.copy()
        for b in range(B):
            for k in range(len(S)):
                y = np.flip(ref[b], axis=2 + k) if flip[b, k] else ref[b]
                ref[b] = np.roll(y, -shift[b, k], axis=2 + k)       # out[i] = mirrored[(i + s) % n]
            ref[b] *= sign[b].reshape((1, C) + (1,) * len(S))
        out = gpu_augment(torch.from_numpy(x), torch.from_numpy(shift), torch.from_numpy(flip),
                          torch.from_numpy(sign)).numpy()
        assert np.abs(out - ref).max() == 0, S


if __name__ == "__main__":
    test_gpu_augment_matches_numpy()
    print("gpu_augment ok")
