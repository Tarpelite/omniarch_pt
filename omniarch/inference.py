"""Load a trained checkpoint and forecast PDE trajectories.

    from omniarch import OmniArchPredictor
    p = OmniArchPredictor("omniarch-mini.pt")
    future = p.predict(history, steps=10)

`history` is a float array (K, C, *space) of K >= 1 consecutive frames. The C
fields need no names; keep their order the same as in training for that kind of
problem (e.g. density, Vx, Vy, pressure for compressible flow).
"""
import os
import time

import numpy as np
import torch

from .model import MAX_CHANNELS, OmniArch, OmniArchConfig

DEFAULT_CKPT_NAMES = ["omniarch-base.pt", "omniarch-mini.pt", "omniarch-tiny.pt"]


def find_checkpoint(path=None):
    """Explicit path > $OMNIARCH_CKPT > ./weights/ > ~/.cache/omniarch/."""
    cands = [path, os.environ.get("OMNIARCH_CKPT")]
    for d in ["weights", os.path.expanduser("~/.cache/omniarch")]:
        cands += [os.path.join(d, n) for n in DEFAULT_CKPT_NAMES]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError("no checkpoint found: pass --ckpt, set OMNIARCH_CKPT, or put omniarch-mini.pt in ./weights/")


def pick_device(device=None):
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class OmniArchPredictor:
    def __init__(self, ckpt=None, device=None, context=15):
        self.ckpt_path = find_checkpoint(ckpt)
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.config = OmniArchConfig.from_dict(state["config"])
        self.model = OmniArch(self.config)
        self.model.load_state_dict(state["model"])
        self.device = pick_device(device)
        self.model.to(self.device).eval()
        self.context = context
        self.info = {k: state[k] for k in ("step", "eval", "preset") if k in state}
        # bf16 on CUDA is fast and accurate enough; CPU/MPS stay in fp32
        self.amp = self.device.type == "cuda"

    @property
    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def check_grid(self, shape):
        dim = len(shape)
        if dim not in (1, 2, 3):
            raise ValueError(f"only 1D/2D/3D fields are supported, got {dim}D")
        m = self.config.modes[dim]
        for n in shape[:-1]:
            if n < 2 * m:
                raise ValueError(f"grid size {n} too small, need >= {2 * m} for {dim}D")
        if shape[-1] // 2 + 1 < m:
            raise ValueError(f"grid size {shape[-1]} too small, need >= {2 * m - 2} for {dim}D")

    def predict(self, history, steps, batched=False):
        """history: (K, C, *S), or (B, K, C, *S) with batched=True. Returns the same layout with `steps` frames."""
        h = np.asarray(history, dtype=np.float32)
        single = not batched
        if single:
            h = h[None]
        if h.ndim < 4:
            raise ValueError(f"history must be (frames, fields, *grid), got shape {h.shape[1:]}")
        if h.shape[2] > MAX_CHANNELS:
            raise ValueError(f"at most {MAX_CHANNELS} fields per problem, got {h.shape[2]}")
        self.check_grid(h.shape[3:])
        x = torch.from_numpy(h).to(self.device)
        types = torch.arange(h.shape[2], device=self.device).expand(x.shape[0], -1)
        t0 = time.time()
        with torch.autocast(self.device.type, torch.bfloat16, enabled=self.amp):
            out = self.model.rollout(x, types, steps, self.context)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_seconds = time.time() - t0
        out = out.float().cpu().numpy()
        return out[0] if single else out


def nrmse_np(pred, true):
    """nRMSE per channel over time and space, for one trajectory (T, C, *S)."""
    red = (0,) + tuple(range(2, pred.ndim))
    err = np.sqrt(((pred - true) ** 2).sum(red))
    ref = np.sqrt((true ** 2).sum(red)).clip(1e-12)
    return err / ref
