"""Time each phase of the real training loop (scripts/train.py), one GPU, real data loaders.

    python scripts/profile_loop.py --data DIR [--preset mini] [--steps 200] [--fast]

Phases: data (next batch + copy + GPU augmentation), fwd/bwd per dimension, the zero-weight "touch" of the
other dimensions' parameters (needed only for multi-GPU DDP), gradient clipping, optimizer, EMA.
--fast applies the cheap fixes: no touch on one GPU, foreach EMA, fewer loader workers per dimension.
"""
import argparse
import copy
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.data import WindowBatches, gpu_augment, load_entries  # noqa: E402
from omniarch.model import PRESETS, OmniArch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--preset", default="mini")
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--fast", action="store_true")
ap.add_argument("--compile", action="store_true")
a = ap.parse_args()

dev = torch.device("cuda")
for k in ("recompile_limit", "cache_size_limit"):        # one compiled graph per (dimension, channel count)
    if hasattr(torch._dynamo.config, k):
        setattr(torch._dynamo.config, k, 64)
torch.backends.cuda.matmul.allow_tf32 = True
cfg = copy.deepcopy(PRESETS[a.preset])
model = OmniArch(cfg).to(dev)
fwd = torch.compile(model, dynamic=False) if a.compile else model
ema = copy.deepcopy(model).eval()
for p in ema.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)
dim_params = {d: list(model.enc[str(d)].parameters()) + list(model.dec[str(d)].parameters()) for d in (1, 2, 3)}
entries = load_entries(a.data)
batch, window = (256, 64, 16), (16, 16, 12)
loaders = {}
for d in (1, 2, 3):
    ds = WindowBatches(entries, d, batch[d - 1], window[d - 1], seed=d, shared_seed=7 + d, augment=True)
    loaders[d] = iter(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=a.workers, pin_memory=True,
                                                  persistent_workers=True, prefetch_factor=4))
ema_p, mod_p = list(ema.parameters()), list(model.parameters())

T = defaultdict(float)


def tick(name, t0):
    torch.cuda.synchronize()
    t = time.perf_counter()
    T[name] += t - t0
    return t


for step in range(a.steps + 20):
    if step == 20:
        T.clear()
        torch.cuda.synchronize()
        wall = time.perf_counter()
    t = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    for d in (1, 2, 3):
        item = next(loaders[d])
        x, types = item[0].to(dev, non_blocking=True), item[1].to(dev, non_blocking=True)
        if item[2] is not None:
            x = gpu_augment(x, *item[2])
        t = tick("data", t)
        with torch.autocast("cuda", torch.bfloat16):
            pn, (mean, std) = fwd(x[:, :-1], types, noise=cfg.noise)
        loss = torch.nn.functional.mse_loss(pn.float(), (x[:, 1:] - mean) / std)
        if not a.fast:
            touch = sum(p_.sum() for d_, ps in dim_params.items() if d_ != d for p_ in ps)
            loss = loss + 0.0 * touch
            t = tick("touch(fwd)", t)
        loss.backward()
        t = tick(f"fwd+bwd {d}D", t)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    t = tick("clip", t)
    opt.step()
    t = tick("optimizer", t)
    with torch.no_grad():
        if a.fast:
            torch._foreach_lerp_(ema_p, mod_p, 0.001)
        else:
            for pe, pm in zip(ema_p, mod_p):
                pe.lerp_(pm, 0.001)
    t = tick("ema", t)
torch.cuda.synchronize()
total = time.perf_counter() - wall
print(f"{a.preset} fast={a.fast} compile={a.compile} workers={a.workers}: {1000 * total / a.steps:.1f} ms/step "
      f"({a.steps / total:.1f} steps/s)")
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f"   {k:14s} {1000 * v / a.steps:7.1f} ms")
