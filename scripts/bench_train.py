"""Training-step throughput on synthetic batches (no data loading): where is the ceiling?

    python scripts/bench_train.py mini            # eager, batches (256, 64, 16)
Prints ms per step for one dimension at a time and for the joint step, eager vs torch.compile,
and with the batch doubled.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.model import PRESETS, OmniArch  # noqa: E402

preset = sys.argv[1] if len(sys.argv) > 1 else "mini"
dev = "cuda"
SHAPES = {1: (16, 1, (256,)), 2: (16, 4, (64, 64)), 3: (12, 5, (32, 32, 32))}
BATCH = {1: 256, 2: 64, 3: 16}


def run(model, opt, dims, scale, n=30):
    xs = {d: torch.randn(BATCH[d] * scale, SHAPES[d][0], SHAPES[d][1], *SHAPES[d][2], device=dev) for d in dims}

    def step():
        opt.zero_grad(set_to_none=True)
        for d in dims:
            x = xs[d]
            with torch.autocast("cuda", torch.bfloat16):
                pn, (mu, sd) = model(x[:, :-1], noise=0.01)
            ((pn.float() - (x[:, 1:] - mu) / sd) ** 2).mean().backward()
        opt.step()
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


for mode in ("eager", "compile"):
    torch.manual_seed(0)
    m = OmniArch(PRESETS[preset]).to(dev)
    model = torch.compile(m, dynamic=False) if mode == "compile" else m
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    for dims in ((1,), (2,), (3,), (1, 2, 3)):
        for scale in (1, 2):
            ms = run(model, opt, dims, scale)
            print(f"{preset} {mode:7s} dims={''.join(map(str, dims)):3s} batch x{scale}: {ms:7.1f} ms/step  "
                  f"({1000 / ms:5.1f} steps/s)  peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
