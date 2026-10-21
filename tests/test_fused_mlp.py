"""Fused Triton decoder MLP: correctness against the PyTorch path, and speed.

    python tests/test_fused_mlp.py          # needs a CUDA GPU with Triton
"""
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from omniarch import fused_mlp  # noqa: E402
from omniarch.model import PRESETS, OmniArch  # noqa: E402
from test_fourier import ref_decode, ref_encode  # noqa: E402


def make_mlp(w, M, dev):
    mlp = nn.Sequential(nn.Linear(w + 1, M), nn.GELU(), nn.Linear(M, M), nn.GELU(), nn.Linear(M, 1)).to(dev)
    for p in mlp.parameters():
        p.data.normal_(0, 0.3)
    return mlp


def torch_mlp(field, skip, mlp):
    """Reference in fp32: concatenate [field channels, skip] per point and apply the Sequential."""
    z = torch.cat([field.movedim(1, -1), skip.unsqueeze(-1)], -1)
    return mlp(z).squeeze(-1)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def correctness(dev):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    rows, worst = [], {"fp32": 0.0, "bf16": 0.0}
    cases = [((256,), 16, 64), ((256,), 8, 32), ((64, 64), 16, 64), ((64, 64), 4, 32), ((48, 40), 8, 64),
             ((32, 32, 32), 8, 64), ((32, 32, 32), 4, 32), ((24, 20, 18), 16, 32)]
    for S, w, M in cases:
        mlp = make_mlp(w, M, dev)
        params = list(mlp.parameters())
        field = torch.randn(6, w, *S, device=dev, requires_grad=True)
        skip = torch.randn(6, *S, device=dev, requires_grad=True)
        y_ref = torch_mlp(field, skip, mlp)
        dy = torch.randn_like(y_ref)
        g_ref = torch.autograd.grad(y_ref, [field, skip] + params, dy)
        names = ["out", "d_field", "d_skip", "dW1", "db1", "dW2", "db2", "dW3", "db3"]
        # the existing PyTorch path under bf16 autocast, fed the same bf16 output gradient, as the fair bf16 yardstick
        with torch.autocast("cuda", torch.bfloat16):
            y_pt = torch_mlp(field, skip, mlp)
        g_pt = torch.autograd.grad(y_pt, [field, skip] + params, dy.to(y_pt.dtype))
        pt_errs = [rel(y_pt, y_ref)] + [rel(a, b) for a, b in zip(g_pt, g_ref)]
        rows.append((S, w, M, "pt16", dict(zip(names, pt_errs))))
        for mode, bf16 in (("fp32", False), ("bf16", True)):
            y = fused_mlp.fused_point_mlp(field, skip, mlp, bf16)
            g = torch.autograd.grad(y, [field, skip] + params, dy.to(y.dtype))
            errs = [rel(y, y_ref)] + [rel(a, b) for a, b in zip(g, g_ref)]
            if bf16:   # bf16 passes if within 2e-2, or no worse than 1.5x the existing PyTorch bf16 path
                worst[mode] = max(worst[mode], max(e if e < 2e-2 else e / max(1.5 * p, 1e-30) * 2e-2
                                                   for e, p in zip(errs, pt_errs)))
            else:
                worst[mode] = max(worst[mode], max(errs))
            rows.append((S, w, M, mode, dict(zip(names, errs))))
    print("correctness (relative L2 error vs. PyTorch fp32 reference)")
    print(f"  {'grid':14s} {'w':>3s} {'mlp':>4s} mode  " + " ".join(f"{n:>8s}" for n in rows[0][4]))
    for S, w, M, mode, e in rows:
        print(f"  {str(S):14s} {w:3d} {M:4d} {mode}  " + " ".join(f"{v:8.1e}" for v in e.values()))
    ok = worst["fp32"] < 1e-5 and worst["bf16"] < 2e-2
    print(f"  worst fp32 {worst['fp32']:.1e} (limit 1e-5); bf16 score {worst['bf16']:.1e} "
          f"(limit 2e-2, entries above 2e-2 scored against 1.5x the PyTorch bf16 error):", "OK" if ok else "FAIL")
    return ok


def set_impl(m, impl):
    for d in ("1", "2", "3"):
        e, dc = m.enc[d], m.dec[d]
        if impl == "fft":
            e.forward = lambda x, _e=e: ref_encode(_e, x)
            dc.forward = lambda h, s, _d=dc: ref_decode(_d, h, s)
        else:
            e.forward = type(e).forward.__get__(e)
            dc.forward = type(dc).forward.__get__(dc)
            dc.fused = impl == "fused"


SHAPES = {1: (16, 1, (256,)), 2: (16, 4, (64, 64)), 3: (12, 5, (32, 32, 32))}
BATCH = {"mini": (256, 64, 16), "tiny": (512, 128, 32)}


def timed(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps * 1000, torch.cuda.max_memory_allocated() / 1e9


def speed(dev, preset):
    torch.backends.cuda.matmul.allow_tf32 = True
    m = OmniArch(PRESETS[preset]).to(dev)
    for p in m.parameters():
        p.data.normal_(0, 0.02)
    batches = [torch.randn(b, T - 1, C, *S, device=dev) for b, (T, C, S) in zip(BATCH[preset], SHAPES.values())]
    full = {}
    for impl in ("fft", "matrix", "fused"):
        set_impl(m, impl)

        def step():
            for x in batches:
                with torch.autocast("cuda", torch.bfloat16):
                    pn, _ = m(x, noise=0.01)
                pn.float().square().mean().backward()
        full[impl] = timed(step)
    print(f"\n{preset}: one training step (3 batches fwd+bwd, bf16 autocast, no compile)")
    for impl, (ms, gb) in full.items():
        print(f"  {impl:7s} {ms:7.1f} ms  peak {gb:5.1f} GB  speed-up vs fft {full['fft'][0] / ms:4.2f}x, vs matrix {full['matrix'][0] / ms:4.2f}x")
    # decoder only, per dimension
    print(f"  decoder only fwd+bwd (ms / peak GB):")
    for (d, (T, C, S)), b in zip(SHAPES.items(), BATCH[preset]):
        dec = m.dec[str(d)]
        N = b * (T - 1) * C
        h = torch.randn(N, m.config.hidden, device=dev, requires_grad=True)
        skip = torch.randn(N, *S, device=dev)
        line = []
        for impl in ("fft", "matrix", "fused"):
            set_impl(m, impl)

            def fb():
                with torch.autocast("cuda", torch.bfloat16):
                    y = dec(h, skip)
                y.float().square().mean().backward()
            ms, gb = timed(fb)
            line.append(f"{impl} {ms:6.1f}/{gb:4.1f}")
        print(f"    {d}D N={N:5d} grid {str(S):13s} " + "   ".join(line))


if __name__ == "__main__":
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(), "torch", torch.__version__)
    ok = correctness(dev)
    if "--no-speed" not in sys.argv:
        for p in ("mini", "tiny"):
            speed(dev, p)
    sys.exit(0 if ok else 1)
