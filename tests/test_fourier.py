"""The matrix-based encoder/decoder must match the original FFT implementation, and be faster.

    python tests/test_fourier.py            # equivalence (fp32) + timing of one training step's encoders/decoders
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.model import PRESETS, OmniArch, _mode_index  # noqa: E402


def ref_encode(enc, x):
    """Original implementation: lift on the full grid, rfftn, keep modes."""
    S = x.shape[1:]
    axes = [torch.linspace(0, 1, n + 1, device=x.device)[:-1] for n in S]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
    h = torch.cat([x.unsqueeze(-1), grid.expand(*x.shape, enc.dim)], -1)
    h = enc.lift(h).movedim(-1, 1).float()
    spec = torch.fft.rfftn(h, dim=tuple(range(2, 2 + enc.dim)), norm="forward")
    for ax, ix in enumerate(_mode_index(S, enc.modes, x.device)):
        spec = spec.index_select(2 + ax, ix)
    z = torch.view_as_real(spec).flatten(1)
    return enc.norm(enc.head(z.to(enc.head.weight.dtype)))


def ref_decode(dec, h, skip):
    """Original implementation: zero-padded spectrum, irfftn, concatenate, MLP."""
    S = skip.shape[1:]
    m = dec.modes
    coef = dec.head(h).float().view(h.shape[0], dec.width, *([2 * m] * (dec.dim - 1)), m, 2)
    coef = torch.view_as_complex(coef.contiguous())
    full = torch.zeros(h.shape[0], dec.width, *S[:-1], S[-1] // 2 + 1, dtype=coef.dtype, device=h.device)
    mesh = torch.meshgrid(*_mode_index(S, m, h.device), indexing="ij")
    full[(slice(None), slice(None)) + tuple(mesh)] = coef
    field_ = torch.fft.irfftn(full, s=S, dim=tuple(range(2, 2 + dec.dim)), norm="forward")
    z = torch.cat([field_.movedim(1, -1), skip.unsqueeze(-1).float()], -1)
    return dec.mlp(z.to(dec.mlp[0].weight.dtype)).squeeze(-1)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def equivalence(dev):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    m = OmniArch(PRESETS["mini"]).to(dev).float()
    for p in m.parameters():                          # the zero-initialised output layer would hide decoder errors
        p.data.normal_(0, 0.05)
    worst = 0.0
    for dim, S in [(1, (256,)), (1, (100,)), (2, (64, 64)), (2, (48, 40)), (3, (32, 32, 32)), (3, (24, 20, 18))]:
        x = torch.randn(6, *S, device=dev, requires_grad=True)
        enc, dec = m.enc[str(dim)], m.dec[str(dim)]
        e_new, e_ref = enc(x), ref_encode(enc, x)
        h = torch.randn(6, m.config.hidden, device=dev, requires_grad=True)
        d_new, d_ref = dec(h, x), ref_decode(dec, h, x)
        # gradients w.r.t. inputs and parameters
        g_new = torch.autograd.grad(e_new.square().sum() + d_new.square().sum(), [x, h] + list(enc.parameters()) + list(dec.parameters()), allow_unused=True)
        x2, h2 = x.detach().requires_grad_(), h.detach().requires_grad_()
        g_ref = torch.autograd.grad(ref_encode(enc, x2).square().sum() + ref_decode(dec, h2, x2).square().sum(),
                                    [x2, h2] + list(enc.parameters()) + list(dec.parameters()), allow_unused=True)
        gerr = max(rel(a, b) for a, b in zip(g_new, g_ref) if a is not None and b is not None and b.norm() > 0)
        errs = (rel(e_new, e_ref), rel(d_new, d_ref), gerr)
        worst = max(worst, *errs)
        print(f"  {dim}D grid {str(S):14s} encoder {errs[0]:.1e}  decoder {errs[1]:.1e}  gradients {errs[2]:.1e}")
    return worst


def timing(dev, preset):
    torch.backends.cuda.matmul.allow_tf32 = True
    m = OmniArch(PRESETS[preset]).to(dev)
    shapes = {1: (256, 16, 1, (256,)), 2: (64, 16, 4, (64, 64)), 3: (16, 12, 5, (32, 32, 32))}
    bs = {"mini": (256, 64, 16), "tiny": (512, 128, 32)}[preset]
    batches = [torch.randn(b, T - 1, C, *S, device=dev) for b, (_, T, C, S) in zip(bs, shapes.values())]
    res = {}
    for name, use_ref in (("original", True), ("new", False)):
        for d in ("1", "2", "3"):
            e, dc = m.enc[d], m.dec[d]
            e.forward = (lambda x, _e=e: ref_encode(_e, x)) if use_ref else type(e).forward.__get__(e)
            dc.forward = (lambda h, s, _d=dc: ref_decode(_d, h, s)) if use_ref else type(dc).forward.__get__(dc)

        def step():
            for x in batches:
                with torch.autocast("cuda", torch.bfloat16):
                    pn, _ = m(x, noise=0.01)
                pn.float().square().mean().backward()
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        for _ in range(10):
            step()
        torch.cuda.synchronize()
        res[name] = ((time.time() - t) / 10 * 1000, torch.cuda.max_memory_allocated() / 1e9)
        print(f"  {preset} {name:9s} {res[name][0]:6.1f} ms per training step, peak {res[name][1]:.1f} GB")
    print(f"  {preset} speed-up {res['original'][0] / res['new'][0]:.2f}x")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("equivalence (fp32, random weights; relative L2 error vs. the FFT implementation)")
    w = equivalence(dev)
    print(f"worst {w:.1e}", "OK" if w < 1e-4 else "MISMATCH")
    if dev == "cuda":
        print("timing (bf16 autocast, 3 batches fwd+bwd, no compile)")
        for p in ("mini", "tiny"):
            timing(dev, p)
