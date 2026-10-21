"""Fused point-wise decoder MLP (Triton), forward and backward.

Per grid point:  y = w3 . gelu(W2 gelu(W1f f + w1s s + b1) + b2) + b3
where f is the w-channel decoded field and s the current field value. The hidden
activations (mlp wide) never leave registers: the forward writes only y, the
backward recomputes the forward for its block and writes only the input gradients
plus one partial weight-gradient slab per program (summed afterwards in PyTorch,
so results are deterministic).

field: (N, w, *S) channel-first, skip: (N, *S), output (N, *S).
"""
import math

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

_SQRT1_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


if HAVE_TRITON:
    @triton.jit
    def _gelu(x):
        return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    @triton.jit
    def _gelu_grad(x):
        cdf = 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
        pdf = tl.exp(-0.5 * x * x) * 0.3989422804014327
        return cdf + x * pdf

    @triton.jit
    def _dot(a, b, BF16: tl.constexpr):
        if BF16:
            return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), out_dtype=tl.float32)
        else:
            return tl.dot(a, b, input_precision="ieee", out_dtype=tl.float32)

    @triton.jit
    def _load_inputs(f_ptr, s_ptr, p, c, P, L, W: tl.constexpr):
        pm = p < P
        n = (p // L).to(tl.int64)          # 64-bit offsets: N * W * L can exceed 2^31 for large batches
        sp = p - n * L
        f_off = n[:, None] * (W * L) + c[None, :] * L + sp[:, None]
        F = tl.load(f_ptr + f_off, mask=pm[:, None] & (c[None, :] < W), other=0.0).to(tl.float32)
        s = tl.load(s_ptr + p, mask=pm, other=0.0).to(tl.float32)
        return F, s, pm, f_off

    @triton.jit
    def _fwd_core(F, s, W1f, w1s, b1, W2, b2, BF16: tl.constexpr):
        a1 = _dot(F, W1f, BF16) + s[:, None] * w1s[None, :] + b1[None, :]       # (BP, M)
        h1 = _gelu(a1)
        a2 = _dot(h1, W2, BF16) + b2[None, :]                                    # (BP, M)
        h2 = _gelu(a2)
        return a1, h1, a2, h2

    @triton.jit
    def _load_weights(w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, c, m, W: tl.constexpr, M: tl.constexpr):
        # W1 is (M, W+1) row-major (nn.Linear weight); load W1f^T as (WP, M), w1s as (M,)
        W1f = tl.load(w1_ptr + m[None, :] * (W + 1) + c[:, None], mask=c[:, None] < W, other=0.0).to(tl.float32)
        w1s = tl.load(w1_ptr + m * (W + 1) + W).to(tl.float32)
        b1 = tl.load(b1_ptr + m).to(tl.float32)
        # W2 is (M_out, M_in); a2 = h1 @ W2^T, so load W2^T[i, o] = W2[o, i]
        W2t = tl.load(w2_ptr + m[None, :] * M + m[:, None]).to(tl.float32)
        b2 = tl.load(b2_ptr + m).to(tl.float32)
        w3 = tl.load(w3_ptr + m).to(tl.float32)
        return W1f, w1s, b1, W2t, b2, w3

    @triton.jit
    def _mlp_fwd_kernel(f_ptr, s_ptr, y_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr,
                        P, L, W: tl.constexpr, WP: tl.constexpr, M: tl.constexpr,
                        BP: tl.constexpr, BF16: tl.constexpr):
        pid = tl.program_id(0)
        p = pid * BP + tl.arange(0, BP)
        c = tl.arange(0, WP)
        m = tl.arange(0, M)
        W1f, w1s, b1, W2t, b2, w3 = _load_weights(w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, c, m, W, M)
        F, s, pm, _ = _load_inputs(f_ptr, s_ptr, p, c, P, L, W)
        _, _, _, h2 = _fwd_core(F, s, W1f, w1s, b1, W2t, b2, BF16)
        y = tl.sum(h2 * w3[None, :], axis=1) + tl.load(b3_ptr).to(tl.float32)
        tl.store(y_ptr + p, y.to(y_ptr.dtype.element_ty), mask=pm)

    @triton.jit
    def _mlp_bwd_kernel(f_ptr, s_ptr, dy_ptr, df_ptr, ds_ptr,
                        w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr,
                        gW1_ptr, gb1_ptr, gW2_ptr, gb2_ptr, gw3_ptr, gb3_ptr,
                        P, L, n_blocks,
                        W: tl.constexpr, WP: tl.constexpr, M: tl.constexpr,
                        BP: tl.constexpr, BF16: tl.constexpr):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        c = tl.arange(0, WP)
        m = tl.arange(0, M)
        W1f, w1s, b1, W2t, b2, w3 = _load_weights(w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, c, m, W, M)
        acc_W1f = tl.zeros((M, WP), dtype=tl.float32)      # dW1 field part, (M, WP)
        acc_w1s = tl.zeros((M,), dtype=tl.float32)
        acc_b1 = tl.zeros((M,), dtype=tl.float32)
        acc_W2 = tl.zeros((M, M), dtype=tl.float32)         # dW2[o, i]
        acc_b2 = tl.zeros((M,), dtype=tl.float32)
        acc_w3 = tl.zeros((M,), dtype=tl.float32)
        acc_b3 = tl.zeros((BP,), dtype=tl.float32)
        for blk in range(pid, n_blocks, nprog):
            p = blk * BP + tl.arange(0, BP)
            F, s, pm, f_off = _load_inputs(f_ptr, s_ptr, p, c, P, L, W)
            dy = tl.load(dy_ptr + p, mask=pm, other=0.0).to(tl.float32)
            a1, h1, a2, h2 = _fwd_core(F, s, W1f, w1s, b1, W2t, b2, BF16)
            acc_w3 += tl.sum(h2 * dy[:, None], axis=0)
            acc_b3 += dy
            da2 = dy[:, None] * w3[None, :] * _gelu_grad(a2)                       # (BP, M)
            acc_W2 += _dot(tl.trans(da2), h1, BF16)                               # (M_out, M_in)
            acc_b2 += tl.sum(da2, axis=0)
            dh1 = _dot(da2, tl.trans(W2t), BF16)                                  # da2 @ W2 -> (BP, M_in)
            da1 = dh1 * _gelu_grad(a1)
            acc_W1f += _dot(tl.trans(da1), F, BF16)                               # (M, WP)
            acc_w1s += tl.sum(da1 * s[:, None], axis=0)
            acc_b1 += tl.sum(da1, axis=0)
            dF = _dot(da1, tl.trans(W1f), BF16)                                   # (BP, WP)
            tl.store(df_ptr + f_off, dF.to(df_ptr.dtype.element_ty), mask=pm[:, None] & (c[None, :] < W))
            ds = tl.sum(da1 * w1s[None, :], axis=1)
            tl.store(ds_ptr + p, ds.to(ds_ptr.dtype.element_ty), mask=pm)
        # one partial slab per program
        tl.store(gW1_ptr + pid * M * (W + 1) + m[:, None] * (W + 1) + c[None, :], acc_W1f, mask=c[None, :] < W)
        tl.store(gW1_ptr + pid * M * (W + 1) + m * (W + 1) + W, acc_w1s)
        tl.store(gb1_ptr + pid * M + m, acc_b1)
        tl.store(gW2_ptr + pid * M * M + m[:, None] * M + m[None, :], acc_W2)
        tl.store(gb2_ptr + pid * M + m, acc_b2)
        tl.store(gw3_ptr + pid * M + m, acc_w3)
        tl.store(gb3_ptr + pid, tl.sum(acc_b3, axis=0))


def _cfg(W, M):
    WP = max(16, triton.next_power_of_2(W))
    BP = 128 if M <= 32 else 64
    return WP, BP


_SM = {}


def _n_sm(device):
    if device not in _SM:
        _SM[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM[device]


class FusedPointMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, field, skip, w1, b1, w2, b2, w3, b3, bf16):
        N, W = field.shape[:2]
        L = skip[0].numel()
        M = w1.shape[0]
        P = N * L
        field = field.contiguous()
        skip = skip.contiguous()
        w1, b1, w2, b2, w3 = (t.contiguous() for t in (w1, b1, w2, b2, w3))
        out_dtype = torch.bfloat16 if bf16 else torch.float32
        y = torch.empty(skip.shape, device=skip.device, dtype=out_dtype)
        WP, BP = _cfg(W, M)
        grid = (triton.cdiv(P, BP),)
        _mlp_fwd_kernel[grid](field, skip, y, w1, b1, w2, b2, w3.view(-1), b3,
                              P, L, W=W, WP=WP, M=M, BP=BP, BF16=bf16, num_warps=4)
        ctx.save_for_backward(field, skip, w1, b1, w2, b2, w3, b3)
        ctx.bf16 = bf16
        return y

    @staticmethod
    def backward(ctx, dy):
        field, skip, w1, b1, w2, b2, w3, b3 = ctx.saved_tensors
        bf16 = ctx.bf16
        N, W = field.shape[:2]
        L = skip[0].numel()
        M = w1.shape[0]
        P = N * L
        WP, BP = _cfg(W, M)
        n_blocks = triton.cdiv(P, BP)
        nprog = min(n_blocks, 4 * _n_sm(field.device))
        dev = field.device
        f32 = dict(device=dev, dtype=torch.float32)
        gW1 = torch.empty(nprog, M, W + 1, **f32)
        gb1 = torch.empty(nprog, M, **f32)
        gW2 = torch.empty(nprog, M, M, **f32)
        gb2 = torch.empty(nprog, M, **f32)
        gw3 = torch.empty(nprog, M, **f32)
        gb3 = torch.empty(nprog, **f32)
        dF = torch.empty_like(field)
        dS = torch.empty_like(skip)
        _mlp_bwd_kernel[(nprog,)](field, skip, dy.contiguous(), dF, dS,
                                  w1, b1, w2, b2, w3.view(-1),
                                  gW1, gb1, gW2, gb2, gw3, gb3,
                                  P, L, n_blocks, W=W, WP=WP, M=M, BP=BP, BF16=bf16, num_warps=4)
        return (dF, dS, gW1.sum(0).to(w1.dtype), gb1.sum(0).to(b1.dtype), gW2.sum(0).to(w2.dtype),
                gb2.sum(0).to(b2.dtype), gw3.sum(0).view_as(w3).to(w3.dtype), gb3.sum().view_as(b3).to(b3.dtype), None)


def fused_point_mlp(field, skip, mlp, bf16):
    """mlp: nn.Sequential(Linear(w+1, M), GELU, Linear(M, M), GELU, Linear(M, 1))."""
    l1, l2, l3 = mlp[0], mlp[2], mlp[4]
    return FusedPointMLP.apply(field, skip, l1.weight, l1.bias, l2.weight, l2.bias, l3.weight, l3.bias, bf16)


def available(device, mlp=None):
    """CUDA + Triton, hidden width a power of two >= 16 (tl.arange / tl.dot constraint), field width <= 64."""
    if not (HAVE_TRITON and device.type == "cuda"):
        return False
    if mlp is not None:
        M, w = mlp[0].weight.shape[0], mlp[0].weight.shape[1] - 1
        return M >= 16 and (M & (M - 1)) == 0 and w <= 64
    return True
