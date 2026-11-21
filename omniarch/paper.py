"""The OmniArch model of the ICML 2025 paper, with its original weights.

This module re-implements, in plain PyTorch, the architecture that produced the
results of the paper (Chen et al., ICML 2025), so that the original `OmniArch-L`
checkpoint (672 M parameters) can be loaded and evaluated without the original
training code, PyTorch Lightning or Hugging Face Transformers.

    from omniarch.paper import PaperPredictor
    p = PaperPredictor("weights/omniarch-large-icml2025.pt")
    nxt = p.next_frame(history)              # history: (K, C, *grid) -> (C, *grid)
    future = p.predict(history, steps=10)    # autoregressive rollout

Architecture (identical to the original code):
- every field of every frame is one token;
- each field is normalised by its own spatial mean and standard deviation;
- a Fourier encoder per spatial dimension lifts (value, coordinates) to 8 channels,
  pads the grid by one cell on each side, keeps 12 modes per axis and projects the
  spectrum to a 1024-dimensional token;
- a learned embedding of the field index is added, followed by LayerNorm;
- a 24-layer Llama-style Transformer (RMSNorm, SwiGLU, rotary positions over the
  token index) with block-causal attention: a token sees every field of its own
  and of all earlier frames;
- a Fourier decoder maps each output token back to the field of the next frame,
  which is then de-normalised.

The checkpoint is a dict with keys `config` and `model`; `scripts/convert_paper_ckpt.py`
produces it from the original PyTorch Lightning checkpoint, which can also be
loaded directly.
"""
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.fft import irfftn, rfftn


@dataclass
class PaperConfig:
    hidden: int = 1024
    layers: int = 24
    heads: int = 16
    intermediate: int = 4096
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_channels: int = 6
    width: int = 8          # channels of the Fourier encoder ("inter_size")
    modes: int = 12         # retained modes per axis
    padding: int = 1        # zero padding of the grid before the FFT
    dims: tuple = (1, 2, 3)
    residual: bool = False  # True: the decoder predicts the change from the input frame (residual option of the original training code)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "dims" in d:
            d["dims"] = tuple(d["dims"])
        return cls(**d)

    def to_dict(self):
        d = asdict(self)
        d["dims"] = list(d["dims"])
        return d


# ---------------------------------------------------------------- Fourier encoder / decoder

def _corner_slices(dim, modes):
    """Index of the retained spectral blocks, in the order of the original code.

    rfft keeps only non-negative frequencies on the last axis; on every other axis
    both the lowest `modes` and the highest `modes` (negative) frequencies are kept.
    """
    lo, hi = slice(None, modes), slice(-modes, None)
    if dim == 1:
        return [(lo,)]
    if dim == 2:
        return [(lo, lo), (hi, lo)]
    return [(lo, lo, lo), (hi, lo, lo), (lo, hi, lo), (hi, hi, lo)]


class FourierEncoder(nn.Module):
    def __init__(self, dim, width, modes, hidden, padding=1):
        super().__init__()
        self.dim, self.width, self.modes, self.padding = dim, width, modes, padding
        self.groups = 2 ** dim // 2
        self.fc0 = nn.Linear(1 + dim, width)
        shape = (width, width) + (modes,) * dim
        for i in range(self.groups):
            self.register_parameter(f"weights{i}_real", nn.Parameter(torch.zeros(shape)))
            self.register_parameter(f"weights{i}_imag", nn.Parameter(torch.zeros(shape)))
        self.head = nn.Linear(width * modes ** dim * 2 ** dim, hidden)

    def forward(self, x, grid):
        """x: (B, N, *S) normalised fields; grid: (B, *S, dim) coordinates -> (B, N, hidden)."""
        B, N = x.shape[:2]
        d = self.dim
        g = grid.unsqueeze(1).expand(B, N, *grid.shape[1:])
        h = torch.cat([x.unsqueeze(-1), g.to(x.dtype)], dim=-1).reshape(B * N, *x.shape[2:], 1 + d)
        h = self.fc0(h).permute(0, d + 1, *range(1, d + 1))          # (BN, width, *S)
        h = F.pad(h, [self.padding] * 2 * d)
        ft = rfftn(h.float(), dim=tuple(range(-d, 0)))
        sym = "xyz"[:d]
        op = f"bi{sym},io{sym}->bo{sym}"
        re, im = [], []
        for i, sl in enumerate(_corner_slices(d, self.modes)):
            blk = ft[(slice(None), slice(None)) + sl]
            # the original multiplies real and imaginary parts separately (not a complex product)
            re.append(torch.einsum(op, blk.real, getattr(self, f"weights{i}_real")).reshape(B * N, self.width, -1))
            im.append(torch.einsum(op, blk.imag, getattr(self, f"weights{i}_imag")).reshape(B * N, self.width, -1))
        feat = torch.cat([torch.cat(re, -1), torch.cat(im, -1)], -1)   # (BN, width, 2*groups*modes^d)
        return self.head(feat.reshape(B * N, -1)).reshape(B, N, -1)


class FourierDecoder(nn.Module):
    def __init__(self, dim, width, modes, hidden, padding=1):
        super().__init__()
        self.dim, self.width, self.modes, self.padding = dim, width, modes, padding
        self.groups = 2 ** dim // 2
        self.decode_head = nn.Linear(hidden, width * modes ** dim * self.groups * 2)
        self.fc1 = nn.Linear(width, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, tok, size):
        """tok: (B, N, hidden); size: spatial shape (tuple) -> (B, N, *size)."""
        B, N = tok.shape[:2]
        d, m, p = self.dim, self.modes, self.padding
        lat = self.decode_head(tok).float().reshape(B * N, self.width, *(m,) * d, self.groups, 2)
        coef = torch.complex(lat[..., 0], lat[..., 1])
        full = tuple(s + 2 * p for s in size)
        out = torch.zeros(B * N, self.width, *full[:-1], full[-1] // 2 + 1, dtype=torch.cfloat, device=tok.device)
        for i, sl in enumerate(_corner_slices(d, m)):
            out[(slice(None), slice(None)) + sl] = coef[..., i]
        h = irfftn(out, s=full, dim=tuple(range(-d, 0)))
        h = h[(slice(None), slice(None)) + tuple(slice(p, -p) for _ in range(d))].to(tok.dtype)
        h = h.permute(0, *range(2, 2 + d), 1)                           # width last
        h = self.fc2(F.gelu(self.fc1(h)))
        return h.reshape(B, N, *size)


# ---------------------------------------------------------------- Llama-style backbone

class RMSNorm(nn.Module):
    def __init__(self, n, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.hd = c.heads, c.hidden // c.heads
        self.q_proj = nn.Linear(c.hidden, c.hidden, bias=False)
        self.k_proj = nn.Linear(c.hidden, c.hidden, bias=False)
        self.v_proj = nn.Linear(c.hidden, c.hidden, bias=False)
        self.o_proj = nn.Linear(c.hidden, c.hidden, bias=False)

    def forward(self, x, cos, sin, mask):
        B, L, _ = x.shape
        q, k, v = (p(x).view(B, L, self.h, self.hd).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(o.transpose(1, 2).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden, c.intermediate, bias=False)
        self.up_proj = nn.Linear(c.hidden, c.intermediate, bias=False)
        self.down_proj = nn.Linear(c.intermediate, c.hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c.hidden, c.rms_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden, c.rms_eps)

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = RMSNorm(c.hidden, c.rms_eps)
        hd = c.hidden // c.heads
        self.register_buffer("inv_freq", 1.0 / (c.rope_theta ** (torch.arange(0, hd, 2).float() / hd)), persistent=False)

    def forward(self, x, mask, checkpoint=False):
        L = x.shape[1]
        pos = torch.arange(L, device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(x.device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        for blk in self.layers:
            if checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, mask, use_reentrant=False)
            else:
                x = blk(x, cos, sin, mask)
        return self.norm(x)


# ---------------------------------------------------------------- full model

class OmniArchPaper(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or PaperConfig()
        self.token_type_embedding = nn.Embedding(c.max_channels, c.hidden)
        self.model = Backbone(c)
        for d in c.dims:
            self.add_module(f"encoder_{d}", FourierEncoder(d, c.width, c.modes, c.hidden, c.padding))
            self.add_module(f"decoder_{d}", FourierDecoder(d, c.width, c.modes, c.hidden, c.padding))
        self.ln = nn.LayerNorm(c.hidden)

    @staticmethod
    def block_causal_mask(time_ids, valid=None):
        """Boolean mask (True = attend): a token of frame t sees the valid tokens of frames <= t.

        Padding tokens (valid = False) are hidden from every query; they attend to themselves
        only, which keeps their rows finite and does not affect valid tokens.
        """
        allowed = time_ids[:, None, :] <= time_ids[:, :, None]
        if valid is not None:
            allowed = allowed & valid[:, None, :]
        eye = torch.eye(time_ids.shape[1], dtype=torch.bool, device=time_ids.device)
        return (allowed | eye)[:, None]

    def forward_tokens(self, x, grid, time_ids, type_ids, valid=None, noise=0.0, checkpoint=False):
        """Token-level interface used for training with padded batches.

        x: (B, N, *S) one field per token; grid: (B, *S, dim); time_ids, type_ids: (B, N) integers;
        valid: (B, N) bool or None. Returns (B, N, *S): the prediction of the same field one frame later.
        """
        S = tuple(x.shape[2:])
        d = len(S)
        red = tuple(range(-d, 0))
        with torch.no_grad():
            std, mean = torch.std_mean(x, dim=red, keepdim=True, unbiased=False)
            std = std + 1e-7
        x = (x - mean) / std
        x_clean = x
        if noise:
            x = x + torch.randn_like(x) * noise
        h = getattr(self, f"encoder_{d}")(x, grid) + self.token_type_embedding(type_ids)
        h = self.ln(h)
        h = self.model(h, self.block_causal_mask(time_ids, valid), checkpoint=checkpoint)
        y = getattr(self, f"decoder_{d}")(h, S)
        if self.config.residual:
            y = y + x_clean.to(y.dtype)
        return y * std + mean

    def forward(self, frames, grid, noise=0.0):
        """frames: (B, T, C, *S) consecutive frames; grid: (B, *S, dim).

        Returns (B, T, C, *S): entry t is the prediction of frame t+1 from frames 0..t.
        """
        B, T, C = frames.shape[:3]
        S = tuple(frames.shape[3:])
        if C > self.config.max_channels:
            raise ValueError(f"at most {self.config.max_channels} fields per problem, got {C}")
        dev = frames.device
        time_ids = torch.arange(T, device=dev).repeat_interleave(C)[None].expand(B, -1)
        type_ids = torch.arange(C, device=dev).repeat(T)[None].expand(B, -1)
        y = self.forward_tokens(frames.reshape(B, T * C, *S), grid, time_ids, type_ids, noise=noise)
        return y.reshape(B, T, C, *S)


# ---------------------------------------------------------------- loading

def _state_from_lightning(path):
    """Model weights from the original PyTorch Lightning checkpoint (optimizer state is skipped)."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # very old torch, or extra objects in the checkpoint
        ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    return {k: v for k, v in sd.items() if not k.startswith("model.embed_tokens")}


def load_paper_model(path, device="cpu", dtype=torch.float32):
    """Load either the converted checkpoint or the original Lightning `.ckpt`."""
    obj = torch.load(path, map_location="cpu", weights_only=True) if path.endswith(".pt") else None
    if obj is not None and "model" in obj and "config" in obj:
        config, sd = PaperConfig.from_dict(obj["config"]), obj["model"]
    else:
        config, sd = PaperConfig(), _state_from_lightning(path)
    model = OmniArchPaper(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("inv_freq")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.to(device=device, dtype=dtype).eval()


def default_grid(shape):
    """Cell-centred coordinates on [0, 1]^d (PDEBench periodic boxes use this convention)."""
    axes = [(np.arange(n) + 0.5) / n for n in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).astype(np.float32)


class PaperPredictor:
    """Forecast with the paper checkpoint. Fields keep the order used in pre-training,
    e.g. (Vx, density, pressure) in 1D and (Vx, Vy, density, pressure) in 2D."""

    def __init__(self, ckpt, device=None):
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = load_paper_model(ckpt, self.device)

    @torch.no_grad()
    def next_frame(self, history, grid=None):
        """history: (K, C, *S) -> prediction of frame K, shape (C, *S)."""
        h = torch.as_tensor(np.asarray(history, np.float32), device=self.device)[None]
        g = default_grid(h.shape[3:]) if grid is None else np.asarray(grid, np.float32)
        g = torch.as_tensor(g, device=self.device)[None]
        return self.model(h, g)[0, -1].cpu().numpy()

    @torch.no_grad()
    def predict(self, history, steps, grid=None, context=None):
        """Autoregressive rollout of `steps` frames; `context` limits the frames given to the model."""
        frames = list(np.asarray(history, np.float32))
        out = []
        for _ in range(steps):
            ctx = frames if context is None else frames[-context:]
            nxt = self.next_frame(np.stack(ctx), grid)
            frames.append(nxt)
            out.append(nxt)
        return np.stack(out)
