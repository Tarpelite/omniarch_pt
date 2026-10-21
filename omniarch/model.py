"""OmniArch-mini: a 1D/2D/3D unified autoregressive PDE foundation model.

Each physical field at each time step becomes one token:

    field (*space) --Fourier encoder--> token (hidden)
    tokens (time x channel) --Transformer, block-causal in time--> hidden states
    hidden state --Fourier decoder + pointwise MLP--> increment to the next frame

The encoder keeps a fixed number of low Fourier modes, so one model accepts any
grid size that holds those modes. Channels are told apart only by their slot in
the input (first, second, ...), as in the original OmniArch: the meaning of a slot
comes from the data, so fields need no names, only a consistent order within a
family of problems. Time steps are marked by rotary position encoding.
"""
from dataclasses import asdict, dataclass, field

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fused_mlp

# OMNIARCH_FUSED=1 uses the fused Triton kernel for the decoder MLP on CUDA (falls back to PyTorch elsewhere)
USE_FUSED = os.environ.get("OMNIARCH_FUSED", "0") == "1"

MAX_CHANNELS = 8


@dataclass
class OmniArchConfig:
    hidden: int = 512
    layers: int = 8
    heads: int = 8
    ffn: int = 1408
    width: dict = field(default_factory=lambda: {1: 16, 2: 16, 3: 8})
    modes: dict = field(default_factory=lambda: {1: 24, 2: 16, 3: 8})
    decoder_mlp: int = 64
    n_types: int = MAX_CHANNELS
    max_frames: int = 256
    noise: float = 0.01

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ("width", "modes"):
            d[k] = {int(i): v for i, v in d[k].items()}
        return cls(**d)

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "tiny": OmniArchConfig(hidden=256, layers=4, heads=4, ffn=704,
                           width={1: 8, 2: 8, 3: 4}, modes={1: 16, 2: 12, 3: 8}, decoder_mlp=32),
    "mini": OmniArchConfig(),
}


def _mode_index(sizes, m, device):
    """Indices of the kept modes along each axis of an rfftn spectrum."""
    idx = []
    for n in sizes[:-1]:
        assert n >= 2 * m, f"grid {n} too small for {m} modes"
        idx.append(torch.cat([torch.arange(m), torch.arange(n - m, n)]).to(device))
    last = sizes[-1] // 2 + 1
    assert last >= m, f"grid {sizes[-1]} too small for {m} modes"
    idx.append(torch.arange(m, device=device))
    return idx


def _n_modes(dim, m):
    return (2 * m) ** (dim - 1) * m


_DFT_CACHE = {}


def _dft_mats(S, m, device):
    """Small DFT matrices for the kept modes, matching rfftn/irfftn(norm="forward") with the
    mode order of _mode_index: non-last axes keep frequencies 0..m-1 and n-m..n-1, the last
    axis keeps 0..m-1. Truncated transforms then become a few small matrix products."""
    key = (tuple(S), m, str(device))
    if key not in _DFT_CACHE:
        fwd, inv = [], []
        for n in S[:-1]:
            k = torch.cat([torch.arange(m), torch.arange(n - m, n)]).double()
            ang = 2 * torch.pi * torch.outer(torch.arange(n).double(), k) / n          # (n, 2m)
            fwd.append(torch.polar(torch.ones_like(ang) / n, -ang).to(torch.complex64).to(device))
            inv.append(torch.polar(torch.ones_like(ang), ang).T.contiguous().to(torch.complex64).to(device))
        n = S[-1]
        k = torch.arange(m).double()
        ang = 2 * torch.pi * torch.outer(torch.arange(n).double(), k) / n              # (n, m)
        w = torch.full((m,), 2.0, dtype=torch.float64)
        w[0] = 1.0
        if n % 2 == 0 and m - 1 == n // 2:
            w[-1] = 1.0                                                                # Nyquist bin
        last = dict(cos=(ang.cos() / n).float().to(device), sin=(-ang.sin() / n).float().to(device),
                    icos=(ang.cos() * w).T.contiguous().float().to(device),
                    isin=(ang.sin() * w).T.contiguous().float().to(device))
        _DFT_CACHE[key] = (fwd, inv, last)
    return _DFT_CACHE[key]


def _apply(y, axis, mat):
    """Contract `axis` of y with mat (n_in, n_out)."""
    return torch.movedim(torch.movedim(y, axis, -1) @ mat, -1, axis)


def _tdft(x, dim, m):
    """x: (B, *S) float32, S with `dim` axes -> (B, *K) complex64 low modes (rfftn, norm="forward")."""
    S = x.shape[-dim:]
    fwd, _, last = _dft_mats(S, m, x.device)
    y = torch.complex(x @ last["cos"], x @ last["sin"])
    for j in range(dim - 1):
        y = _apply(y, x.ndim - dim + j, fwd[j])
    return y


def _itdft(y, S, m):
    """y: (B, *K) complex64 low modes -> (B, *S) float32 (irfftn(s=S, norm="forward") of the zero-padded spectrum)."""
    dim = len(S)
    _, inv, last = _dft_mats(S, m, y.device)
    for j in range(dim - 1):
        y = _apply(y, y.ndim - dim + j, inv[j])
    return y.real @ last["icos"] - y.imag @ last["isin"]


class FourierEncoder(nn.Module):
    """Field -> token. Mathematically: lift (value, coordinates) to `width` channels, rfftn, keep low modes, linear head.

    The lift is linear, so it is applied in mode space: only the scalar field is transformed,
    and the coordinate and bias terms use the (cached) transform of the grid itself.
    """

    def __init__(self, dim, width, modes, hidden):
        super().__init__()
        self.dim, self.width, self.modes = dim, width, modes
        self.lift = nn.Linear(1 + dim, width)
        self.head = nn.Linear(width * _n_modes(dim, modes) * 2, hidden)
        self.norm = nn.LayerNorm(hidden)
        self._grid_spec = {}

    def grid_spec(self, S, device):
        key = (tuple(S), str(device))
        if key not in self._grid_spec:
            axes = [torch.linspace(0, 1, n + 1, device=device)[:-1] for n in S]
            grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0)                   # (d, *S)
            fields = torch.cat([grid, torch.ones(1, *S, device=device)], 0)               # coordinates, then 1
            self._grid_spec[key] = _tdft(fields.float(), self.dim, self.modes)            # (d+1, *K)
        return self._grid_spec[key]

    def forward(self, x):                                   # x: (N, *S)
        S = x.shape[1:]
        _mode_index(S, self.modes, x.device)                # grid-size check
        with torch.autocast(x.device.type, enabled=False):
            X = _tdft(x.float(), self.dim, self.modes)                                    # (N, *K)
            W = self.lift.weight.float()                                                  # (w, 1+d)
            const = torch.cat([W[:, 1:], self.lift.bias.float()[:, None]], 1).to(X.dtype)  # (w, d+1)
            G = self.grid_spec(S, x.device)
            spec = W[:, 0].to(X.dtype).view(1, -1, *([1] * self.dim)) * X.unsqueeze(1) \
                + torch.tensordot(const, G, dims=([1], [0])).unsqueeze(0)               # (N, w, *K)
            z = torch.view_as_real(spec).flatten(1)
        return self.norm(self.head(z.to(self.head.weight.dtype)))


class FourierDecoder(nn.Module):
    """Token -> next-frame increment: linear head to low modes, inverse transform, point-wise MLP with the current field."""

    def __init__(self, dim, width, modes, hidden, mlp):
        super().__init__()
        self.dim, self.width, self.modes = dim, width, modes
        self.head = nn.Linear(hidden, width * _n_modes(dim, modes) * 2)
        self.mlp = nn.Sequential(nn.Linear(width + 1, mlp), nn.GELU(), nn.Linear(mlp, mlp), nn.GELU(),
                                 nn.Linear(mlp, 1))
        self.fused = None                                   # None: follow OMNIARCH_FUSED; True/False: override

    def forward(self, h, skip):                             # h: (N, hidden), skip: (N, *S)
        S = skip.shape[1:]
        m = self.modes
        _mode_index(S, m, h.device)                         # grid-size check
        coef = self.head(h).float().view(h.shape[0], self.width, *([2 * m] * (self.dim - 1)), m, 2)
        with torch.autocast(h.device.type, enabled=False):
            field_ = _itdft(torch.view_as_complex(coef.contiguous()), S, m)             # (N, w, *S)
        if (self.fused if self.fused is not None else USE_FUSED) and fused_mlp.available(h.device, self.mlp):
            bf16 = torch.is_autocast_enabled() and torch.get_autocast_dtype(h.device.type) == torch.bfloat16
            return fused_mlp.fused_point_mlp(field_, skip, self.mlp, bf16)
        # first MLP layer split into field and skip parts: no concatenated copy of the grid
        l1 = self.mlp[0]
        dt = l1.weight.dtype if not torch.is_autocast_enabled() else torch.get_autocast_dtype(h.device.type)
        z = torch.nn.functional.linear(field_.movedim(1, -1).to(dt), l1.weight[:, :self.width].to(dt), l1.bias.to(dt)) \
            + skip.unsqueeze(-1).to(dt) * l1.weight[:, self.width].to(dt)
        return self.mlp[1:](z).squeeze(-1)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _rope(x, pos, base=10000.0):                            # x: (B, heads, L, d)
    d = x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos.float()[:, None] * inv[None]                  # (L, d/2)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.n1, self.n2 = RMSNorm(c.hidden), RMSNorm(c.hidden)
        self.qkv = nn.Linear(c.hidden, 3 * c.hidden, bias=False)
        self.o = nn.Linear(c.hidden, c.hidden, bias=False)
        self.gate_up = nn.Linear(c.hidden, 2 * c.ffn, bias=False)
        self.down = nn.Linear(c.ffn, c.hidden, bias=False)

    def forward(self, x, pos, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = _rope(q, pos), _rope(k, pos)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, L, D))
        g, u = self.gate_up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(g) * u)


class OmniArch(nn.Module):
    def __init__(self, config: OmniArchConfig):
        super().__init__()
        c = self.config = config
        self.enc = nn.ModuleDict({str(d): FourierEncoder(d, c.width[d], c.modes[d], c.hidden) for d in (1, 2, 3)})
        self.dec = nn.ModuleDict({str(d): FourierDecoder(d, c.width[d], c.modes[d], c.hidden, c.decoder_mlp)
                                  for d in (1, 2, 3)})
        self.type_emb = nn.Embedding(c.n_types, c.hidden)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = RMSNorm(c.hidden)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.zeros_(b.o.weight)
            nn.init.zeros_(b.down.weight)
        for d in self.dec.values():
            nn.init.zeros_(d.mlp[-1].weight)
            nn.init.zeros_(d.mlp[-1].bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def stats(x):
        """Per-sample, per-channel mean and std over all given frames and grid points."""
        red = (1,) + tuple(range(3, x.ndim))
        mean = x.mean(red, keepdim=True)
        std = x.std(red, keepdim=True, unbiased=False).clamp_min(1e-4)
        return mean, std

    def forward(self, x, types=None, noise=None):
        """x: (B, T, C, *S) consecutive frames; types: (B, C) slot ids, default 0..C-1.

        Returns the normalized prediction of frames 1..T (B, T, C, *S) and (mean, std)
        so that prediction * std + mean is in physical units.
        """
        B, T, C = x.shape[:3]
        S = x.shape[3:]
        if types is None:
            types = torch.arange(C, device=x.device).expand(B, C)
        dim = len(S)
        mean, std = self.stats(x)
        xn = (x - mean) / std
        if noise:
            xn = xn + noise * torch.randn_like(xn)
        flat = xn.reshape(B * T * C, *S)
        tok = self.enc[str(dim)](flat).view(B, T, C, -1)
        tok = tok + self.type_emb(types)[:, None]
        tok = tok.view(B, T * C, -1)
        pos = torch.arange(T, device=x.device).repeat_interleave(C)
        mask = pos[None, :] <= pos[:, None]                 # block-causal: see every channel up to own time
        for blk in self.blocks:
            tok = blk(tok, pos, mask)
        h = self.norm(tok).reshape(B * T * C, -1)
        delta = self.dec[str(dim)](h, flat).view(B, T, C, *S)
        return xn + delta.to(xn.dtype), (mean, std)

    @torch.no_grad()
    def rollout(self, history, types=None, steps=1, context=15):
        """Autoregressive forecast. history: (B, K, C, *S) -> (B, steps, C, *S) in physical units."""
        traj = history
        out = []
        for _ in range(steps):
            inp = traj[:, -context:]
            pn, (mean, std) = self(inp, types)
            nxt = (pn[:, -1:] * std + mean).to(history.dtype)
            out.append(nxt)
            traj = torch.cat([traj, nxt], 1)
        return torch.cat(out, 1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
