"""Latent-token variant of OmniArch (2D pilot).

Same transformer and normalisation as omniarch.model, different tokenizer:

    field (H, W) --conv encoder (/16)--> latent grid (H/16, W/16) of tokens
    tokens (time x field x patch) --transformer, block-causal in time--> hidden states
    hidden states --conv decoder (x16), U-Net skips from the encoder--> increment to the next frame

Each field becomes (H/16)*(W/16) tokens instead of one token of truncated Fourier modes, so
fine structure no longer has to pass through a handful of low modes.
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MAX_CHANNELS, Block, OmniArch, RMSNorm


@dataclass
class LatentConfig:
    hidden: int = 256
    layers: int = 4
    heads: int = 4
    ffn: int = 704
    ch: tuple = (32, 64, 128, 128)       # conv widths at /2, /4, /8, /16
    n_types: int = MAX_CHANNELS
    noise: float = 0.01
    max_frames: int = 256

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["ch"] = tuple(d["ch"])
        return cls(**d)


LATENT_PRESETS = {
    "ltiny": LatentConfig(),
    "lmini": LatentConfig(hidden=512, layers=8, heads=8, ffn=1408, ch=(32, 64, 128, 256)),
}


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c), nn.Conv2d(c, c, 3, padding=1, padding_mode="circular")
        self.n2, self.c2 = nn.GroupNorm(8, c), nn.Conv2d(c, c, 3, padding=1, padding_mode="circular")

    def forward(self, x):
        return x + self.c2(F.gelu(self.n2(self.c1(F.gelu(self.n1(x))))))


class Encoder2D(nn.Module):
    """(N, 1, H, W) -> features at /2, /4, /8 (skips) and latent tokens at /16."""

    def __init__(self, ch, hidden):
        super().__init__()
        self.stem = nn.Conv2d(1, ch[0], 3, padding=1, padding_mode="circular")
        self.down = nn.ModuleList()
        cin = ch[0]
        for c in ch:
            self.down.append(nn.Sequential(nn.Conv2d(cin, c, 4, stride=2, padding=1, padding_mode="circular"), ResBlock(c)))
            cin = c
        self.to_tok = nn.Conv2d(ch[-1], hidden, 1)

    def forward(self, x):
        h = self.stem(x)
        skips = [h]
        for d in self.down:
            h = d(h)
            skips.append(h)
        return self.to_tok(h), skips[:-1]             # tokens (N, hidden, H/16, W/16); skips at /1 .. /8


class Decoder2D(nn.Module):
    def __init__(self, ch, hidden):
        super().__init__()
        self.from_tok = nn.Conv2d(hidden, ch[-1], 1)
        self.up = nn.ModuleList()
        outs = list(ch[:-1])[::-1] + [ch[0]]          # /8, /4, /2, /1 widths
        cin = ch[-1]
        for c, skip_c in zip(outs, list(ch[:-1])[::-1] + [ch[0]]):
            self.up.append(nn.ModuleDict(dict(
                up=nn.ConvTranspose2d(cin, c, 4, stride=2, padding=1),
                mix=nn.Conv2d(c + skip_c, c, 3, padding=1, padding_mode="circular"),
                res=ResBlock(c))))
            cin = c
        self.out = nn.Sequential(nn.Conv2d(ch[0] + 1, 64, 1), nn.GELU(), nn.Conv2d(64, 1, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, tok, skips, x):
        h = self.from_tok(tok)
        for blk, s in zip(self.up, skips[::-1]):
            h = blk["up"](h)
            h = blk["res"](blk["mix"](torch.cat([h, s], 1)))
        return self.out(torch.cat([h, x], 1))


class OmniArchLatent2D(nn.Module):
    """forward(x (B,T,C,H,W), types) -> (normalised next frames, (mean, std)), like OmniArch."""

    def __init__(self, config: LatentConfig):
        super().__init__()
        c = self.config = config
        self.enc = Encoder2D(c.ch, c.hidden)
        self.dec = Decoder2D(c.ch, c.hidden)
        self.type_emb = nn.Embedding(c.n_types, c.hidden)
        self.pos = nn.Sequential(nn.Linear(4, c.hidden), nn.GELU(), nn.Linear(c.hidden, c.hidden))
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = RMSNorm(c.hidden)
        for b in self.blocks:
            nn.init.zeros_(b.o.weight)
            nn.init.zeros_(b.down.weight)

    stats = staticmethod(OmniArch.stats)
    rollout = OmniArch.rollout

    def patch_pos(self, h, w, device):
        """Sine/cosine of the patch centre on the periodic unit square, fed through a small MLP."""
        ys = (torch.arange(h, device=device) + 0.5) / h
        xs = (torch.arange(w, device=device) + 0.5) / w
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        f = torch.stack([torch.sin(2 * torch.pi * yy), torch.cos(2 * torch.pi * yy),
                         torch.sin(2 * torch.pi * xx), torch.cos(2 * torch.pi * xx)], -1)
        return self.pos(f.view(h * w, 4))                         # (P, hidden)

    def forward(self, x, types=None, noise=None):
        B, T, C, H, W = x.shape
        assert H % 16 == 0 and W % 16 == 0, "grid must be a multiple of 16"
        if types is None:
            types = torch.arange(C, device=x.device).expand(B, C)
        mean, std = self.stats(x)
        xn = (x - mean) / std
        if noise:
            xn = xn + noise * torch.randn_like(xn)
        flat = xn.reshape(B * T * C, 1, H, W)
        tok, skips = self.enc(flat)                               # (BTC, D, h, w)
        D, h, w = tok.shape[1:]
        P = h * w
        tok = tok.flatten(2).transpose(1, 2).view(B, T, C, P, D)
        tok = tok + self.type_emb(types)[:, None, :, None] + self.patch_pos(h, w, x.device)[None, None, None]
        tok = tok.reshape(B, T * C * P, D)
        pos = torch.arange(T, device=x.device).repeat_interleave(C * P)
        mask = pos[None, :] <= pos[:, None]
        for blk in self.blocks:
            tok = blk(tok, pos, mask)
        hs = self.norm(tok).view(B * T * C, P, D).transpose(1, 2).reshape(B * T * C, D, h, w)
        delta = self.dec(hs.to(skips[0].dtype), skips, flat).view(B, T, C, H, W)
        return xn + delta.to(xn.dtype), (mean, std)
