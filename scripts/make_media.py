"""Animations for the README: forecasts against ground truth, and a step-by-step view of how the model works.

    python scripts/make_media.py --ckpt weights/omniarch-base.pt --out docs/media

forecast.gif      2D cases side by side; top row ground truth, bottom row OmniArch (ten given frames, then forecast)
how_it_works.gif  one case, one forecast step per frame: the input field, its Fourier spectrum with the kept modes,
                  the attention of the new tokens over the previous frames, and the predicted field against the truth
"""
import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.inference import OmniArchPredictor  # noqa: E402
from omniarch.model import _rope  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--examples", default=os.path.join(os.path.dirname(__file__), "..", "examples"))
ap.add_argument("--out", default="docs/media")
ap.add_argument("--device", default="auto")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
p = OmniArchPredictor(a.ckpt, a.device)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "font.family": "DejaVu Sans"})
K = 10                                              # given frames


def load(name):
    z = np.load(os.path.join(a.examples, f"{name}.npz"), allow_pickle=True)
    return z["data"].astype(np.float32), [str(c) for c in z["channels"]]


def to_gif(figs, path, ms, hold=6):
    """figs: list of RGB arrays; the last frame is held a little longer."""
    ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=255) for f in figs]
    durations = [ms] * (len(ims) - 1) + [ms * hold]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=durations, loop=0, optimize=True, disposal=2)
    print(path, f"{os.path.getsize(path) / 1e6:.2f} MB", flush=True)


def render(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


# ---------------------------------------------------------------- forecast.gif
CASES = [("2d_shallow_water", 0, "Shallow water: radial dam break", "Blues"),
         ("2d_diffusion_reaction", 0, "Diffusion-reaction: activator u", "RdBu_r")]
series = []
for name, ch, title, cmap in CASES:
    d, _ = load(name)
    pred = p.predict(d[:K], d.shape[0] - K)
    full = np.concatenate([d[:K], pred])
    err = np.linalg.norm(pred - d[K:]) / np.linalg.norm(d[K:])
    lo, hi = np.percentile(d[:, ch], [1, 99])
    series.append((d[:, ch], full[:, ch], title, cmap, lo, hi))
    print(f"{name}: nRMSE over the forecast {err:.3f}", flush=True)

T = series[0][0].shape[0]
frames = []
fig, axs = plt.subplots(2, len(series), figsize=(2.6 * len(series) + 0.4, 5.6), dpi=110)
fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.02, wspace=0.06, hspace=0.12)
for t in range(T):
    for j, (truth, model, title, cmap, lo, hi) in enumerate(series):
        for i, (field, row) in enumerate(((truth, "Ground truth"), (model, "OmniArch"))):
            ax = axs[i, j]
            ax.clear()
            ax.imshow(field[t].T, origin="lower", cmap=cmap, vmin=lo, vmax=hi, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(row, fontsize=9.5)
            if i == 1 and t >= K:
                for s in ax.spines.values():
                    s.set_edgecolor("#e6550d"); s.set_linewidth(2)
    phase = f"given frame {t + 1} of {K}" if t < K else f"forecast step {t - K + 1} of {T - K}"
    fig.suptitle(f"OmniArch base, one set of weights\n{phase}", fontsize=10,
                 color="#333333" if t < K else "#e6550d")
    frames.append(render(fig))
plt.close(fig)
to_gif(frames, os.path.join(a.out, "forecast.gif"), ms=320)

# ---------------------------------------------------------------- how_it_works.gif
name, ch = "2d_diffusion_reaction", 0
d, chans = load(name)
model, dev = p.model, p.device
m = model.config.modes[2]
blocks = model.blocks
captured = []


def hook(mod, args):
    captured.append((mod, args[0].detach(), args[1], args[2]))


hs = [b.register_forward_pre_hook(hook) for b in blocks]


@torch.no_grad()
def attention_over_time(C, T_ctx):
    """Attention of the newest time block (all fields) over all earlier tokens, averaged over heads and layers.
    Returns (T_ctx, C)."""
    acc = torch.zeros(T_ctx * C, device=dev)
    for blk, x, pos, mask in captured:
        B, L, D = x.shape
        q, k, _ = blk.qkv(blk.n1(x)).view(B, L, 3, blk.heads, D // blk.heads).permute(2, 0, 3, 1, 4)
        q, k = _rope(q, pos), _rope(k, pos)
        s = (q[:, :, -C:] @ k.transpose(-1, -2)) / math.sqrt(D // blk.heads)
        s = s.masked_fill(~mask[-C:], float("-inf"))
        acc += F.softmax(s.float(), -1).mean((0, 1, 2))
    return (acc / len(captured)).view(T_ctx, C).cpu().numpy()


hist = torch.from_numpy(d[:K]).to(dev)[None]
types = torch.arange(d.shape[1], device=dev)[None]
traj = hist
lo, hi = np.percentile(d[:, ch], [1, 99])
frames = []
fig = plt.figure(figsize=(11.0, 2.95), dpi=110)
gs = fig.add_gridspec(1, 5, left=0.015, right=0.99, top=0.74, bottom=0.08, wspace=0.22)
axs = [fig.add_subplot(gs[0, i]) for i in range(5)]
cm = "RdBu_r"
steps = d.shape[0] - K
for step in range(steps):
    captured.clear()
    inp = traj[:, -15:]
    with torch.no_grad():
        pn, (mean, std) = model(inp.float(), types)
    nxt = (pn[:, -1:] * std + mean)
    att = attention_over_time(inp.shape[2], inp.shape[1])
    traj = torch.cat([traj, nxt], 1)
    cur = inp[0, -1, ch].cpu().numpy()
    pred, truth = nxt[0, 0, ch].cpu().numpy(), d[K + step, ch]
    spec = np.fft.fftshift(np.abs(np.fft.fft2(cur - cur.mean())))
    for ax in axs:
        ax.clear(); ax.set_xticks([]); ax.set_yticks([])
    axs[0].imshow(cur.T, origin="lower", cmap=cm, vmin=lo, vmax=hi, interpolation="bilinear")
    axs[0].set_title("1  Current field u(t)")
    n = spec.shape[0]
    axs[1].imshow(np.log10(spec.T + 1e-6), origin="lower", cmap="viridis")
    axs[1].add_patch(Rectangle((n // 2 - m - 0.5, n // 2 - m - 0.5), 2 * m, 2 * m, fill=False, ec="#ffd92f", lw=1.6))
    axs[1].set_title("2  Fourier modes; kept ones boxed")
    axs[2].imshow(att.T, origin="lower", cmap="Oranges", aspect="auto", interpolation="nearest")
    axs[2].set_yticks(range(len(chans)), chans, fontsize=8)
    axs[2].set_xticks([0, att.shape[0] - 1], ["oldest", "newest"], fontsize=7.5)
    axs[2].set_title("3  Attention over past tokens")
    axs[3].imshow(pred.T, origin="lower", cmap=cm, vmin=lo, vmax=hi, interpolation="bilinear")
    axs[3].set_title("4  Predicted u(t+1)")
    axs[4].imshow(truth.T, origin="lower", cmap=cm, vmin=lo, vmax=hi, interpolation="bilinear")
    axs[4].set_title("Ground truth u(t+1)")
    fig.suptitle(f"Inside OmniArch, forecast step {step + 1} of {steps}: encode each field by its low Fourier modes, "
                 f"attend to the history, decode the next frame, repeat", fontsize=9.5, y=0.96)
    frames.append(render(fig))
plt.close(fig)
for h in hs:
    h.remove()
to_gif(frames, os.path.join(a.out, "how_it_works.gif"), ms=700, hold=3)
