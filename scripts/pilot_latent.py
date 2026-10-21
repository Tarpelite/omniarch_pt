"""2D pilot: Fourier tokens (OmniArch) vs latent tokens (OmniArchLatent2D), same data, steps and evaluation.

    python scripts/pilot_latent.py --arch latent --preset ltiny --data DIR --out runs/pilot/ltiny
    python scripts/pilot_latent.py --arch fourier --preset tiny --data DIR --out runs/pilot/tiny

Single GPU. Trains on the 2D datasets only, then rolls out every 2D test trajectory from 10 given
frames to the end and writes results.json (per dataset, per family, and the copy-last-frame reference).
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.data import WindowBatches, family, gpu_augment, load_entries, nrmse  # noqa: E402
from omniarch.latent import LATENT_PRESETS, OmniArchLatent2D  # noqa: E402
from omniarch.model import PRESETS, OmniArch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arch", choices=["fourier", "latent"], required=True)
ap.add_argument("--preset", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--window", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

dev = torch.device("cuda")
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
if a.arch == "latent":
    cfg = copy.deepcopy(LATENT_PRESETS[a.preset])
    model = OmniArchLatent2D(cfg).to(dev)
    used = sum(p.numel() for p in model.parameters())
else:
    cfg = copy.deepcopy(PRESETS[a.preset])
    model = OmniArch(cfg).to(dev)
    # only the 2D encoder/decoder take part; count what the 2D task actually uses
    used = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("enc.1", "enc.3", "dec.1", "dec.3")))
ema = copy.deepcopy(model).eval()
for p in ema.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps)))))

entries = [e for e in load_entries(a.data) if e.dim == 2]
ds = WindowBatches(entries, 2, a.batch, a.window, seed=1, augment=True)
loader = iter(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=a.workers, pin_memory=True,
                                          persistent_workers=True, prefetch_factor=4))
os.makedirs(a.out, exist_ok=True)
json.dump(vars(a) | {"config": cfg.to_dict(), "params_used": used}, open(os.path.join(a.out, "args.json"), "w"), indent=1)
log = open(os.path.join(a.out, "log.jsonl"), "w")
print(f"{a.arch} {a.preset}: {used / 1e6:.2f}M parameters used for 2D", flush=True)

t0 = time.time()
for step in range(a.steps):
    item = next(loader)
    x, types = item[0].to(dev, non_blocking=True), item[1].to(dev, non_blocking=True)
    x = gpu_augment(x, *item[2])
    with torch.autocast("cuda", torch.bfloat16):
        pn, (mean, std) = model(x[:, :-1], types, noise=cfg.noise)
    loss = torch.nn.functional.mse_loss(pn.float(), (x[:, 1:] - mean) / std)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    with torch.no_grad():
        dk = min(0.999, (1 + step) / (10 + step))
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - dk)
    if step % 100 == 0 or step == a.steps - 1:
        rec = dict(step=step, loss=loss.item(), gn=gn.item(), sps=(step + 1) / (time.time() - t0),
                   mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 1000 == 0:
            print(rec, flush=True)
torch.save(dict(arch=a.arch, config=cfg.to_dict(), model=ema.state_dict()), os.path.join(a.out, "model.pt"))

# evaluation: all 2D test trajectories, 10 given frames, roll out to the end
res = {}
with torch.no_grad():
    for e in entries:
        arr = e.array("test")
        preds = []
        for i in range(0, arr.shape[0], 16):
            xb = torch.from_numpy(np.asarray(arr[i:i + 16])).to(dev)
            types = torch.as_tensor(e.types, device=dev).expand(xb.shape[0], -1)
            with torch.autocast("cuda", torch.bfloat16):
                preds.append(ema.rollout(xb[:, :10], types, xb.shape[1] - 10).float().cpu())
        true = torch.from_numpy(np.asarray(arr[:, 10:]))
        last = torch.from_numpy(np.asarray(arr[:, 9:10])).expand_as(true)
        res[e.name] = dict(family=family(e.name), model=nrmse(torch.cat(preds), true), copy_last=nrmse(last, true))
        print(f"{e.name:28s} {res[e.name]['model']:.4f}  (copy-last {res[e.name]['copy_last']:.4f})", flush=True)
fams = sorted({v["family"] for v in res.values()})
per_family = {f: dict(model=float(np.mean([v["model"] for v in res.values() if v["family"] == f])),
                      copy_last=float(np.mean([v["copy_last"] for v in res.values() if v["family"] == f]))) for f in fams}
summary = dict(arch=a.arch, preset=a.preset, params_used=used, steps=a.steps, train_hours=(time.time() - t0) / 3600,
               per_dataset=res, per_family=per_family,
               mean_family=float(np.mean([v["model"] for v in per_family.values()])))
json.dump(summary, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print("per family:", {k: round(v["model"], 4) for k, v in per_family.items()}, "mean", round(summary["mean_family"], 4))
