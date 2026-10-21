"""Multi-GPU pre-training of OmniArch-mini on the prepared PDEBench arrays.

    torchrun --nproc_per_node 4 scripts/train.py --data DIR --out runs/mini --preset mini

Every step draws one batch per spatial dimension (1D, 2D, 3D), so all encoders
and decoders get gradients each step and DDP needs no unused-parameter search.
"""
import argparse
import contextlib
import copy
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.data import WindowBatches, gpu_augment, load_entries, nrmse  # noqa: E402
from omniarch.model import PRESETS, OmniArch, count_params  # noqa: E402

# Optional site hook: OMNIARCH_METRICS_LIB points to a directory with metrics.py exposing
# log(name, value, step, split=...), e.g. to feed a cluster dashboard. Without it, metrics
# stay in <out>/log.jsonl only.
try:
    if not os.environ.get("OMNIARCH_METRICS_LIB"):
        raise ImportError
    sys.path.insert(0, os.environ["OMNIARCH_METRICS_LIB"])
    from metrics import log as metric  # noqa: E402
except ImportError:
    def metric(*args, **kwargs):
        pass


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="mini", choices=list(PRESETS))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--batch", type=int, nargs=3, default=[256, 64, 16], help="per-GPU batch for 1D 2D 3D")
    ap.add_argument("--window", type=int, nargs=3, default=[16, 16, 12])
    ap.add_argument("--loss_w", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--balanced", action="store_true", help="all GPUs draw the same dataset each step")
    ap.add_argument("--compile", action="store_true", help="torch.compile the transformer blocks")
    ap.add_argument("--augment", action="store_true", help="random periodic shifts and mirror images")
    ap.add_argument("--data_frac", type=float, default=1.0, help="fraction of training trajectories per dataset")
    return ap.parse_args()


@torch.no_grad()
def quick_eval(model, entries, device, n=16, history=10):
    model.eval()
    res = {}
    for e in entries:
        a = torch.from_numpy(e.array("test")[:n].copy()).to(device)
        types = torch.as_tensor(e.types, device=device).expand(a.shape[0], -1)
        chunk = 4 if e.dim == 3 else n
        preds = []
        for i in range(0, a.shape[0], chunk):
            with torch.autocast("cuda", torch.bfloat16):
                preds.append(model.rollout(a[i:i + chunk, :history], types[i:i + chunk], a.shape[1] - history).float())
        res[e.name] = round(nrmse(torch.cat(preds), a[:, history:]), 4)
    model.train()
    return res


def main():
    a = get_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.manual_seed(1234 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = copy.deepcopy(PRESETS[a.preset])
    if a.noise is not None:
        cfg.noise = a.noise
    model = OmniArch(cfg).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    if a.compile:
        for i, blk in enumerate(model.blocks):
            model.blocks[i] = torch.compile(blk, dynamic=True)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    dim_params = {d: list(model.enc[str(d)].parameters()) + list(model.dec[str(d)].parameters()) for d in (1, 2, 3)}

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "norm" in n_ or "type_emb" in n_ else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps)))))

    entries = load_entries(a.data)
    loaders = []
    for i, d in enumerate((1, 2, 3)):
        ds = WindowBatches(entries, d, a.batch[i], a.window[i], seed=rank * 10 + d,
                           shared_seed=(7 + d) if a.balanced else None, augment=a.augment, data_frac=a.data_frac)
        loaders.append(iter(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=a.workers,
                                                        pin_memory=True, persistent_workers=True,
                                                        prefetch_factor=4)))
    eval_entries = [e for e in entries if e.name in
                    ("adv_beta1.0", "burgers_nu0.01", "reacdiff_nu1.0_rho2.0", "cfd1d_rand_eta0.01", "diffsorp",
                     "swe2d", "diffreact2d", "cfd2d_rand_m0.1_eta0.01", "cfd2d_turb_m1.0", "cfd3d_rand", "cfd3d_turb")]

    start = 0
    os.makedirs(a.out, exist_ok=True)
    ck = os.path.join(a.out, "last.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"])
        opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); start = s["step"]
    if rank == 0:
        print(f"params {count_params(model) / 1e6:.2f}M  world {world}  start {start}", flush=True)
        json.dump(vars(a) | {"config": cfg.to_dict(), "params": count_params(model)},
                  open(os.path.join(a.out, "args.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "a") if rank == 0 else None

    t0 = time.time()
    for step in range(start, a.steps):
        losses = []
        opt.zero_grad(set_to_none=True)
        # one backward per dimension: peak memory is the largest of the three graphs, not their sum;
        # gradients accumulate locally and are all-reduced once, on the last backward
        for i in range(3):
            item = next(loaders[i])
            x, types = item[0].to(dev, non_blocking=True), item[1].to(dev, non_blocking=True)
            if len(item) == 3:
                x = gpu_augment(x, *item[2])
            ctx = ddp.no_sync() if i < 2 else contextlib.nullcontext()
            with ctx:
                with torch.autocast("cuda", torch.bfloat16):
                    pn, (mean, std) = ddp(x[:, :-1], types, noise=cfg.noise)
                target = (x[:, 1:] - mean) / std
                l = torch.nn.functional.mse_loss(pn.float(), target)
                # zero-weight touch of the other dimensions' encoder/decoder so DDP sees a gradient
                # for every parameter in every backward (they are not part of this graph otherwise)
                touch = sum(p_.sum() for d_, ps in dim_params.items() if d_ != i + 1 for p_ in ps)
                (a.loss_w[i] * l + 0.0 * touch).backward()
            losses.append(l.detach())
            del pn, mean, std, target, l, x
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            dk = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - dk)
        if rank == 0 and (step % 50 == 0 or step == a.steps - 1):
            ls = torch.stack(losses).tolist()
            rec = dict(step=step, l1d=ls[0], l2d=ls[1], l3d=ls[2], gn=gn.item(), lr=sched.get_last_lr()[0],
                       sps=(step + 1 - start) / (time.time() - t0))
            log.write(json.dumps(rec) + "\n"); log.flush()
            for n_, v in (("loss_1d", ls[0]), ("loss_2d", ls[1]), ("loss_3d", ls[2]), ("lr", rec["lr"])):
                metric(n_, v, step)
            if step % 500 == 0:
                rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
                print(rec, flush=True)
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            if rank == 0:
                res = quick_eval(ema, eval_entries, dev)
                res["mean"] = round(sum(res.values()) / len(res), 4)
                print("eval", step + 1, res, flush=True)
                log.write(json.dumps({"step": step + 1, "eval": res}) + "\n"); log.flush()
                for n_, v in res.items():
                    metric(f"nrmse_{n_}", v, step + 1, split="val")
                torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                                sched=sched.state_dict(), step=step + 1), ck + ".tmp")
                os.replace(ck + ".tmp", ck)
                torch.save(dict(config=cfg.to_dict(), model=ema.state_dict(), step=step + 1, eval=res),
                           os.path.join(a.out, "omniarch.pt"))
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
