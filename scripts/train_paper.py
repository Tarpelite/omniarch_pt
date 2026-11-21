"""Pre-train the paper model (OmniArch-L, ICML 2025) on PDEBench with the recipe of the original code.

    torchrun --nproc_per_node 4 scripts/train_paper.py --pdebench ROOT [ROOT2 ...] --out runs/paper_l --steps 100000

Recipe of the original training code (configs/OA-L.yaml), kept unless noted:
- every step takes one batch of each spatial dimension (gradients accumulated over the three) (per GPU: 42 trajectories in 1D, 3 in 2D, 1 in 3D);
  sequences of different length are padded and masked;
- teacher forcing over the whole trajectory: token k predicts the same field one frame later;
- loss: nRMSE per token over space, averaged over valid tokens, summed over the three dimensions;
- inputs normalised per token, Gaussian noise of 1e-4 added after normalisation;
- AdamW, learning rate 1e-4, 500 warm-up steps, gradient clipping at 0.5.
Changes: cosine decay to 1 % over the planned number of steps (the original schedule spanned 500 epochs
of which about 30 were run); bf16 autocast (Fourier transforms and normalisation stay in float32);
backbone weights initialised with standard deviation 0.02.
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.paper import OmniArchPaper, PaperConfig  # noqa: E402
from omniarch.pdebench import FAMILIES, Reader, family_files, split_indices, upsample3d  # noqa: E402


class Trajectories(torch.utils.data.Dataset):
    """All training (or test) trajectories of the given files; HDF5 files are opened inside each worker."""

    def __init__(self, files, part="train", limit=0, block=1):
        self.files = files                      # list of (path, dim, T)
        self.items = []                         # (file id, [indices]): `block` neighbouring trajectories
        self.n_traj = 0
        for fid, (path, dim, T) in enumerate(files):
            n = len(Reader(path, dim, T + 1))
            tr, te = split_indices(n)
            idx = tr if part == "train" else te
            if limit:
                idx = idx[:limit]
            self.n_traj += len(idx)
            self.items += [(fid, idx[j:j + block]) for j in range(0, len(idx) - block + 1, block)]
        self.readers = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        fid, idx = self.items[k]
        if fid not in self.readers:
            path, dim, T = self.files[fid]
            self.readers[fid] = Reader(path, dim, T + 1)
        rd = self.readers[fid]
        g = torch.from_numpy(rd.grid)
        return [(torch.from_numpy(x), g, fid) for x in rd.read_block(idx)]


class Packed(torch.utils.data.Dataset):
    """Training split packed by scripts/pack_fp16.py: fp16 normalised tokens plus float32 mean and std."""

    def __init__(self, prefixes, block=1):
        self.prefixes = prefixes
        self.items, self.n_traj, self.arrays = [], 0, {}
        for fid, pre in enumerate(prefixes):
            n = len(np.load(pre + ".index.npy"))
            self.n_traj += n
            self.items += [(fid, list(range(j, j + block))) for j in range(0, n - block + 1, block)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        fid, idx = self.items[k]
        if fid not in self.arrays:
            pre = self.prefixes[fid]
            self.arrays[fid] = (np.load(pre + ".data.npy", mmap_mode="r"), np.load(pre + ".stats.npy", mmap_mode="r"),
                                torch.from_numpy(np.load(pre + ".grid.npy")))
        data, stats, g = self.arrays[fid]
        lo, hi = idx[0], idx[-1] + 1
        x = torch.from_numpy(np.asarray(data[lo:hi], dtype=np.float32))
        st = torch.from_numpy(np.array(stats[lo:hi]))
        sh = st.shape[:3] + (1,) * (x.dim() - 3)
        x = x * st[..., 1].reshape(sh) + st[..., 0].reshape(sh)
        if x.dim() == 6 and x.shape[-1] != g.shape[-2]:        # Maxwell packed at 32^3
            x = torch.from_numpy(upsample3d(x.numpy()))
        return [(xi, g, fid) for xi in x]


def packed_prefixes(dirs, dim):
    out = {}
    for d in dirs:
        sub = os.path.join(d, f"{dim}D")
        if os.path.isdir(sub):
            for f in sorted(os.listdir(sub)):
                if f.endswith(".index.npy"):
                    out.setdefault(f[:-len(".index.npy")], os.path.join(sub, f[:-len(".index.npy")]))
    return [out[k] for k in sorted(out)]


class Collate:
    """Picklable collate with optional time-step augmentation: each trajectory keeps every s-th frame from a
    random offset, s uniform in 1..max_stride (a new task with a larger time step)."""

    def __init__(self, max_stride=1):
        self.max_stride = max_stride

    def __call__(self, batch):
        if self.max_stride > 1:
            out = []
            for blk in batch:
                nb = []
                for x, g, fid in blk:
                    s = int(torch.randint(1, self.max_stride + 1, ()))
                    o = int(torch.randint(0, s, ()))
                    nb.append((x[o::s], g, fid))
                out.append(nb)
            batch = out
        return collate(batch)


def collate(batch):
    """-> inputs (B, N, *S), labels (B, N, *S), time ids, type ids, valid mask, grid, file ids."""
    batch = [t for blk in batch for t in blk]
    S = tuple(batch[0][0].shape[2:])
    seqs = []
    for x, g, fid in batch:
        T, C = x.shape[0] - 1, x.shape[1]
        seqs.append((x[:T].reshape(T * C, *S), x[1:].reshape(T * C, *S),
                     torch.arange(T).repeat_interleave(C), torch.arange(C).repeat(T)))
    N = max(s[0].shape[0] for s in seqs)
    B = len(batch)
    inp = torch.zeros(B, N, *S)
    lab = torch.zeros(B, N, *S)
    tid = torch.full((B, N), 10 ** 6, dtype=torch.long)
    typ = torch.zeros(B, N, dtype=torch.long)
    valid = torch.zeros(B, N, dtype=torch.bool)
    for b, (i, l, t, c) in enumerate(seqs):
        n = i.shape[0]
        inp[b, :n], lab[b, :n], tid[b, :n], typ[b, :n], valid[b, :n] = i, l, t, c, True
    grid = torch.stack([g for _, g, _ in batch])
    return inp, lab, tid, typ, valid, grid, torch.tensor([f for _, _, f in batch])


def nrmse_loss(pred, lab, valid, dim):
    red = tuple(range(-dim, 0))
    per_tok = ((pred - lab) ** 2).mean(red) / ((lab ** 2).mean(red) + 1e-8) + 1e-8
    return per_tok[valid].sqrt().mean()


def allreduce_grads(params, world, bucket_elems=50_000_000):
    """Average gradients over ranks after the three backward passes of a step."""
    bucket, size = [], 0

    def flush():
        flat = torch.cat([g.reshape(-1) for g in bucket])
        dist.all_reduce(flat)
        flat /= world
        off = 0
        for g in bucket:
            g.copy_(flat[off:off + g.numel()].view_as(g))
            off += g.numel()

    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        bucket.append(p.grad)
        size += p.grad.numel()
        if size >= bucket_elems:
            flush()
            bucket, size = [], 0
    if bucket:
        flush()


class TokenForward(torch.nn.Module):
    """Lets DDP wrap `forward_tokens`."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, *args, **kw):
        return self.m.forward_tokens(*args, **kw)


def init_weights(model, std=0.02):
    for name, mod in model.model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.normal_(mod.weight, std=std)
    for d in model.config.dims:
        enc = getattr(model, f"encoder_{d}")
        for n, p in enc.named_parameters():
            if n.startswith("weights"):
                with torch.no_grad():
                    p.uniform_(0, 1.0 / (enc.width * enc.width))


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    prog = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


class FamilySampler(torch.utils.data.Sampler):
    """Draws items with replacement so that family f gets a share proportional to n_f ** alpha
    (n_f: its items). alpha = 1 is the plain mix; alpha = 0 gives every family the same share.
    Each rank draws len(dataset) / world items per epoch from its own random stream."""

    def __init__(self, item_family, alpha, rank=0, world=1, seed=0, keep=None):
        names, inv, counts = np.unique(np.asarray(item_family), return_inverse=True, return_counts=True)
        share = counts.astype(np.float64) ** alpha
        if keep:
            share *= np.array([n in keep for n in names], dtype=np.float64)
        share /= share.sum()
        self.w = torch.from_numpy(share[inv] / counts[inv])
        self.n = len(inv) // world
        self.rank, self.seed, self.epoch = rank, seed, 0
        self.shares = dict(zip(names.tolist(), share.tolist()))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + 1000 * self.epoch + self.rank)
        return iter(torch.multinomial(self.w, self.n, replacement=True, generator=g).tolist())


def loop(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for b in loader:
            yield b
        epoch += 1


@torch.no_grad()
def quick_eval(model, test_sets, device):
    """Last-frame nRMSE on a fixed subset of the test split, per family."""
    model.eval()
    out = {}
    for fam, (ds, T) in test_sets.items():
        vals = []
        for k in range(len(ds)):
            x, g, _ = ds[k][0]
            x, g = x[None].to(device), g[None].to(device)
            with torch.autocast("cuda", torch.bfloat16):
                pred = model(x[:, :T], g)[:, -1].float()
            d = x.dim() - 3
            red = tuple(range(-d, 0))
            vals.append((((pred - x[:, T]) ** 2).mean(red) / (x[:, T] ** 2).mean(red)).sqrt().mean().item())
        out[fam] = float(np.mean(vals))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdebench", nargs="+", required=True, help="raw PDEBench roots (test split, and training without --packed)")
    ap.add_argument("--packed", nargs="*", default=None, help="directories written by scripts/pack_fp16.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, nargs=3, default=[42, 3, 1])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--checkpoint", type=int, nargs=3, default=[1, 0, 0], metavar=("C1D", "C2D", "C3D"),
                    help="activation checkpointing in the backbone, per dimension (1 = on)")
    ap.add_argument("--workers", type=int, nargs=3, default=[8, 6, 3])
    ap.add_argument("--block", type=int, nargs=3, default=[6, 1, 1],
                    help="neighbouring trajectories read together (the batch is made of shuffled blocks)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-samples", type=int, default=4, help="test trajectories per file in quick evaluation")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--max-steps-this-run", type=int, default=0, help="stop early (for smoke tests)")
    ap.add_argument("--max-stride", type=int, nargs=3, default=[1, 1, 1], metavar=("S1D", "S2D", "S3D"),
                    help="time-step augmentation: keep every s-th frame, s uniform in 1..S (1 = off)")
    ap.add_argument("--only", nargs="+", default=None, metavar="FAMILY",
                    help="fine-tune on these families only (other dimensions are not trained)")
    ap.add_argument("--loss-residual", action="store_true",
                    help="nRMSE on the change from the input frame, nRMSE(pred - x, label - x), as in the original code")
    ap.add_argument("--residual", action="store_true", help="decoder predicts the change from the input frame")
    ap.add_argument("--init", default=None, help="start from these weights (converted .pt); optimizer starts fresh")
    ap.add_argument("--balance", type=float, default=1.0,
                    help="family share proportional to (training trajectories) ** balance; 1 = plain mix (original)")
    ap.add_argument("--sync", choices=["ddp", "manual"], default="ddp",
                    help="ddp: 3D and 2D accumulate locally, gradients are reduced during the 1D backward (overlapped)")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1234 + rank)
    os.makedirs(a.out, exist_ok=True)

    files = {1: [], 2: [], 3: []}
    family_of = {}
    for fam in FAMILIES:
        paths, dim, T = family_files(a.pdebench, fam)
        files[dim] += [(p, dim, T) for p in paths]
        family_of.update({os.path.basename(p).rsplit(".", 1)[0]: fam for p in paths})
    loaders = {}
    dims_on = sorted({FAMILIES[f][1] for f in a.only}) if a.only else [1, 2, 3]
    for d in dims_on:
        assert a.batch[d - 1] % a.block[d - 1] == 0
        if a.packed:
            pre = packed_prefixes(a.packed, d)
            assert len(pre) == len(files[d]), f"{d}D: {len(pre)} packed files, {len(files[d])} raw files"
            ds = Packed(pre, block=a.block[d - 1])
        else:
            ds = Trajectories(files[d], "train", block=a.block[d - 1])
        if a.balance != 1.0 or a.only:
            names = ([os.path.basename(q) for q in pre] if a.packed
                     else [os.path.basename(f[0]).rsplit(".", 1)[0] for f in files[d]])
            sampler = FamilySampler([family_of[names[fid]] for fid, _ in ds.items], a.balance, rank, world,
                                    keep=set(a.only) if a.only else None)
            if rank == 0:
                print(f"{d}D family shares: " + ", ".join(f"{k} {v:.3f}" for k, v in sampler.shares.items()), flush=True)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(ds, world, rank, shuffle=True, seed=0) if ddp else None
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch[d - 1] // a.block[d - 1], sampler=sampler, shuffle=sampler is None,
                                         num_workers=a.workers[d - 1], collate_fn=Collate(a.max_stride[d - 1]), drop_last=True,
                                         persistent_workers=True, prefetch_factor=4, pin_memory=True)
        loaders[d] = loop(dl, sampler)
        if rank == 0:
            print(f"{d}D: {len(files[d])} files, {ds.n_traj} training trajectories", flush=True)

    test_sets = {}
    if rank == 0:
        for fam in FAMILIES:
            paths, dim, T = family_files(a.pdebench, fam)
            test_sets[fam] = (Trajectories([(p, dim, T) for p in paths], "test", limit=a.eval_samples), T)

    cfg = PaperConfig(hidden=a.hidden, layers=a.layers, heads=a.heads, intermediate=4 * a.hidden, residual=a.residual)
    model = OmniArchPaper(cfg)
    init_weights(model)
    if a.init:
        sd = torch.load(a.init, map_location="cpu", weights_only=True)["model"]
        miss, unexp = model.load_state_dict(sd, strict=False)
        miss = [k for k in miss if not k.endswith("inv_freq")]
        assert not miss and not unexp, (miss[:5], unexp[:5])
        if rank == 0:
            print(f"initialised from {a.init}", flush=True)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    step = 0
    ck = os.path.join(a.out, "last.pt")
    if os.path.exists(ck):
        st = torch.load(ck, map_location="cpu", weights_only=True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        if rank == 0:
            print(f"resumed from step {step}", flush=True)
    params = list(model.parameters())
    fwd = TokenForward(model)
    if ddp and a.sync == "ddp":
        fwd = torch.nn.parallel.DistributedDataParallel(fwd, device_ids=[local], find_unused_parameters=True,
                                                        gradient_as_bucket_view=True, bucket_cap_mb=100)
    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"model {n / 1e6:.1f} M parameters, world {world}", flush=True)
        json.dump(vars(a) | {"config": cfg.to_dict(), "world": world}, open(os.path.join(a.out, "args.json"), "w"))
    log = open(os.path.join(a.out, "log.jsonl"), "a") if rank == 0 else None

    model.train()
    t_last, wait, acc, n_acc = time.time(), 0.0, np.zeros(3), 0
    start_step = step
    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        opt.zero_grad(set_to_none=True)
        losses = {}
        dims_step = [d for d in (3, 2, 1) if d in loaders]
        for i, d in enumerate(dims_step):     # one backward per dimension frees each graph before the next
            last = i == len(dims_step) - 1
            ctx = fwd.no_sync() if (ddp and a.sync == "ddp" and not last) else contextlib.nullcontext()
            t1 = time.time()
            inp, lab, tid, typ, valid, grid, _ = next(loaders[d])
            inp, lab, tid, typ, valid, grid = (t.to(device, non_blocking=True)
                                                for t in (inp, lab, tid, typ, valid, grid))
            wait += time.time() - t1
            with ctx:
                with torch.autocast("cuda", torch.bfloat16):
                    pred = fwd(inp, grid, tid, typ, valid, noise=a.noise, checkpoint=bool(a.checkpoint[d - 1]))
                if a.loss_residual:
                    x0 = inp.float()
                    loss = nrmse_loss(pred.float() - x0, lab - x0, valid, d)
                else:
                    loss = nrmse_loss(pred.float(), lab, valid, d)
                loss.backward()
            losses[d] = loss.detach()
            del pred, loss
        losses = [losses.get(k, torch.zeros((), device=device)) for k in (1, 2, 3)]
        if ddp and a.sync == "manual":
            allreduce_grads(params, world)
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        step += 1
        acc += np.array([l.item() for l in losses])
        n_acc += 1

        if rank == 0 and step % a.log_every == 0:
            dt = time.time() - t_last
            rec = dict(step=step, lr=lr_at(step, a), loss1d=acc[0] / n_acc, loss2d=acc[1] / n_acc,
                       loss3d=acc[2] / n_acc, grad_norm=float(gn), sec_per_step=dt / n_acc,
                       data_wait=wait / n_acc, mem_gb=torch.cuda.max_memory_allocated() / 1e9)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t_last, wait, acc, n_acc = time.time(), 0.0, np.zeros(3), 0
        if step % a.save_every == 0 or step == a.steps:
            if rank == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ck + ".tmp")
                os.replace(ck + ".tmp", ck)
                torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step},
                           os.path.join(a.out, "model.pt"))
            if ddp:
                dist.barrier()
        if step % a.eval_every == 0 or step == a.steps:
            if rank == 0:
                ev = quick_eval(model, test_sets, device)
                rec = dict(step=step, eval=ev)
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if ddp:
                dist.barrier()
        if a.max_steps_this_run and step - start_step >= a.max_steps_this_run:
            break
    if rank == 0:
        open(os.path.join(a.out, "DONE" if step >= a.steps else "STOPPED"), "w").write(str(step))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
