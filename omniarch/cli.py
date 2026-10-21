"""Command line entry point: `omniarch <command>`.

    omniarch info                         model size, device, supported channels
    omniarch examples                     list bundled example cases
    omniarch predict CASE [--steps N]     forecast a case, print nRMSE, save prediction (and a GIF)
    omniarch from-pdebench FILE.hdf5      cut one trajectory out of a raw PDEBench file into a case
    omniarch serve                        start the web demo
"""
import argparse
import json
import os
import sys

import numpy as np

from .model import MAX_CHANNELS

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = [os.path.join(HERE, "..", "examples"), os.path.join(HERE, "examples")]


def example_dir():
    for d in EXAMPLES:
        if os.path.isdir(d):
            return os.path.abspath(d)
    return None


def load_case(path):
    """A case is an .npz with `data` (T, C, *S); optional `channels` (display names), `title`, `history`."""
    if not os.path.exists(path):
        d = example_dir()
        alt = os.path.join(d, path if path.endswith(".npz") else path + ".npz") if d else None
        if alt and os.path.exists(alt):
            path = alt
        else:
            raise SystemExit(f"case not found: {path}")
    z = np.load(path, allow_pickle=False)
    data = z["data"].astype(np.float32)
    names = [str(c) for c in z["channels"]] if "channels" in z else [f"field{i}" for i in range(data.shape[1])]
    case = {"path": path, "data": data, "channels": names}
    for k in ("title", "note", "source"):
        if k in z:
            case[k] = str(z[k])
    case["history"] = int(z["history"]) if "history" in z else 10
    return case


def cmd_info(a):
    from .inference import OmniArchPredictor
    p = OmniArchPredictor(a.ckpt, a.device)
    c = p.config
    print(f"checkpoint : {p.ckpt_path}")
    print(f"parameters : {p.n_params / 1e6:.1f} M  (hidden {c.hidden}, {c.layers} layers)")
    print(f"device     : {p.device}")
    print(f"fields     : up to {MAX_CHANNELS} per problem, no names needed; keep the training order")
    print("min grid   : " + ", ".join(f"{d}D >= {2 * c.modes[d]}" for d in (1, 2, 3)))
    if p.info.get("eval"):
        print(f"train eval : mean nRMSE {p.info['eval'].get('mean')}")


def cmd_examples(a):
    d = example_dir()
    if not d:
        raise SystemExit("no examples directory found")
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz"):
            c = load_case(os.path.join(d, f))
            shape = "x".join(map(str, c["data"].shape[2:]))
            print(f"{f[:-4]:24s} {len(c['data'].shape) - 2}D {shape:10s} {c['data'].shape[0]:3d} frames  "
                  f"{','.join(c['channels']):28s} {c.get('title', '')}")


def save_gif(path, true, pred, channels, history, ch=0, fps=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    dim = true.ndim - 2
    T = true.shape[0]

    def frame2d(a, t):
        f = a[t, ch]
        return f[f.shape[0] // 2] if dim == 3 else f

    fig, axes = plt.subplots(1, 3 if dim > 1 else 1, figsize=(10 if dim > 1 else 6, 3.4))
    if dim == 1:
        ax = axes
        x = np.arange(true.shape[-1])
        lt, = ax.plot(x, true[0, ch], color="#222", lw=1.6, label="ground truth")
        lp, = ax.plot(x, true[0, ch], color="#e4572e", lw=1.4, ls="--", label="OmniArch")
        lo, hi = true[:, ch].min(), true[:, ch].max()
        ax.set_ylim(lo - 0.1 * (hi - lo + 1e-9), hi + 0.1 * (hi - lo + 1e-9))
        ax.legend(loc="upper right", frameon=False)
        title = ax.set_title("")

        def upd(t):
            lt.set_ydata(true[t, ch])
            lp.set_ydata(pred[t, ch] if t >= history else true[t, ch])
            title.set_text(f"{channels[ch]}  t={t}" + ("  (given)" if t < history else "  (forecast)"))
    else:
        vmin, vmax = np.percentile(true[:, ch], [1, 99])
        ims = []
        for ax, name in zip(axes, ["ground truth", "OmniArch", "|error|"]):
            im = ax.imshow(frame2d(true, 0), cmap="magma" if name == "|error|" else "RdBu_r",
                           vmin=0 if name == "|error|" else vmin, vmax=(vmax - vmin) * 0.3 if name == "|error|" else vmax,
                           origin="lower")
            ax.set_title(name); ax.axis("off"); ims.append(im)
        sup = fig.suptitle("")

        def upd(t):
            p = pred if t >= history else true
            ims[0].set_data(frame2d(true, t)); ims[1].set_data(frame2d(p, t))
            ims[2].set_data(np.abs(frame2d(p, t) - frame2d(true, t)))
            sup.set_text(f"{channels[ch]}  t={t}" + ("  (given)" if t < history else "  (forecast)"))
    fig.tight_layout()
    FuncAnimation(fig, upd, frames=T).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def cmd_predict(a):
    from .inference import OmniArchPredictor, nrmse_np
    case = load_case(a.case)
    data, chans = case["data"], case["channels"]
    k = a.history or case["history"]
    steps = a.steps or data.shape[0] - k
    p = OmniArchPredictor(a.ckpt, a.device)
    pred = p.predict(data[:k], steps)
    full = np.concatenate([data[:k], pred], 0)
    print(f"case      : {case.get('title', os.path.basename(case['path']))}")
    print(f"input     : {k} frames, {len(chans)} channels, grid {'x'.join(map(str, data.shape[2:]))}")
    print(f"forecast  : {steps} steps in {p.last_seconds * 1000:.0f} ms on {p.device}")
    n_true = min(steps, data.shape[0] - k)
    if n_true > 0:
        e = nrmse_np(pred[:n_true], data[k:k + n_true])
        base = nrmse_np(np.repeat(data[k - 1:k], n_true, 0), data[k:k + n_true])
        for c, v, b in zip(chans, e, base):
            print(f"nRMSE     : {c:9s} {v:.4f}   (copy-last-frame baseline {b:.4f})")
        print(f"nRMSE     : {'mean':9s} {e.mean():.4f}   (baseline {base.mean():.4f})")
    out = a.out or os.path.splitext(os.path.basename(case["path"]))[0] + "_pred.npz"
    np.savez_compressed(out, data=full, channels=np.array(chans), history=k)
    print(f"saved     : {out}  (frames 0..{k - 1} given, {k}.. predicted)")
    if a.gif:
        if data.shape[0] < full.shape[0]:
            print("gif skipped: no ground truth for the extra steps")
        else:
            save_gif(a.gif, data[:full.shape[0]], full, chans, k, ch=chans.index(a.channel) if a.channel else 0)
            print(f"gif       : {a.gif}")


def cmd_from_pdebench(a):
    """Cut one trajectory out of a raw PDEBench file, subsampling while reading to keep memory low."""
    import h5py

    def read(ds, sample_axis):
        nd = ds.ndim - (2 if sample_axis else 1)
        sl = (slice(None, None, a.time_stride),) + (slice(None, None, a.space_stride),) * nd
        return ds[(a.index,) + sl] if sample_axis else ds[sl]

    with h5py.File(a.file, "r") as h:
        keys = list(h.keys())
        n_t = h["t-coordinate"].shape[0] if "t-coordinate" in keys else -1
        if "tensor" in keys:
            ds = h["tensor"]
            arr = read(ds, ds.shape[0] not in (n_t, n_t - 1))[:, None]
            chans = ["u"]
        elif "density" in keys:
            chans = [c for c in ["density", "Vx", "Vy", "Vz", "pressure"] if c in keys]
            # single-trajectory files (PDEBench Test/) have no sample axis: first axis is time
            single = h[chans[0]].shape[0] in (n_t, n_t - 1)
            arr = np.stack([read(h[c], not single) for c in chans], 1)
        else:
            g = h[f"{a.index:04d}"]["data"]
            nd = g.ndim - 2
            arr = np.moveaxis(g[(slice(None, None, a.time_stride),) + (slice(None, None, a.space_stride),) * nd], -1, 1)
            chans = ["u", "v"] if arr.shape[1] == 2 else ["u"]
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if a.channels:
        chans = a.channels.split(",")
    np.savez_compressed(a.out, data=arr, channels=np.array(chans), history=a.history,
                        title=a.title or os.path.basename(a.file), source=os.path.basename(a.file))
    print(f"saved {a.out}: {arr.shape} channels={chans}")


def cmd_serve(a):
    from .server import run
    run(a.host, a.port, a.ckpt, a.device)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omniarch", description="OmniArch (智算天璇) mini: PDE forecasting from a few frames")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt", help="checkpoint path (default: $OMNIARCH_CKPT, ./weights, ~/.cache/omniarch)")
    common.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1 | mps")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", parents=[common]).set_defaults(fn=cmd_info)
    sub.add_parser("examples").set_defaults(fn=cmd_examples)
    p = sub.add_parser("predict", parents=[common])
    p.add_argument("case", help="example name or .npz path")
    p.add_argument("--history", type=int, help="number of given frames (default: from the case, usually 10)")
    p.add_argument("--steps", type=int, help="frames to forecast (default: to the end of the case)")
    p.add_argument("--out", help="output .npz")
    p.add_argument("--gif", help="also write an animation comparing forecast and truth")
    p.add_argument("--channel", help="channel shown in the GIF")
    p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("from-pdebench")
    p.add_argument("file")
    p.add_argument("--index", type=int, default=0, help="sample index inside the file")
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--space-stride", type=int, default=1)
    p.add_argument("--channels", help="comma separated names, override auto detection")
    p.add_argument("--history", type=int, default=10)
    p.add_argument("--title")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_from_pdebench)
    p = sub.add_parser("serve", parents=[common])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(fn=cmd_serve)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"omniarch: {e}")


if __name__ == "__main__":
    main()
