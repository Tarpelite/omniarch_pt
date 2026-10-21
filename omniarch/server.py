"""Web demo: `omniarch serve` then open http://127.0.0.1:7860

Pick an example (or upload a case .npz), choose how many frames to give the
model and how far to forecast, and compare the forecast with ground truth.
"""
import base64
import io
import os
import tempfile
import threading

import numpy as np

from .cli import example_dir, load_case
from .inference import OmniArchPredictor, nrmse_np

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_VIEW = 128          # longest displayed axis; larger grids are subsampled for the browser only


def b64(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return {"shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode()}


def view(a):
    """(T, C, *S) -> what the browser draws: 1D stays 1D, 3D becomes three mid-plane slices."""
    dim = a.ndim - 2
    if dim == 3:
        X, Y, Z = a.shape[2:]
        s = [a[:, :, X // 2], a[:, :, :, Y // 2], a[:, :, :, :, Z // 2]]
        return [view(x)[0] for x in s]
    step = max(1, max(a.shape[2:]) // MAX_VIEW)
    sl = (slice(None), slice(None)) + (slice(None, None, step),) * dim
    return [a[sl]]


def per_step_nrmse(pred, true):
    """(T, C, *S) -> (T, C) relative L2 error of each frame."""
    red = tuple(range(2, pred.ndim))
    return np.sqrt(((pred - true) ** 2).sum(red)) / np.sqrt((true ** 2).sum(red)).clip(1e-12)


def create_app(ckpt=None, device=None):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    app = FastAPI(title="OmniArch mini demo")
    pred = OmniArchPredictor(ckpt, device)
    lock = threading.Lock()
    uploads = {}
    updir = tempfile.mkdtemp(prefix="omniarch_")

    def get_case(name):
        if name in uploads:
            return load_case(uploads[name])
        d = example_dir()
        p = os.path.join(d, name + ".npz") if d else None
        if not p or not os.path.exists(p):
            raise HTTPException(404, f"unknown case {name}")
        return load_case(p)

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/api/info")
    def info():
        c = pred.config
        return {"params": pred.n_params, "device": str(pred.device), "hidden": c.hidden, "layers": c.layers,
                "ckpt": os.path.basename(pred.ckpt_path), "eval": pred.info.get("eval", {})}

    @app.get("/api/cases")
    def cases():
        d = example_dir()
        out = []
        for f in sorted(os.listdir(d)) if d else []:
            if not f.endswith(".npz"):
                continue
            c = load_case(os.path.join(d, f))
            # thumbnail: middle frame of the field that varies most
            mid = c["data"][c["data"].shape[0] // 2]
            rel = [float(f.std() / (abs(f.mean()) + f.std() + 1e-12)) for f in mid]
            ch = int(np.argmax(rel))
            first = view(c["data"][c["data"].shape[0] // 2:][:1, ch:ch + 1])[0][0, 0]
            out.append({"id": f[:-4], "title": c.get("title", f[:-4]), "note": c.get("note", ""),
                        "dim": c["data"].ndim - 2, "grid": list(c["data"].shape[2:]), "frames": c["data"].shape[0],
                        "channels": c["channels"], "history": c["history"], "thumb": b64(first)})
        return out

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        raw = await file.read()
        try:
            z = np.load(io.BytesIO(raw), allow_pickle=False)
            data = z["data"]
            assert data.ndim in (3, 4, 5), f"data has shape {data.shape}"
            if "channels" in z:
                assert len(z["channels"]) == data.shape[1], "channels length does not match data"
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"not a valid case file: need data (T, C, *grid): {e}")
        cid = f"upload{len(uploads)}"
        p = os.path.join(updir, cid + ".npz")
        open(p, "wb").write(raw)
        uploads[cid] = p
        return {"id": cid}

    @app.post("/api/forecast")
    async def forecast(file: UploadFile = File(...), history: int = 0, steps: int = 10):
        """Programmatic use: send an .npz whose `data` (T, C, *grid) holds the latest frames, get back an
        .npz with `prediction` (steps, C, *grid) and `seconds`. history=0 (default) uses all T frames;
        history=K uses the last K. Example:
            curl -F file=@case.npz "http://host:7860/api/forecast?history=10&steps=20" -o pred.npz
        """
        from fastapi.responses import Response
        try:
            z = np.load(io.BytesIO(await file.read()), allow_pickle=False)
            data = z["data"].astype(np.float32)
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "not a valid .npz file with an array named `data` of shape (frames, fields, *grid)")
        try:
            assert data.ndim in (3, 4, 5), f"data has shape {data.shape}, need (T, C, *grid)"
            history = history or data.shape[0]
            assert 1 <= history <= data.shape[0], f"history={history} but the file has {data.shape[0]} frames"
            assert 1 <= steps <= 1000, "steps must be between 1 and 1000"
            with lock:
                fut = pred.predict(data[data.shape[0] - history:], steps)
        except (AssertionError, ValueError, KeyError) as e:
            raise HTTPException(400, str(e))
        buf = io.BytesIO()
        np.savez_compressed(buf, prediction=fut, seconds=pred.last_seconds)
        return Response(buf.getvalue(), media_type="application/octet-stream",
                        headers={"Content-Disposition": "attachment; filename=prediction.npz"})

    class Req(BaseModel):
        case: str
        history: int = 10
        steps: int = 0

    @app.post("/api/predict")
    def predict(r: Req):
        c = get_case(r.case)
        data, chans = c["data"], c["channels"]
        T = data.shape[0]
        k = max(1, min(r.history, T - 1))
        steps = r.steps if r.steps > 0 else T - k
        try:
            with lock:
                fut = pred.predict(data[:k], steps)
        except ValueError as e:
            raise HTTPException(400, str(e))
        full = np.concatenate([data[:k], fut], 0)
        n = min(steps, T - k)
        truth = data[:k + n]
        err = per_step_nrmse(full[k:k + n], truth[k:]) if n > 0 else np.zeros((0, len(chans)))
        base = per_step_nrmse(np.repeat(data[k - 1:k], n, 0), truth[k:]) if n > 0 else err
        return {"channels": chans, "history": k, "steps": steps, "dim": data.ndim - 2, "grid": list(data.shape[2:]),
                "seconds": pred.last_seconds, "device": str(pred.device),
                "truth": [b64(v) for v in view(truth)], "pred": [b64(v) for v in view(full)],
                "nrmse_t": err.tolist(), "baseline_t": base.tolist(),
                "nrmse": float(nrmse_np(full[k:k + n], truth[k:]).mean()) if n else None,
                "baseline": float(nrmse_np(np.repeat(data[k - 1:k], n, 0), truth[k:]).mean()) if n else None}

    return app


def run(host="127.0.0.1", port=7860, ckpt=None, device=None):
    import uvicorn
    app = create_app(ckpt, device)
    print(f"OmniArch demo on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
