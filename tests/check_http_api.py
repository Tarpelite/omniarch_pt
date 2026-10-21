"""Smoke test of a running `omniarch serve`: python tests/check_http_api.py http://127.0.0.1:7860"""
import io
import sys

import numpy as np
import requests

url = sys.argv[1].rstrip("/")
z = np.load("examples/2d_turbulence.npz")
data = z["data"].astype("float32")
buf = io.BytesIO()
np.savez(buf, data=data[:10])
r = requests.post(f"{url}/api/forecast", params={"steps": 11}, files={"file": ("case.npz", buf.getvalue())})
assert r.status_code == 200, (r.status_code, r.text)
out = np.load(io.BytesIO(r.content))
p, t = out["prediction"], data[10:21]
err = np.mean(np.sqrt(((p - t) ** 2).sum((0, 2, 3))) / np.sqrt((t ** 2).sum((0, 2, 3))))
print(f"forecast ok: prediction {p.shape}, {float(out['seconds']):.2f} s, nRMSE vs truth {err:.4f}")
r = requests.post(f"{url}/api/forecast", params={"steps": 5}, files={"file": ("bad.npz", b"not an npz")})
print(f"bad file -> {r.status_code}: {r.text[:90]}")
assert r.status_code == 400
