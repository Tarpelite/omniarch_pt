"""Convert the original OmniArch-L PyTorch Lightning checkpoint into a plain weights file.

    python scripts/convert_paper_ckpt.py OA-L-RPO-nrmse-0.03878.ckpt weights/omniarch-large-icml2025.pt

The original file (7.3 GiB) also holds the optimizer state; the converted file keeps only the
model weights in float32 (2.6 GB) together with the configuration, and loads without
PyTorch Lightning or Hugging Face Transformers. The unused token-embedding table of the
language-model backbone (32000 x 1024) is dropped.
"""
import argparse
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.paper import PaperConfig, _state_from_lightning, load_paper_model  # noqa: E402


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
a = ap.parse_args()

sd = _state_from_lightning(a.src)
sd = {k: v.float().contiguous() for k, v in sd.items()}
obj = {"config": PaperConfig().to_dict(), "model": sd,
       "source": os.path.basename(a.src),
       "note": "OmniArch-L, ICML 2025 (Chen et al.): original pre-trained weights, model part only"}
os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
torch.save(obj, a.dst)
model = load_paper_model(a.dst)          # round trip: strict key check
n = sum(p.numel() for p in model.parameters())
print(f"wrote {a.dst}: {len(sd)} tensors, {n / 1e6:.1f} M parameters, {os.path.getsize(a.dst) / 1e9:.2f} GB")
print(f"sha256 {sha256(a.dst)}")
