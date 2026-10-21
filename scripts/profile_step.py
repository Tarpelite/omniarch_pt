"""Where does one training step spend GPU time? Groups CUDA kernels by model part."""
import os, sys, time, torch
from torch.profiler import profile, ProfilerActivity, record_function
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.model import PRESETS, OmniArch

preset = sys.argv[1] if len(sys.argv) > 1 else "mini"
dev = "cuda"
m = OmniArch(PRESETS[preset]).to(dev)
# wrap parts so the profiler can attribute time
for d in ("1", "2", "3"):
    for part, mod in (("enc", m.enc[d]), ("dec", m.dec[d])):
        f = mod.forward
        def wrapped(*a, _f=f, _n=f"{part}{d}d", **k):
            with record_function(_n):
                return _f(*a, **k)
        mod.forward = wrapped
shapes = {1: (256, 16, 1, (256,)), 2: (64, 16, 4, (64, 64)), 3: (16, 12, 5, (32, 32, 32))}
bs = {"mini": (256, 64, 16), "tiny": (512, 128, 32)}[preset]
batches = [torch.randn(b, T, C, *S, device=dev) for b, (_, T, C, S) in zip(bs, shapes.values())]

def step():
    for x in batches:
        with torch.autocast("cuda", torch.bfloat16):
            pn, (mu, sd) = m(x[:, :-1], noise=0.01)
        ((pn.float() - x[:, 1:]) ** 2).mean().backward()

for _ in range(3):
    step()
torch.cuda.synchronize()
t = time.time()
for _ in range(5):
    step()
torch.cuda.synchronize()
print(f"{preset}: {(time.time() - t) / 5 * 1000:.0f} ms per step (3 batches fwd+bwd, 1 GPU, no compile)")

with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    step(); torch.cuda.synchronize()
ev = prof.key_averages()
tot = sum(e.device_time_total for e in ev if e.device_type.name == "CUDA") or 1
print("\n== by model part (forward only, CUDA time inside the region)")
for e in sorted(ev, key=lambda e: -e.device_time_total):
    if e.key.startswith(("enc", "dec")) and e.key[-2:] == "2d" or e.key[-2:] in ("1d", "3d") and e.key.startswith(("enc", "dec")):
        print(f"  {e.key:8s} {e.device_time_total / 1000:7.1f} ms")
print("\n== top CUDA kernels / ops")
rows = sorted([e for e in ev if e.device_time_total > 0 and not e.key.startswith(("enc", "dec"))],
              key=lambda e: -e.self_device_time_total)[:18]
for e in rows:
    print(f"  {e.self_device_time_total / 1000:7.1f} ms  {100 * e.self_device_time_total / tot:5.1f}%  {e.key[:90]}")
