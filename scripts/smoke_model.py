import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.model import PRESETS, OmniArch, count_params
dev = "cuda"
for name, cfg in PRESETS.items():
    m = OmniArch(cfg).to(dev)
    print(name, f"{count_params(m)/1e6:.2f}M", {k: f"{sum(p.numel() for p in v.parameters())/1e6:.2f}M" for k, v in [("enc", m.enc), ("dec", m.dec), ("blocks", m.blocks)]})
    for shape, B in [((256,), 64), ((64, 64), 16), ((32, 32, 32), 4)]:
        C = 3
        x = torch.randn(B, 15, C, *shape, device=dev)
        types = torch.randint(0, 8, (B, C), device=dev)
        torch.cuda.reset_peak_memory_stats(); t = time.time()
        with torch.autocast("cuda", torch.bfloat16):
            pn, (mu, sd) = m(x, types, noise=0.01)
        loss = pn.float().pow(2).mean(); loss.backward(); torch.cuda.synchronize()
        print(" ", shape, tuple(pn.shape), f"{time.time()-t:.3f}s", f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    # resolution transfer + rollout
    with torch.autocast("cuda", torch.bfloat16):
        r = m.rollout(torch.randn(2, 10, 1, 128, 128, device=dev), torch.zeros(2, 1, dtype=torch.long, device=dev), 3)
    print("  rollout", tuple(r.shape))
# block-causal check: changing a later frame must not change earlier outputs
m = OmniArch(PRESETS["tiny"]).to(dev).eval()
for p in m.parameters(): p.data.normal_(0, 0.02)
x = torch.randn(1, 6, 2, 64, device=dev); t = torch.tensor([[0, 1]], device=dev)
y1, _ = m(x, t)
x2 = x.clone(); x2[:, 4:] += 1
# stats change with x2, so compare in a way that removes normalization: use stats-free check with identical stats
x2[:, 4:] = x[:, 4:].flip(-1)  # same per-channel mean/std overall
y2, _ = m(x2, t)
print("causal max diff early frames", (y1[:, :4] - y2[:, :4]).abs().max().item(), "late", (y1[:, 4:] - y2[:, 4:]).abs().max().item())
