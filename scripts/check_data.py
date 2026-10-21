import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.data import load_entries
for e in load_entries(sys.argv[1]):
    a = e.array("train"); b = e.array("test")
    s = np.asarray(a[:20])
    per_ch = [f"{c}:{s[:, :, i].mean():.3g}±{s[:, :, i].std():.3g}" for i, c in enumerate(e.meta["channels"])]
    # how much do consecutive frames change relative to field scale
    d = np.abs(np.diff(s, axis=1)).mean() / (np.abs(s).mean() + 1e-12)
    print(f"{e.name:28s} {str(a.shape):28s} test{b.shape[0]:4d} step_change={d:.3f} {' '.join(per_ch)}")
