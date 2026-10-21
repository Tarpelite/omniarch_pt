"""How many batches per second can the loader of one GPU rank deliver, per dimension?"""
import os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniarch.data import WindowBatches, load_entries

data, workers = sys.argv[1], int(sys.argv[2])
batch = [int(x) for x in sys.argv[3].split(",")]
entries = load_entries(data)
for aug in (False, True):
    for i, d in enumerate((1, 2, 3)):
        ds = WindowBatches(entries, d, batch[i], (16, 16, 12)[i], seed=d, shared_seed=7 + d, augment=aug)
        it = iter(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, persistent_workers=True, prefetch_factor=4))
        for _ in range(3 * workers):
            next(it)
        t = time.time(); n = 30
        for _ in range(n):
            next(it)
        print(f"augment={aug} {d}D batch {batch[i]:4d}: {n / (time.time() - t):6.1f} batches/s  ({workers} workers)", flush=True)
        del it
