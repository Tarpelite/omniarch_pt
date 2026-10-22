# Using OmniArch-pt

HTTP API (served by `omniarch serve`): send the latest frames, get the forecast back.

```bash
curl -F file=@my_case.npz "http://127.0.0.1:7860/api/forecast?steps=20" -o pred.npz   # pred.npz: prediction, seconds
```

Python API:

```python
from omniarch import OmniArchPredictor
p = OmniArchPredictor("weights/omniarch-base.pt")          # device picked automatically
future = p.predict(history, steps=10)
# history: float array (K, C, *grid), K >= 1 given frames, C <= 8 fields; returns (steps, C, *grid)
```

Fields need no names, but their order matters: keep the order used in training for that kind of problem. For compressible flow it is density, Vx, (Vy, Vz), pressure; for shallow water the water height; for 2D diffusion-reaction u then v.

## How many frames to give

The model reads the equation and the time step from the given frames, so more history helps. Same forecast frames, `omniarch-base`, mean nRMSE over the 9 families: 10 frames 0.160, 4 frames 0.190, 2 frames 0.301, 1 frame 0.409. Only advection is fine with 2 frames (0.172 vs 0.169). Give at least 4 frames, 10 if you have them (`results/context.json`).

## Your own data

A case is an `.npz` with `data` of shape `(T, C, *grid)`. `channels` is optional and only labels the fields in the demo:

```python
np.savez("my_case.npz", data=arr.astype("float32"), channels=np.array(["density", "Vx", "pressure"]))
```

Raw PDEBench files can be cut directly: `omniarch from-pdebench FILE.hdf5 --index 0 --space-stride 2 --out case.npz` (needs `pip install h5py`).

Minimum grid: 48 points (1D), 32×32 (2D), 16³ (3D). Periodic boundaries work best; the model was trained on periodic and simple non-periodic problems only.
