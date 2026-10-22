# Usage

## Python interface

```python
from omniarch import OmniArchPredictor

p = OmniArchPredictor("weights/omniarch-base.pt")     # the device is selected automatically
future = p.predict(history, steps=10)
```

`history` is a floating-point array of shape `(K, C, *grid)`, where `K` is the number of given frames and `C` the number of physical fields (at most eight). The method returns an array of shape `(steps, C, *grid)`.

Fields are identified by their position, not by name. The order of the fields must therefore match the order used in training for the corresponding class of problems: density, velocity components and pressure for compressible flow; water height for shallow water; `u` followed by `v` for the two-dimensional diffusion-reaction system.

## HTTP interface

`omniarch serve` starts a web demonstration and an HTTP interface. A forecast is requested by sending the most recent frames as an `.npz` file:

```bash
curl -F file=@my_case.npz "http://127.0.0.1:7860/api/forecast?steps=20" -o pred.npz
```

The returned file contains the predicted frames (`prediction`) and the computation time in seconds.

## Number of given frames

The model infers the governing dynamics and the time step from the given frames, and its accuracy therefore depends on the length of the history. The table reports the mean nRMSE of the v0.2 `omniarch-base` checkpoint over the nine PDE families when the same frames are forecast from histories of different lengths.

| Given frames | 10 | 4 | 2 | 1 |
|---|---|---|---|---|
| Mean nRMSE | 0.160 | 0.190 | 0.301 | 0.409 |

With two frames the error is close to twice that obtained with ten. Only linear advection is predicted equally well from two frames (0.172 against 0.169). We recommend providing at least four frames and preferably ten. The per-family values are given in `results/context.json`.

## Input format

A case is stored as an `.npz` file containing an array `data` of shape `(T, C, *grid)`. An optional array `channels` provides field names, which are used only for display:

```python
np.savez("my_case.npz", data=arr.astype("float32"), channels=np.array(["density", "Vx", "pressure"]))
```

Files from PDEBench can be converted directly with `omniarch from-pdebench FILE.hdf5 --index 0 --space-stride 2 --out case.npz`, which requires the `h5py` package.

The grid must contain at least 48 points in one dimension, 32 × 32 points in two dimensions and 16³ points in three dimensions. Periodic boundary conditions give the best results, since the training data consist of periodic and simple non-periodic problems.
