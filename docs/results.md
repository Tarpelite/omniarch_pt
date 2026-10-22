# Benchmark results

## Protocol

Each test trajectory is forecast from its first ten frames by autoregressive rollout to the final frame. The error is the normalised root-mean-square error (nRMSE) over all forecast frames, averaged over samples and fields; lower values are better. As a reference, the last given frame is copied forward. The grids are subsampled from PDEBench to 256 points in one dimension, 64 × 64 in two dimensions and 32³ in three dimensions. Because the original paper used a different resolution and protocol, the values below are not comparable with those reported there.

## Results by PDE family

| PDE family | Dim. | OmniArch base | OmniArch mini | OmniArch tiny | Copy last frame |
|---|---|---|---|---|---|
| Advection | 1D | 0.106 | 0.200 | 0.354 | 1.001 |
| Burgers | 1D | 0.123 | 0.145 | 0.172 | 0.335 |
| Compressible flow | 1D | 0.135 | 0.146 | 0.249 | 0.453 |
| Diffusion-sorption | 1D | 0.042 | 0.032 | 0.028 | 0.157 |
| Reaction-diffusion | 1D | 0.047 | 0.043 | 0.047 | 0.042 |
| Compressible Navier-Stokes | 2D | 0.214 | 0.250 | 0.326 | 0.648 |
| Diffusion-reaction | 2D | 0.120 | 0.156 | 0.164 | 0.496 |
| Shallow water | 2D | 0.003 | 0.004 | 0.005 | 0.110 |
| Compressible Navier-Stokes and turbulence | 3D | 0.513 | 0.535 | 0.565 | 0.886 |
| **Mean over families** | | **0.145** | **0.168** | **0.212** | **0.459** |

The `base` column refers to the v0.4 checkpoint (72k steps); the v0.2 checkpoint (36k steps) reached 0.159. `mini` and `tiny` are the v0.2 checkpoints. Controlled comparisons of model size and training length are reported in [studies.md](studies.md#scaling).

Two limitations should be noted. In the one-dimensional reaction-diffusion data of PDEBench, every trajectory is a constant field from the first saved frame onwards, so that copying the last frame is already almost exact; this family is therefore uninformative. The three-dimensional family has the fewest training trajectories and remains the most difficult.

## Zero-shot cases

The following problems are taken from the PDEBench test collection and were not used in training; they differ from the training data in grid size and time step.

| Case | base (v0.4) | base (v0.2) | Copy last frame |
|---|---|---|---|
| 1D Sod shock tube | 0.014 | 0.011 | 0.018 |
| 2D Orszag-Tang vortex | 0.234 | 0.233 | 0.434 |
| 2D Kelvin-Helmholtz instability | 0.385 | 0.358 | 0.301 |
| 3D turbulence, subsampled from 256³ to 32³ | 0.634 | 0.623 | 1.003 |

Longer training improves the benchmark families but not these unseen problems, on which the v0.4 checkpoint is slightly worse.
