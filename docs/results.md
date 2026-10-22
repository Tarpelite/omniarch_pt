# Benchmark results

PDEBench protocol: 10 given frames, autoregressive rollout to the end of the trajectory, nRMSE over all forecast frames (lower is better). The reference row copies the last given frame forward.

| PDE family | Dim | OmniArch base | OmniArch mini | OmniArch tiny | Copy last frame |
|---|---|---|---|---|---|
| Advection | 1D | 0.170 | 0.200 | 0.354 | 1.001 |
| Burgers | 1D | 0.146 | 0.145 | 0.172 | 0.335 |
| Compressible flow | 1D | 0.139 | 0.146 | 0.249 | 0.453 |
| Diffusion-sorption | 1D | 0.039 | 0.032 | 0.028 | 0.157 |
| Reaction-diffusion | 1D | 0.043 | 0.043 | 0.047 | 0.042 |
| Compressible NS | 2D | 0.236 | 0.250 | 0.326 | 0.648 |
| Diffusion-reaction | 2D | 0.142 | 0.156 | 0.164 | 0.496 |
| Shallow water | 2D | 0.004 | 0.004 | 0.005 | 0.110 |
| Compressible NS / turbulence | 3D | 0.518 | 0.535 | 0.565 | 0.886 |
| **Mean of families** | | **0.159** | **0.168** | **0.212** | **0.459** |

Going from 14M to 308M parameters lowers the mean error by a quarter; see [Studies](studies.md#scaling) for controlled comparisons.

Grids are subsampled from PDEBench: 256 points in 1D, 64×64 in 2D, 32³ in 3D. Numbers are not comparable with the paper, which used a different resolution and protocol. Two honest caveats: reaction-diffusion trajectories flatten out within a few frames, so copying the last frame is already near-perfect and the model is slightly worse; 3D is the weakest family (fewest training trajectories).

**Zero-shot cases** (PDEBench `Test/` problems never seen in training, different grids and time steps; `omniarch-base`, nRMSE):

| Case | OmniArch base | Copy last frame |
|---|---|---|
| 1D Sod shock tube | 0.011 | 0.018 |
| 2D Orszag–Tang vortex | 0.234 | 0.434 |
| 2D Kelvin–Helmholtz instability | 0.358 | 0.301 |
| 3D turbulence, 256³ subsampled to 32³ | 0.623 | 1.003 |
