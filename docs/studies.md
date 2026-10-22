# Studies

Controlled experiments behind the released checkpoints. Raw numbers are in `../results/`.

## Scaling

![scaling](scaling.png)

36 training runs on PDEBench (single GPU each; 1D/2D/3D batches 256/64/16 per step), evaluated with the benchmark protocol in [results.md](results.md). Raw numbers in `results/scaling.json`, analysis in `scripts/analyze_scaling.py`.

| Model | Params | 12k steps (3 seeds) | Longer training |
|---|---|---|---|
| nano | 4M | 0.322 ± 0.002 | 0.257 at 108k |
| tiny | 14M | 0.271 ± 0.003 | 0.212 at 108k |
| mini | 77M | 0.217 ± 0.004 | 0.168 at 72k |
| base | 308M | 0.187 ± 0.001 | 0.159 at 36k |
| large | 833M | 0.180 (1 seed) | - |

- **Model size.** At a fixed 12k steps the error follows err ∝ N^-0.11 from 4M to 308M parameters; seed-to-seed spread is 1-2% of the gaps between sizes. The 833M model falls above the trend line: at this budget it is under-trained.
- **Training length matters more than size at this scale.** base trained 3x longer (0.159) beats large at 12k steps (0.180) for less compute.
- **Data starts to matter with longer training.** At 12k steps 25% of the training set is as good as 100%; at 36k-108k steps the 25% runs are 5-7% worse.
- **3D barely moves** (N^-0.03, and little gain from longer training). It is limited by something other than scale, most likely training data (700 trajectories) and the 32³ resolution.


## Cross-dimension training

![cross-dimension study](crossdim.png)

Does one model for 1D, 2D and 3D beat one model per dimension? 22 runs, 36k steps each, `mini` with 2 seeds and `tiny` with 1. Per-dimension mean nRMSE; 1D leaves out reaction-diffusion, whose PDEBench trajectories are constant after the first saved frame.

| Trained on | mini 1D | mini 2D | mini 3D | tiny 1D | tiny 2D | tiny 3D |
|---|---|---|---|---|---|---|
| one dimension only | 0.126 | 0.140 | 0.549 | 0.206 | 0.179 | 0.578 |
| 1D+2D+3D | 0.143 | 0.148 | 0.547 | 0.210 | 0.197 | 0.576 |
| 3D only, 25% of the 3D data | | | 0.667 | | | 0.602 |
| 1D+2D+3D, 25% of the 3D data | | | **0.557** | | | 0.588 |

- **With plenty of data, sharing costs a little.** 1D and 2D lose 2-14%, 3D is unchanged: at these sizes the dimensions compete for capacity.
- **With scarce data, the other dimensions help a lot.** With a quarter of the 3D trajectories, joint training cuts the 3D error of `mini` by 17%, close to `mini` trained on all 3D data (0.549). Both seeds agree (0.556, 0.557).
- **The bigger the model, the larger the gain.** `tiny` barely overfits the smaller 3D set (0.578 to 0.602) and gains 2%; `mini` overfits more (0.549 to 0.667) and gains 17%.

So the value of one model for all dimensions lies in carrying problems with little data, which is the usual case for a new 3D problem. Raw numbers: `results/crossdim.json`; analysis: `scripts/analyze_crossdim.py`.


## Frame spacing

OmniArch is never told the time step; it reads it from how much the given frames change. The v0.2 checkpoints only saw one spacing per dataset, and on compressible flow they fail when frames are 3-4x further apart. Training with a random spacing of 1-4 frames fixes this. Ratio of model error to copying the last frame, mean over 8 families (4 given frames, 4 steps; lower is better):

| Training | spacing 1x | 2x | 3x | 4x |
|---|---|---|---|---|
| fixed spacing (as v0.2) | 0.50 | 0.60 | 0.71 | 0.87, worse than copying on 1D and 2D compressible flow |
| random spacing 1-4 (`omniarch-mini-anystep`) | 0.52 | **0.33** | **0.30** | **0.33** |

Telling the model the spacing explicitly gave the same numbers, so reading it from the context is enough. Raw numbers: `results/anystep.json`; train with `--max_stride 4`, evaluate with `scripts/eval_stride.py`.
