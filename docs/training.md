# Training

## Data preparation and training

With Slurm, the paths and cluster settings are read from `.env`, which is excluded from version control:

```bash
cp .env.example .env                     # set data paths and Slurm options
scripts/slurm/submit.sh prepare          # convert PDEBench (about 24 GB after subsampling), one array task per file
scripts/slurm/submit.sh train mini       # four GPUs on one node; further train.py options follow the preset
```

Without Slurm, the same steps are run directly:

```bash
python scripts/prepare_pdebench.py --root /path/to/pdebench --out data/omniarch
OMNIARCH_FUSED=1 torchrun --standalone --nproc_per_node 4 scripts/train.py \
    --data data/omniarch --out runs/mini --preset mini --balanced --augment --compile
python scripts/evaluate.py --ckpt runs/mini/omniarch.pt --data data/omniarch --out results/mini.json
```

Each training step processes one batch from each spatial dimension, with a separate backward pass per dimension so that the peak memory is determined by the largest batch. With `--balanced`, all GPUs draw from the same dataset at each step, which avoids waiting at the gradient synchronisation. `--augment` applies random periodic shifts and reflections on the GPU, reversing the sign of the reflected velocity components. `--max_stride K` samples the frame spacing uniformly between 1 and K. `scripts/evaluate.py --start S --history K` forecasts from frame S using the K preceding frames, so that histories of different lengths can be compared on identical targets.

## Training speed

The point-wise multilayer perceptron of the decoder is limited by memory bandwidth. `omniarch/fused_mlp.py` implements it as a fused Triton kernel for the forward and backward passes, which keeps the hidden activations on chip. The kernel is enabled with `OMNIARCH_FUSED=1` and falls back to PyTorch on CPU, on MPS or when Triton is unavailable. All variants produce identical results up to floating-point rounding (see `tests/`). One training step of `mini` on an RTX PRO 6000 (bf16; 1D, 2D and 3D batches of 256, 64 and 16):

| | FFT implementation | Matrix DFT | Matrix DFT and fused MLP |
|---|---|---|---|
| Step time | 239 ms | 223 ms | 133 ms |
| Peak memory | 21.6 GB | 20.4 GB | 5.8 GB |

Since v0.3, `--compile` compiles the complete model with `torch.compile`, producing one graph per spatial dimension and number of fields; compilation takes a few minutes once per run. A single-GPU run no longer uses the DDP wrapper. The table gives steady-state throughput for `mini` on all three dimensions on one RTX 5090.

| Configuration | Steps per second | Relative throughput |
|---|---|---|
| Training loop of v0.2 | 4.45 | 1.0 |
| `--compile`, single-GPU path | 6.55 | 1.5 |
| Two compiled runs sharing one GPU | 5.33 + 5.61 | 2.5 |

A `mini` run occupies about 11 GB, so that two runs fit on a 32 GB GPU; `scripts/slurm/run_py.sbatch` executes several commands within one Slurm job. `scripts/profile_loop.py` and `scripts/bench_train.py` report the time spent in each phase of a training step.

Inference on an eight-core CPU with the bundled cases takes, for `base` and `tiny` respectively, 1.8 s and 0.1 s for 31 steps of one-dimensional Burgers, 2.2 s and 0.5 s for 11 steps in two dimensions (64 × 64, four fields), and 10.5 s and 4.6 s for 11 steps in three dimensions (32³, five fields). On a CPU, `mini` is nearly as slow as `base`, because the decoder rather than the Transformer dominates the cost.

## Repository layout

```
omniarch/model.py       Fourier encoders and decoders, Transformer
omniarch/fused_mlp.py   fused Triton kernel for the decoder MLP (OMNIARCH_FUSED=1)
omniarch/inference.py   OmniArchPredictor: loading, grid checks, rollout
omniarch/cli.py         command-line tool
omniarch/server.py      web demonstration and HTTP interface (FastAPI); page in omniarch/static/
scripts/                data preparation, training, evaluation, analysis
scripts/slurm/          Slurm job scripts, configured through .env
tests/                  equivalence tests for the accelerated code paths (GPU required)
results/                benchmark and study results
examples/               demonstration cases from PDEBench, including zero-shot problems
docs/                   documentation and figures
```
