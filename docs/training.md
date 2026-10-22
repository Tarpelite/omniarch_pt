# Training

```bash
cp .env.example .env        # fill in paths and Slurm settings; .env is git-ignored
scripts/slurm/submit.sh prepare          # convert PDEBench (about 24 GB after subsampling), one array task per file
scripts/slurm/submit.sh train mini       # 4 GPUs on one node; add train.py flags after the preset
```

Without Slurm, run the same steps directly:

```bash
python scripts/prepare_pdebench.py --root /path/to/pdebench --out data/omniarch
OMNIARCH_FUSED=1 torchrun --standalone --nproc_per_node 4 scripts/train.py --data data/omniarch --out runs/mini --preset mini --balanced --augment
python scripts/evaluate.py --ckpt runs/mini/omniarch.pt --data data/omniarch --out results/mini.json
```

Each step trains on one 1D, one 2D and one 3D batch, with one backward pass per dimension so peak memory is that of the largest. `--balanced` makes all GPUs draw the same dataset each step (no waiting at the gradient sync); `--augment` adds random periodic shifts and mirror images (velocity components flip sign), done on the GPU.

## Speed

Inference on an 8-core CPU (bundled cases), `base` / `tiny`: 31 steps of 1D Burgers 1.8 s / 0.1 s, 11 steps of 2D (64×64, 4 fields) 2.2 s / 0.5 s, 11 steps of 3D (32³, 5 fields) 10.5 s / 4.6 s. `mini` is close to `base` on CPU because the decoder, not the transformer, dominates.

Training: the decoder's point-wise MLP is memory-bound. `omniarch/fused_mlp.py` provides a fused Triton kernel (forward and backward, hidden activations stay on chip). Enable it with `OMNIARCH_FUSED=1`; it falls back to PyTorch on CPU, MPS or without Triton. One training step on an RTX PRO 6000 (bf16, 1D/2D/3D batches 256/64/16):

| | Original FFT path | + matrix DFT | + fused MLP |
|---|---|---|---|
| mini, step time | 239 ms | 223 ms | 133 ms |
| mini, peak memory | 21.6 GB | 20.4 GB | 5.8 GB |

The v0.2 checkpoints were trained with the fused kernel; all three paths give the same outputs up to floating-point rounding (see `tests/`), so any of them can run inference.

Since v0.3, `scripts/train.py --compile` compiles the whole model (one graph per dimension and field count; a few minutes once), and a single-GPU run skips the DDP wrapper. Steady state, `mini`, all three dimensions, one RTX 5090:

| | Steps per second | vs. before |
|---|---|---|
| v0.2 training loop | 4.45 | 1.0x |
| `--compile`, single-GPU path | 6.55 | 1.5x |
| two compiled jobs sharing the GPU | 5.33 + 5.61 | 2.5x |

`mini` uses about 11 GB, so two jobs fit on a 32 GB card; `scripts/slurm/run_py.sbatch` runs several commands in one Slurm job. `scripts/profile_loop.py` and `scripts/bench_train.py` break a step down by phase.


## Layout

```
omniarch/model.py       Fourier encoders/decoders + transformer
omniarch/fused_mlp.py   fused Triton kernel for the decoder MLP (opt-in, OMNIARCH_FUSED=1)
scripts/slurm/          submit.sh + job scripts, configured through .env
omniarch/inference.py   OmniArchPredictor: load, check grid, roll out
omniarch/cli.py         `omniarch` command
omniarch/server.py      web demo backend (FastAPI), page in omniarch/static/
scripts/                data preparation, training, evaluation, demo cases
tests/                  equivalence tests for the fast paths (run on a GPU)
results/                full benchmark output of the released checkpoints
examples/               demo cases (PDEBench test trajectories and zero-shot problems)
docs/                   deployment guide (Chinese), figures
```
