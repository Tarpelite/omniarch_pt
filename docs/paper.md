# Reproducing Table 1 of the ICML 2025 paper

`omniarch-large-pt.pt` is OmniArch-L (24 layers, hidden size 1024, 640 M parameters) re-trained from scratch with this PyTorch code on PDEBench and PDEArena Maxwell-3D. One command evaluates it on the pre-training rows of Table 1 of the paper:

```bash
pip install gdown && gdown 1NJkxgkW8OzK4pCfjCFkos6SyRlnZhkI- -O weights/omniarch-large-pt.pt   # 2.6 GB
python scripts/reproduce_paper.py --pdebench /path/to/PDEBench /path/to/maxwell --ckpt weights/omniarch-large-pt.pt
```

## Results

| PDE | OmniArch-L (this release) | Recovered | Paper (Table 1) | Copy last frame |
|---|---|---|---|---|
| 1D CFD | 0.0347 | 0.0306 | 0.0250 | 0.0729 |
| 1D Advection | 0.0145 | 0.0145 | 0.0182 | 0.2424 |
| 1D Burgers | 0.0031 | 0.0030 | 0.0063 | 0.0055 |
| 1D Diffusion-sorption | 0.0013 | 0.0004 | 0.0015 | 0.0032 |
| 1D Reaction-diffusion | 0.0001 | 0.0000 | 0.0004 | 0.0001 |
| 2D CFD ¹ | 0.0055 | 0.0047 | 0.0148 | 0.0598 |
| 2D Reaction-diffusion | 0.0060 | 0.0029 | 0.0105 | 0.0132 |
| 2D Shallow water | 0.0042 | 0.0004 | 0.0014 | 0.0060 |
| 3D CFD | 0.4329 | 0.3666 | 0.4531 | 0.4642 |
| 3D Maxwell | 0.4564 | 0.4552 | 0.2268 | 1.4089 |
| 3D Maxwell, fine-tuned ² | 0.2971 | 0.2963 | 0.2268 | 1.4089 |

¹ The single 2D CFD file used in Table 1, `Rand_M1.0_Eta0.1_Zeta0.1_128`; the mean over all eight 2D CFD files is 0.1324.  
² `omniarch-large-pt-maxwell.pt`: the released checkpoint fine-tuned for 10k steps on Maxwell only, as the original code fine-tuned per data set; the other columns use the pre-trained checkpoint.

nRMSE of the next-frame prediction on the in-domain test split (lower is better). *Recovered* is the same error after the per-file affine map `x*std+mean` that the original validation code applied to prediction and target before computing the error; the paper does not state which of the two it reports. Over all eight 2D CFD files, copying the last frame gives 0.3070 (model: 0.1324). 2D incompressible Navier-Stokes (2.3 TB) is not evaluated.

3D Maxwell: the pre-trained checkpoint gives 0.4564. Fine-tuning it for 10k steps on Maxwell alone, as the original code did per data set (`configs/finetuning`), gives 0.2971 (`omniarch-large-pt-maxwell.pt`, https://drive.google.com/file/d/1O9Mw74cCRtCRcTwglpkod8Q44L45whUY/view). The Maxwell test set is the 10 % split of all PDEArena files concatenated in sorted order.

## Evaluation protocol

The protocol follows the original evaluation code:

- each data file is split 90/10 with `torch.utils.data.random_split` and seed 42; the 10 % part is the test set and is never seen in training;
- grids are those of pre-training: 1024 points in 1D, 128 × 128 in 2D (512 × 512 files subsampled by 4), 64³ in 3D (128³ files subsampled by 2; Maxwell upsampled from 32³);
- given the true frames 0 to T−1, the model predicts frame T in one step;
- nRMSE = ‖pred − true‖₂ / ‖true‖₂ over space, per field, averaged over fields and samples; a family is the mean over its files.

The full evaluation takes about 20 minutes on one RTX 5090. `--max-samples 16` gives a quick check in about a minute.

## Data

- PDEBench: the 1D, 2D and 3D files of Table 1 except 2D incompressible Navier-Stokes, read directly from the original HDF5 download.
- 3D Maxwell: `python scripts/fetch_maxwell.py --out /path/to/maxwell` downloads PDEArena Maxwell-3D from Hugging Face (121 GB, deleted file by file after conversion) and writes `Maxwell3D_pdearena_32.h5` (60 GB, all splits concatenated in sorted file order, as in the original code).

## Training

```bash
python scripts/pack_fp16.py --pdebench /path/to/PDEBench --out data/packed        # optional, faster loading
torchrun --standalone --nproc_per_node 4 scripts/train_paper.py \
    --pdebench /path/to/PDEBench /path/to/maxwell --packed data/packed --out runs/large \
    --steps 40000 --residual --balance 0.5 --max-stride 1 1 4
```

The recipe of the released checkpoint:

| | |
|---|---|
| Model | OmniArch-L: Fourier encoder and decoder (12 modes, width 8), 24-layer Transformer with block-causal attention over time |
| Objective | nRMSE of all next frames; the decoder predicts the change from the input frame (the residual option of the original training code) |
| Batch | 42 / 3 / 1 trajectories per GPU for 1D / 2D / 3D, one step per dimension, 4 GPUs |
| Optimiser | AdamW, learning rate 1e-4 with 500 warm-up steps and cosine decay to 1e-6, gradient clipping 0.5, input noise 1e-4 |
| Sampling | PDE families drawn in proportion to the square root of their number of trajectories; 3D frame spacing drawn from 1 to 4 |
| Schedule | 5k steps with uniform sampling, then 40k steps with the recipe above; Maxwell joined after the first 5k steps of the 40k-step run; the Maxwell checkpoint is then fine-tuned for 10k steps on Maxwell alone (`--only "3D Maxwell" --batch 42 3 2`) |
| Compute | 4 × RTX 5090 (32 GB): about 13 hours for pre-training and 1.5 hours for the Maxwell fine-tuning |

## Loading the model in Python

```python
from omniarch.paper import PaperPredictor

p = PaperPredictor("weights/omniarch-large-pt.pt")
nxt = p.next_frame(history)               # history: (K, C, *grid) -> next frame (C, *grid)
future = p.predict(history, steps=10)     # autoregressive rollout
```

Fields keep the order of pre-training, e.g. (Vx, density, pressure) in 1D and (Vx, Vy, density, pressure) in 2D compressible flow. Without `grid`, cell-centred coordinates on [0, 1] are used; `scripts/reproduce_paper.py` passes the coordinates stored in each PDEBench file.
