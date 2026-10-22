<p align="center"><a href="https://cnai4s.com"><img src="docs/cnai4s_logo.png" alt="CNAI4S" height="56"></a></p>

<h1 align="center">OmniArch-pt (智算天璇)</h1>

<p align="center">One set of weights forecasts 1D, 2D and 3D PDEs. PyTorch rewrite of the ICML 2025 OmniArch model.</p>

<p align="center">
  <a href="https://arxiv.org/abs/2402.16014"><img src="https://img.shields.io/badge/arXiv-2402.16014-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <a href="https://github.com/Tarpelite/omniarch_pt/stargazers"><img src="https://img.shields.io/github/stars/Tarpelite/omniarch_pt?style=social" alt="GitHub stars"></a>
</p>

A PyTorch re-implementation of **OmniArch**, a foundation model that forecasts 1D, 2D and 3D PDEs with one set of weights. Give it the first few frames of a simulation; it rolls the solution forward on its own.

> **Preview release (v0.3).** The checkpoints here are models retrained from scratch on a subsampled PDEBench. They are meant for deployment and demos, not as a reproduction of the paper's model or numbers. We release new results and checkpoints as they come, usually daily.

## News

- **2026-10-22, v0.3.** Joint 1D+2D+3D training cuts the 3D error by 17% when 3D data is scarce ([study](docs/studies.md#cross-dimension-training)). New `omniarch-mini-anystep` checkpoint for data with a different frame spacing ([study](docs/studies.md#frame-spacing)). Training 1.5x faster per job ([details](docs/training.md#speed)).
- **2026-10-22, v0.2.** Public release: `base`, `mini` and `tiny` checkpoints from a 36-run [scaling study](docs/studies.md#scaling), web demo, HTTP API.
- **2026-10-21, v0.1.** First PyTorch rewrite with a command line tool and web demo.

## How it works

![OmniArch architecture](docs/architecture.png)

<sub>Architecture figure from the ICML 2025 paper.</sub>

Every physical field at every time step becomes one token. A Fourier encoder per dimension (1D, 2D, 3D) turns a field into a token; a shared transformer, causal in time, predicts the next tokens; a Fourier decoder turns them back into fields. Fields are told apart by their order, not their names, so one model takes any mix of physical quantities and grid sizes.

## Quick start

```bash
git clone https://github.com/Tarpelite/omniarch_pt.git && cd omniarch_pt
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build for your GPU
pip install -e .
# download a checkpoint into ./weights/ (see Weights below)

omniarch info                      # model, device, supported quantities
omniarch examples                  # bundled cases
omniarch predict 2d_turbulence --gif turb.gif
omniarch serve                     # web demo at http://127.0.0.1:7860
```

```python
from omniarch import OmniArchPredictor
p = OmniArchPredictor("weights/omniarch-base.pt")
future = p.predict(history, steps=10)      # history: (K, C, *grid), K >= 4 frames recommended
```

More in [docs/usage.md](docs/usage.md).

## Weights

Default checkpoints (from v0.2):

| Checkpoint | Params | Training | Download (BUAA Pan, valid until 2027-10-22) |
|---|---|---|---|
| `omniarch-base.pt` (1.2 GB) | 308 M | 36k steps | https://bhpan.buaa.edu.cn/link/AA645380522EB54391B3A496443C6CD709 |
| `omniarch-mini.pt` (293 MB) | 77 M | 72k steps | https://bhpan.buaa.edu.cn/link/AA00790700453A44E893F3130C7DD2B618 |
| `omniarch-tiny.pt` (54 MB) | 14 M | 108k steps | https://bhpan.buaa.edu.cn/link/AA5516D6EE8EF74748AD809475482F5340 |

v0.3 adds one checkpoint for data whose frame spacing differs from training or is unknown:

| Checkpoint | Params | Training | Download (BUAA Pan, valid until 2027-10-22) |
|---|---|---|---|
| `omniarch-mini-anystep.pt` (293 MB) | 77 M | 36k steps, frame spacing 1-4 | https://bhpan.buaa.edu.cn/link/AAE32DDE73BDBF4EB19B5066189F1AA90A |

Load it with `--ckpt weights/omniarch-mini-anystep.pt` (CLI) or `OmniArchPredictor("weights/omniarch-mini-anystep.pt")`. On the standard benchmark it is 13% behind a regular `mini` trained for the same 36k steps (0.204 vs 0.180); see [Frame spacing](docs/studies.md#frame-spacing) for what it buys.

`omniarch` looks for `weights/omniarch-base.pt`, then `-mini`, then `-tiny`. v0.1 checkpoints (mini 0.179, tiny 0.243) remain available: [mini](https://bhpan.buaa.edu.cn/link/AAA0212103021B4601B97DA21AE1A37922), [tiny](https://bhpan.buaa.edu.cn/link/AA4BD26995FE344D5EA1622A3F6ECECC6F) (valid until 2027-01-01).

## Results at a glance

PDEBench protocol: 10 given frames, roll out to the end of the trajectory, nRMSE over all forecast frames (lower is better), mean over 9 PDE families in 1D, 2D and 3D.

| OmniArch base | OmniArch mini | OmniArch tiny | Copy last frame |
|---|---|---|---|
| **0.159** | 0.168 | 0.212 | 0.459 |

Per-family numbers and zero-shot cases: [docs/results.md](docs/results.md).

## Roadmap

New results and checkpoints are released as they come, usually daily.

- [x] PyTorch rewrite, CLI, web demo, HTTP API (v0.1)
- [x] `base`, `mini`, `tiny` checkpoints and scaling study (v0.2)
- [x] Cross-dimension study, frame-spacing-robust checkpoint, faster training (v0.3)
- [ ] Larger checkpoints, towards about 1B parameters
- [ ] More training data beyond PDEBench
- [ ] Higher-resolution grids
- [ ] Problems beyond periodic boxes
- [ ] Better zero-shot use on new problems

## Documentation

- [Usage](docs/usage.md): Python and HTTP APIs, your own data, how many frames to give
- [Results](docs/results.md): per-family benchmark and zero-shot cases
- [Studies](docs/studies.md): scaling, cross-dimension training, frame spacing
- [Training](docs/training.md): data preparation, training, speed, code layout
- [部署与使用指南](docs/deploy_guide.md) (Chinese deployment guide)

## Citation

```bibtex
@inproceedings{chen2025omniarch,
  title     = {OmniArch: Building Foundation Model For Scientific Computing},
  author    = {Chen, Tianyu and Zhou, Haoyi and Li, Ying and Wang, Hao and Gao, Chonghan and Shi, Rongye and Zhang, Shanghang and Li, Jianxin},
  booktitle = {International Conference on Machine Learning},
  year      = {2025}
}
```

Data: PDEBench (Takamoto et al., NeurIPS 2022 Datasets and Benchmarks), CC BY 4.0.

## Contact

Tianyu Chen, tianyuc@buaa.edu.cn. Questions, bug reports and collaboration are welcome; GitHub issues work too.

Part of [CNAI4S](https://cnai4s.com) (人工智能科学计算共性平台).

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=Tarpelite/omniarch_pt&type=Date)](https://www.star-history.com/#Tarpelite/omniarch_pt&Date)

## License

Code: MIT. Bundled example cases are derived from PDEBench (CC BY 4.0). The architecture figure is from the OmniArch paper.
