# MRSM: generative virtual sensing of physical fields

Code for **Generative virtual sensing of continuous physical fields from sparse measurements**.
MRSM combines denoising score matching with Score Fokker–Planck (FPE) regularization.
Inference uses stochastic diffusion sampling, implicit score sampling (ISS), or
physics-guided ISS (PI-ISS). The examples cover acoustic free fields, standing-wave
tubes, a measured virtual array, Kolmogorov flow and ERA5 weather fields.

- [Paper on arXiv](https://arxiv.org/abs/2605.22338)
- [Data and model downloads](https://doi.org/10.5281/zenodo.23938017)
- [Case settings and notebook instructions](docs/cases.md)
- [Data files and model packages](docs/data.md)
- [Training and sampling commands](docs/running.md)
- [Tests and reference results](docs/validation.md)

## Installation

Use Python 3.12. For an NVIDIA GPU with a driver supporting CUDA 11.8:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cu118.txt
python -m pip install -r requirements-notebooks.txt
```

For CPU code checks, install PyTorch from its CPU wheel index first, followed by
`requirements.txt`. Run commands from the repository root. GPU sampling requires
the case data, configuration, normalization statistics and model checkpoint
listed in [docs/data.md](docs/data.md).

## Examples

The root notebooks provide free-field, Kolmogorov and ERA5 examples. Additional
notebooks under `notebooks/` include measured standing-wave and virtual-array
experiments. Their first cell selects the repository root and device. Each
notebook can be run in its own fresh kernel.

The free-field evaluators support random observations and regular spatial
downsampling. For one regular-downsampling case:

```bash
python scripts/evaluate_freefield_manuscript_protocol.py \
  --checkpoint source=results/2DSound_v22/checkpoint_200.pth \
  --device cuda:0 --scales 6 --starts 95 --methods ISS \
  --output outputs/free_field_6x.json
```

The full regular grid is selected with `--scales 2,3,4,5,6`,
`--starts 95,15,30,158,128` and `--methods SDE,ISS,PI-ISS`.
FNO and LNO each use a separately trained checkpoint at each downsampling factor.
Their training sources and the per-factor comparisons are provided with the
case notebooks and baseline scripts.

## Training

Train the score model from a case configuration:

```bash
python train.py --config results/2DSound_v22/config.json \
  --data data/anechoic_high_prec_dataset_last700_2.npy \
  --output-dir outputs/free_field_dsm --device cuda:0 \
  --batch-size 1 --total-steps 1000
```

Fine-tune an existing checkpoint with Score FPE regularization:

```bash
python train_FT.py --config results/2DSound_v22/config.json \
  --checkpoint results/2DSound_v22/checkpoint_200.pth \
  --data data/anechoic_high_prec_dataset_last700_2.npy --lr 2e-6 \
  --output-dir outputs/free_field_fpe --device cuda:0 \
  --batch-size 1 --total-steps 1000
```

`--total-steps` counts optimizer updates. Reverse sampling steps are configured
separately in the sampling commands and notebooks. Case configurations and
the checkpoints used for evaluation are supplied in the model packages.
Fine-tuning writes to a new output directory.
Use `--resume` with the same output directory to continue a saved training run.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover checkpoint compatibility, the analytic Gaussian Score FPE
residual, gradient propagation and deterministic training continuation.

## Citation

arXiv preprint:

```bibtex
@article{zhu2026physicsinformed,
  title   = {Physics-Informed Generative Solver: Bridging Data-Driven Priors and Conservation Laws for Stable Spatiotemporal Field Reconstruction},
  author  = {Zhu, Ziyuan and Hu, Keyu and Chen, Zhifei and Shi, Yuhao and Bao, Ming and Zhao, Jing and Wang, Gang and Xu, Haitan and Li, Jiadong and Zhao, Qijun and Li, Xiaodong and Lu, Minghui and Chen, Yanfeng},
  journal = {arXiv preprint arXiv:2605.22338},
  year    = {2026}
}
```

## License

Code is distributed under the [MIT License](LICENSE). Data and weights follow
the license on their Zenodo record. ERA5 use also follows the Copernicus data
license and the source attribution described in the data documentation.
