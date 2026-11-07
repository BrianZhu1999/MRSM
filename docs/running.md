# Training and sampling

Run these commands from the repository root after extracting the model and data
packages described in [data.md](data.md). Replace `cuda:0` with the desired local
GPU. The case notebooks give the sampling settings and plotting commands.

## Free-field comparisons

```bash
python scripts/evaluate_freefield_manuscript_protocol.py \
  --checkpoint source=results/2DSound_v22/checkpoint_200.pth \
  --device cuda:0 --output outputs/free_field_regular.json

python scripts/evaluate_freefield_observation_curve_protocol.py \
  --checkpoint source=results/2DSound_v22/checkpoint_200.pth \
  --device cuda:0 --output outputs/free_field_random.json
```

The regular grid uses 2–6× downsampling and five temporal starts. The random
observation grid uses the six observation proportions in the paper. Both
commands use the sampling seed and model mode specified in the notebooks. The CSVs under
`source_data/` provide the numerical comparison values.

## Case sampling and paired checkpoint comparisons

```bash
python scripts/evaluate_fpe_sampling.py --case tube \
  --config results/1dtube_v31/config.json \
  --checkpoint source=results/1dtube_v31/checkpoint.pth \
  --device cuda:0 --output-dir outputs/tube --save-predictions

python scripts/evaluate_fpe_sampling.py --case line-measured \
  --config results/line_array_v33/config.json \
  --checkpoint source=results/line_array_v33/checkpoint_600.pth \
  --methods pi-iss --device cuda:0 --output-dir outputs/line --save-predictions

python scripts/evaluate_fpe_sampling.py --case era5 \
  --config results/era5_v0/config.json \
  --checkpoint source=results/era5_v0/checkpoint.pth \
  --device cuda:0 --output-dir outputs/era5 --save-predictions
```

Repeat `--checkpoint name=path` to compare two models using paired inputs and
random seeds. `--max-trials 1 --reverse-steps 2` selects a short implementation
check; omit these overrides for the case evaluation. The measured-array
comparison command fixes its noise seed. See [case settings](cases.md#sampling-and-metrics)
for the figure notebook's noise and reference measurements.

`--power-ceiling-w` sets a runtime GPU power threshold for the evaluation
scripts. `--forward-sleep-seconds` in the multi-case evaluator and
`--step-sleep-seconds` in the free-field evaluator reduce the duty cycle when
needed. GPU indices refer to the devices visible to the process.

## Optimization settings

The `train.py` and `train_FT.py` commands in the README accept the same case
configuration. The FPE stage additionally loads an existing score checkpoint.
Use `--help` for learning rate, batch size, regularization and continuation
options. The data window is shared by the denoising and FPE losses, while their
diffusion times and Gaussian perturbations are sampled independently.

Use the checkpoints in the model packages to reproduce the reported figures.
