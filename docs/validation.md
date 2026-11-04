# Release checks (4 November 2026)

The public code was checked in a separate Python 3.12 environment with the
versions in `requirements-cu118.txt` and `requirements-notebooks.txt`.
`pip check` completed successfully.

- Ten CPU tests passed: strict checkpoint loading, EMA compatibility, the
  analytic Gaussian FPE residual and gradient, baseline input layouts and
  training continuation across an epoch boundary.
- Nine released score checkpoints loaded with matching parameter names and
  shapes; all model and EMA parameters were finite.
- Thirty-one data and normalization arrays matched their documented shapes.
- Eleven reconstruction/evaluation notebooks executed their imports, model
  loading and input preparation on CPU before the sampling cell. All fifteen
  notebook files passed code-cell parsing and compilation.
- The tube pressure–velocity, pressure-only tube, measured line-array and ERA5
  evaluators completed one short trial with two reverse steps and finite
  outputs. These checks exercised data, checkpoint and sampler interfaces.

## Recorded free-field results

Two calculations used 375 Hz, 6× downsampling, temporal start 95 and 100 reverse
steps, preserving the recorded model mode and seed. The frame-averaged relative
L2 error was compared with the original CSV row:

| Method | Recomputed nRMSE | Recorded nRMSE | Absolute difference |
| --- | ---: | ---: | ---: |
| ISS | 0.081832844302 | 0.08183285 | 5.7e-09 |
| PI-ISS | 0.030888171969 | 0.03088817 | 1.97e-09 |

The full observation-rate and multi-seed grids are provided as runnable
notebooks and scripts. This release check evaluated the two recorded conditions
above and the short case-interface runs; the remaining grid values are the
original stored results.

## Downloaded-asset check

After extracting all data and model packages into the repository root, run:

```bash
python tests/check_release_assets.py --output outputs/asset_check.json
```

This checks array headers and strictly loads each score checkpoint on CPU.
