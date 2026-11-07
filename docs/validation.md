# Tests and reference results

## Software tests

Use Python 3.12 with the dependencies in `requirements-cu118.txt` and
`requirements-notebooks.txt`, then run:

```bash
python -m pip check
python -m unittest discover -s tests -v
```

The tests cover checkpoint loading, EMA compatibility, the analytic Gaussian
FPE residual and gradient, baseline input layouts, and training continuation
across an epoch boundary.

## Free-field reference values

The two evaluations below use 375 Hz, 6× downsampling, temporal start 95 and
100 reverse steps, with the model mode and seed specified by the evaluator.
The metric is the frame-averaged relative L2 error.

| Method | Recomputed nRMSE | Recorded nRMSE | Absolute difference |
| --- | ---: | ---: | ---: |
| ISS | 0.081832844302 | 0.08183285 | 5.7e-09 |
| PI-ISS | 0.030888171969 | 0.03088817 | 1.97e-09 |

The CSV files in `source_data/` contain the full observation-rate and
downsampling comparisons. Commands for evaluating those settings are in
[running.md](running.md).

## Data and checkpoint checks

After extracting all data and model packages into the repository root, run:

```bash
python tests/check_release_assets.py --output outputs/asset_check.json
```

This checks array headers and strictly loads each score checkpoint on CPU.
