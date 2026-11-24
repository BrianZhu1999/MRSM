# Data and model files

The published data and model checkpoints are available at
[Zenodo](https://doi.org/10.5281/zenodo.19398314).

| Archive | Contents |
| --- | --- |
| `weights_free_field.zip` | Free-field score model |
| `weights_kolmogorov.zip` | DSM and FPE flow models |
| `weights_era5.zip` | ERA5 score model |
| `weights_standing_wave.zip` | Pressure and pressure–velocity tube models |
| `weights_line_array.zip` | Simulated-array, measured-array and FPE models |
| `weights_baselines.zip` | FNO and LNO models |
| `acoustic_free_field.npy` | Free-field training data |
| `data_free_field_test.zip` | Free-field test data |
| `data_kolmogorov_train.zip.001`–`.009` | Flow training data |
| `data_kolmogorov_test.zip` | Flow test data |
| `data_era5.zip` | ERA5 training and test data |
| `data_standing_wave.zip` | Tube training data and measured recordings |
| `data_line_array.zip` | Twelve Figure 5 conditions, coordinates and FDTD training data |
| `data_line_array_simulation.zip` | Simulated-array training data |
| `figure5_results.zip` | Figure 5 predictions, reference fields and result tables |

The Kolmogorov training archive is distributed as nine consecutive binary
volumes, together with `README_KOLMOGOROV_ARCHIVE.txt` and
`merge_kolmogorov_archive.py`. Run the merger in the download directory to
reconstruct and verify `data_kolmogorov_train.zip`.

Download the relevant archives and extract them into the repository root. The
archives create `data/`, `results/<model>/` and the FNO/LNO result directories.
Every model package includes its configuration and applicable normalization statistics.
The updated standing-wave and measured-array data ZIPs contain the NumPy
arrays. Their dimensions, variables and processing are described in
[Case notebooks](cases.md) and the corresponding notebooks.

`acoustic_free_field.npy` is the free-field **training** array. Place it at
`data/anechoic_high_prec_dataset_last700_2.npy`. The separate test array is
supplied in `data_free_field_test.zip`.

## Model selection

| Case | Model directory | Checkpoint |
| --- | --- | --- |
| Free field | `results/2DSound_v22` | `checkpoint_200.pth` |
| Kolmogorov, DSM | `results/kol_v1` | `checkpoint_517.pth` |
| Kolmogorov, Score FPE | `results/kol_v1_finetune_FPE` | `checkpoint_517.pth` |
| ERA5 | `results/era5_v0` | `checkpoint.pth` |
| Standing wave, pressure and velocity | `results/1dtube_v31` | `checkpoint.pth` |
| Standing wave, pressure held out | `results/exp_p_v21` | `checkpoint.pth` |
| Figure 5 line array | `results/line_array_fpe` | `checkpoint.pth` |
| Measured-array pretraining | `results/line_array_v33` | `checkpoint_600.pth` |
| Simulated line array | `results/line_array_v3` | `checkpoint.pth` |

The case notebooks use the checkpoints listed above. The
Kolmogorov package includes both training stages for the regularization
comparison. The FNO/LNO package contains separate 2×, 3×, 4×, 5× and 6× models
for each operator and each of the free-field and Kolmogorov cases.

## Normalization

Load the statistics accompanying the selected model. Free-field data contain
pressure and two particle-velocity components. Standing-wave and array examples
specify their channel and impedance transformations in the corresponding
notebook. ERA5 arrays have already been normalized; `scalars_era5.npy` supplies
the inverse transformation to physical variables. Keep the observation masks,
frequency selection and normalization order given by each case notebook.

The CSV files under `source_data/` contain the free-field observation-rate and
regular-downsampling results. [Figure statistics](cases.md#figure-statistics)
describes their aggregation and filtering.

For Figure 5, `data/figure5/localization/` and `data/figure5/heldout/` contain
four-channel fields with shape `(7,4,1002)`. Pressure is in Pa and velocity is
in m/s. Filenames use `<frequency-pair>_<configuration>`, with configurations
`A`, `B`, `C` and `D` (for example, `2800_3000_A.npz`). `coordinates.npz` gives
physical and virtual coordinates in metres.
`figure5_results.zip` extracts predictions and tables to `source_data/figure5/`.
The array paths and sampling settings are specified in
[`configs/figure5_line_array.json`](../configs/figure5_line_array.json).

ERA5 is supplied by ECMWF through the Copernicus Climate Change Service. The
processed subset uses eastward wind, northward wind, temperature and pressure.
The original data-source citation and Copernicus terms continue to apply.
