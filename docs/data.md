# Data and model files

The published record is [Zenodo 19398315](https://doi.org/10.5281/zenodo.19398315).
It contains the free-field training array and the original acoustic and
Kolmogorov model archives. The complete data and model packages for the
repository examples are listed below and are supplied through the Zenodo
release associated with this repository.

| Archive | Size (GB) |
| --- | ---: |
| `weights_free_field.zip` | 0.302 |
| `weights_kolmogorov.zip` | 0.605 |
| `weights_era5.zip` | 0.302 |
| `weights_standing_wave.zip` | 0.550 |
| `weights_line_array.zip` | 0.103 |
| `weights_baselines.zip` | 0.741 |
| `data_free_field_test.zip` | 0.310 |
| `data_kolmogorov_train.zip.001`–`.009` | 9.175 |
| `data_kolmogorov_test.zip` | 0.524 |
| `data_era5.zip` | 2.142 |
| `data_standing_wave.zip` | 0.764 |
| `data_line_array.zip` | 0.308 |
| `data_line_array_simulation.zip` | 2.048 |

The Kolmogorov training archive is distributed as nine consecutive binary
volumes, together with `README_KOLMOGOROV_ARCHIVE.txt` and
`merge_kolmogorov_archive.py`. Run the merger in the download directory to
reconstruct and verify `data_kolmogorov_train.zip`.

Download the relevant archives and extract them into the repository root. The
archives create `data/`, `results/<model>/` and the FNO/LNO result directories.
Every model package includes its configuration and normalization statistics.
Every data package includes an English description of array dimensions,
variables and processing. The numerical data in the archives are unchanged.

The existing `acoustic_free_field.npy` is the free-field **training** array.
Place it at `data/anechoic_high_prec_dataset_last700_2.npy`. The separate test
array is supplied in `data_free_field_test.zip`.

## Model selection

| Case | Model directory | Checkpoint |
| --- | --- | --- |
| Free field | `results/2DSound_v22` | `checkpoint_200.pth` |
| Kolmogorov, DSM | `results/kol_v1` | `checkpoint_517.pth` |
| Kolmogorov, Score FPE | `results/kol_v1_finetune_FPE` | `checkpoint_517.pth` |
| ERA5 | `results/era5_v0` | `checkpoint.pth` |
| Standing wave, pressure and velocity | `results/1dtube_v31` | `checkpoint.pth` |
| Standing wave, pressure held out | `results/exp_p_v21` | `checkpoint.pth` |
| Standing wave, high frequency | `results/1dtube_v41` | `checkpoint.pth` |
| Measured line array | `results/line_array_v33` | `checkpoint_600.pth` |
| Simulated line array | `results/line_array_v3` | `checkpoint.pth` |

The model packages follow the retained figure-generating notebooks. The
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

The small CSV files under `source_data/` contain the recorded free-field
observation-rate and regular-downsampling results. The plotting notebook
contains their aggregation and upper-tail filtering steps.

ERA5 is supplied by ECMWF through the Copernicus Climate Change Service. The
processed subset uses eastward wind, northward wind, temperature and pressure.
The original data-source citation and Copernicus terms continue to apply.
