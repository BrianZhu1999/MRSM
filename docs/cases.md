# Case notebooks

Open a notebook with the repository as its working directory and run its cells in order. The first code cell finds the repository root, sets relative data paths and selects `cuda:0` when CUDA is available. Checkpoint numbers name saved training states. The reverse-diffusion sampling counts are listed separately below. The three root notebooks are entry copies of the corresponding reconstruction notebooks.

Install data and model files at the paths listed in [data.md](data.md). Notebook outputs are written under `outputs/<case>/`. Dependencies are listed in the repository requirements files.

## Run order and settings

| Case | Notebook | Checkpoint directory | Reverse steps | Main setting and output |
| --- | --- | --- | --- | --- |
| Free-field example | `notebooks/free_field/reconstruction.ipynb` | `results/2DSound_v22`, checkpoint 200 | 100 | 375 Hz, regular 6× sampling; SDE, ISS, PI-ISS, FNO and LNO; channel errors and physical residuals |
| Free-field random measurements | `notebooks/free_field/random_observations.ipynb` | `results/2DSound_v22`, checkpoint 200 | 100 | 2, 3, 5, 8, 10, 15 and 20% observed positions; five seeds; nRMSE and Pearson correlation |
| Free-field spatial downsampling | `notebooks/free_field/downsampling.ipynb` | `results/2DSound_v22`, checkpoint 200 | 100 | 375 Hz, 2–6× downsampling; five selected time windows; FNO/LNO weights matched to each scale |
| Free-field plots from CSV | `notebooks/free_field/source_figures.ipynb` | — | — | Plot the two source-data CSV tables |
| Pressure-only tube | `notebooks/tube/pressure.ipynb` | `results/exp_p_v21` | 500 | 395 Hz; seven of 16 positions observed; nine positions withheld |
| Pressure–velocity tube | `notebooks/tube/pressure_velocity.ipynb` | `results/1dtube_v31` | 100 | 385 Hz; 12 valid positions; pressure observations and reconstructed velocity by default |
| Measured line array | `notebooks/line_array/reconstruction_music.ipynb` | `results/line_array_v33`, checkpoint 600 | 500 | Seven sensors, 32 virtual positions; full-vector MUSIC at 2950 Hz |
| Kolmogorov reconstruction and spectrum | `notebooks/kolmogorov/reconstruction_spectrum.ipynb` | `results/kol_v1_finetune_FPE`, checkpoint 517 | 100 | Regular 6× sampling; velocity error and kinetic-energy spectrum |
| Kolmogorov spatial downsampling | `notebooks/kolmogorov/downsampling.ipynb` | `results/kol_v1_finetune_FPE`, checkpoint 517 | 100 | 2–6× downsampling and five seeds |
| ERA5 reconstruction | `notebooks/era5/reconstruction.ipynb` | `results/era5_v0` | 50 | 5% observed positions; four channels |
| ERA5 reconstruction error | `notebooks/era5/nrmse.ipynb` | `results/era5_v0` | 50 | 2–20% observed positions; per-channel nRMSE across five sampling seeds |
| ERA5 correlation | `notebooks/era5/pearson.ipynb` | `results/era5_v0` | 50 | Same rates and seeds; per-channel Pearson correlation |

## Input arrays and normalization

| Input | Stored shape | Meaning and normalization |
| --- | --- | --- |
| `data/anechoic_high_prec_dataset_last700_test_2.npy` | `(9,700,64,64,3)` | Sample, physical time, two spatial axes, pressure/velocity-x/velocity-y. The notebook transposes channels and uses `results/2DSound_v22/scalar.npy`. |
| `data/tube395_resampled_calibrated.npy` | `(1,100000,1,16)` | Sample, physical time, pressure, sensor. The pressure notebook standardizes the measurement during loading and again before sampling. It applies one inverse transform to predictions and compares against the once-standardized reference. |
| `data/pu385.npy` | `(1,50000,2,15)` | Sample, time, pressure/velocity, sensor. Valid indices are `[1,3,4,5,6,7,8,10,11,12,13,14]`. Means and standard deviations are calculated separately by channel over the full measured record at those positions. |
| `data/Aligned_2900_3000_pos5.npy` | `(7,4,5000)` | Sensor, pressure/three velocity channels, time. Velocity channels are multiplied by acoustic impedance before reconstruction and divided by the same impedance afterward. |
| `data/kolmogorov/kolmogorov_flow_test.npy` | `(20,400,64,64,2)` | Trajectory, time, spatial axes, two velocity components. The notebook reshapes these into five Reynolds-number groups, uses the checkpoint scalar, and appends forcing and Reynolds-number conditioning channels. |
| `data/era5/era5_test.npy` | `(1,273,64,64,4)` | Sample, time, spatial axes, zonal wind/meridional wind/temperature/pressure. The supplied fields are already standardized. |

Sampling uses 10-frame physical-time windows with two overlapping frames; requests for 100 physical frames produce 106 frames. The tube pressure–velocity example requests 200 physical frames. The line-array example requests 1000 physical frames. Reverse-diffusion steps are listed separately in the table above.

## Sampling and metrics

The free-field demonstration uses PI-ISS observation factor 1 and equal acoustic residual factors. The multi-trial evaluation uses observation factor 1.4 and multiplies continuity and momentum residual weights by 1e8. Both use the base factor 5e-8 and 100 reverse steps.

For pressure-only tube reconstruction, `evaluate_fpe_sampling.py --case tube-pressure-heldout` observes ten positions and evaluates six held-out positions. The `pressure.ipynb` example observes seven positions and holds out nine.

The pressure–velocity tube notebook uses `MASK_SECOND_CHANNEL=True` and `sensors_to_hide=None`: all 12 valid pressure positions are observed, and all velocity measurements are masked. The per-channel error table evaluates those positions using checkpoint `1dtube_v31`.

The measured line-array notebook adds 10 dB Gaussian noise before reconstruction, using the current NumPy random state. Its seven-sensor MUSIC reference uses the measurement before this noise addition. PI-ISS uses 500 reverse steps. The local interference region is x from −0.6 to 0.6 m and z from 1.6 to 1.8 m. Both MUSIC calculations use pressure and all three velocity channels at 2950 Hz.

Kolmogorov energy is calculated with NumPy's unnormalized two-dimensional Fourier transform. Squared Fourier amplitudes are summed within unit-width radial shells, multiplied by one half, and averaged over physical time frames. The Kolmogorov notebooks use FPE checkpoint 517.

The ERA5 reconstruction example uses 5% observations and SDE observation factor 0.2. The observation-rate comparison uses SDE factor 0.1 and ISS factor 0.3, each divided by the square root of the observation fraction. Sampling noise varies over seeds `[1,42,123,456,789]`; the observation-mask seed is fixed. Standard deviations use NumPy's population convention (`ddof=0`).

## Figure statistics

The two free-field CSV tables are described in [source_data/README.md](../source_data/README.md). `source_figures.ipynb` plots observation rates of 3, 5, 8, 10, 15 and 20%; `random_observations.ipynb` also displays 2%. In the observation-rate panels, means and standard deviations are each averaged across the three physical channels.

For the downsampling panel, `source_figures.ipynb` omits values above Q3 + 1.5 times the interquartile range. Means and population standard deviations use four trials for LNO and FNO at 4× and 5× downsampling and SDE at 6×; the other combinations use five trials. The CSV contains all five trial values per combination.

The downsampling evaluator aggregates errors below 999; the value 999 marks a failed evaluation.
