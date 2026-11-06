# Case notebooks

Open a notebook with the repository as its working directory and run its cells in order. The first code cell finds the repository root, sets relative data paths and selects `cuda:0` when CUDA is available. Checkpoint numbers name saved training states. The reverse-diffusion sampling counts are listed separately below. The three root notebooks are entry copies of the corresponding reconstruction notebooks.

Install data and model files at the paths supplied by the release asset collection. Notebook outputs are written under `outputs/<case>/`. Plotting uses Arial and compact numerical tick labels. Core dependencies are Python, PyTorch, NumPy, SciPy, pandas, matplotlib, einops, tqdm and Jupyter. Model dependencies are listed in the repository environment file.

## Run order and settings

| Case | Notebook | Checkpoint directory | Reverse steps | Main setting and output |
| --- | --- | --- | --- | --- |
| Free-field example | `notebooks/free_field/reconstruction.ipynb` | `results/2DSound_v22`, checkpoint 200 | 100 | 375 Hz, regular 6× sampling; SDE, ISS, PI-ISS, FNO and LNO; channel errors and physical residuals |
| Free-field random measurements | `notebooks/free_field/random_observations.ipynb` | `results/2DSound_v22`, checkpoint 200 | 100 | 2, 3, 5, 8, 10, 15 and 20% observed positions; five seeds; nRMSE and Pearson correlation |
| Free-field spatial downsampling | `notebooks/free_field/downsampling.ipynb` | `results/2DSound_v22`, checkpoint 200 | 100 | 375 Hz, 2–6× downsampling; five selected time windows; FNO/LNO weights matched to each scale |
| Original free-field panels | `notebooks/free_field/source_figures.ipynb` | No model loading | — | Plot the two stored source-data CSV tables |
| Pressure-only tube | `notebooks/tube/pressure.ipynb` | `results/exp_p_v21` | 500 | 395 Hz; seven of 16 positions observed; nine positions withheld |
| Pressure–velocity tube | `notebooks/tube/pressure_velocity.ipynb` | `results/1dtube_v31` | 100 | 385 Hz; 12 valid positions; pressure observations and reconstructed velocity by default |
| Measured line array | `notebooks/line_array/reconstruction_music.ipynb` | `results/line_array_v33`, checkpoint 600 | 500 | Seven sensors, 32 virtual positions; full-vector MUSIC at 2950 Hz |
| Kolmogorov reconstruction and spectrum | `notebooks/kolmogorov/reconstruction_spectrum.ipynb` | `results/kol_v1_finetune_FPE`, checkpoint 517 | 100 | Regular 6× sampling; velocity error and kinetic-energy spectrum |
| Kolmogorov spatial downsampling | `notebooks/kolmogorov/downsampling.ipynb` | `results/kol_v1_finetune_FPE`, checkpoint 517 | 100 | 2–6× downsampling and five seeds |
| ERA5 reconstruction | `notebooks/era5/reconstruction.ipynb` | `results/era5_v0` | 50 | 5% observed positions in the source demonstration; four channels |
| ERA5 reconstruction error | `notebooks/era5/nrmse.ipynb` | `results/era5_v0` | 50 | 2–20% observed positions; per-channel nRMSE across five sampling seeds |
| ERA5 correlation | `notebooks/era5/pearson.ipynb` | `results/era5_v0` | 50 | Same rates and seeds; per-channel Pearson correlation |

## Input arrays and normalization

Shapes below were read from the distributed NumPy array headers.

| Input | Stored shape | Meaning and normalization |
| --- | --- | --- |
| `data/anechoic_high_prec_dataset_last700_test_2.npy` | `(9,700,64,64,3)` | Sample, physical time, two spatial axes, pressure/velocity-x/velocity-y. The notebook transposes channels and uses `results/2DSound_v22/scalar.npy`. |
| `data/tube395_resampled_calibrated.npy` | `(1,100000,1,16)` | Sample, physical time, pressure, sensor. The original pressure notebook standardizes the measurement during loading and again before sampling. It applies one inverse transform to predictions and compares against the once-standardized reference. This sequence is retained. |
| `data/pu385.npy` | `(1,50000,2,15)` | Sample, time, pressure/velocity, sensor. Valid indices are `[1,3,4,5,6,7,8,10,11,12,13,14]`. Means and standard deviations are calculated separately by channel over the full measured record at those positions. |
| `data/Aligned_2900_3000_pos5.npy` | `(7,4,5000)` | Sensor, pressure/three velocity channels, time. Velocity channels are multiplied by acoustic impedance before reconstruction and divided by the same impedance afterward. |
| `data/kolmogorov/kolmogorov_flow_test.npy` | `(20,400,64,64,2)` | Trajectory, time, spatial axes, two velocity components. The source reshapes these into five Reynolds-number groups, uses the checkpoint scalar, and appends forcing and Reynolds-number conditioning channels. |
| `data/era5/era5_test.npy` | `(1,273,64,64,4)` | Sample, time, spatial axes, zonal wind/meridional wind/temperature/pressure. The supplied fields are already standardized. |

Physical time frames and diffusion steps describe different axes. The source uses 10-frame windows with two overlapping frames; requests for 100 physical frames therefore produce 106 frames. The tube pressure–velocity example requests 200 physical frames. The line-array example requests 1000 physical frames.

## Source computation details

The reconstruction notebooks retain the effective sampler settings from their source notebooks. The free-field demonstration and multi-trial evaluation use different PI-ISS weights: the demonstration uses observation factor 1 and equal acoustic residual factors; the evaluation uses observation factor 1.4 and multiplies continuity and momentum residual weights by 1e8. Both retain the base factor 5e-8 and 100 reverse steps.

The pressure–velocity tube source enables `MASK_SECOND_CHANNEL=True`, so all velocity measurements are masked. Its final `sensors_to_hide=None` setting retains all 12 valid pressure positions. The exported per-channel error table evaluates those measured positions; velocity is reconstructed without observing that channel. `1dtube_v31` is the active checkpoint in this source, while the other tube notebooks and training assets include additional checkpoints.

The measured line-array source adds 10 dB Gaussian noise before reconstruction without setting a separate seed for this added noise. Its seven-sensor MUSIC reference uses the measurement before the extra noise is added. The line-array evaluation uses the active PI-ISS calculation with 500 reverse steps. The local interference region is x from −0.6 to 0.6 m and z from 1.6 to 1.8 m. Both MUSIC calculations use pressure and all three velocity channels and the midpoint frequency 2950 Hz.

Kolmogorov energy is calculated with NumPy's unnormalized two-dimensional Fourier transform. Squared Fourier amplitudes are summed within unit-width radial shells, multiplied by one half, and averaged over physical time frames. The saved spectrum CSV preserves that convention. The FPE checkpoint used by the default Kolmogorov notebooks is the original checkpoint indexed 517.

The ERA5 reconstruction example uses 5% observations and SDE observation factor 0.2. The rate-sweep source uses SDE factor 0.1 and ISS factor 0.3, each divided by the square root of the observation fraction. The sweep varies sampling noise over seeds `[1,42,123,456,789]` while retaining the observation-mask seed from data preparation. Reported standard deviations use NumPy's population convention (`ddof=0`).

## Stored figure data

`source_data/experiment_results_20260331_204711.csv` and `source_data/baselines_comparison_20260331_213441.csv` are numerical tables from the free-field analysis. `source_figures.ipynb` uses the plotted observation-rate subset and upper-outlier removal specified for the downsampling panel: a value is omitted when it exceeds Q3 + 1.5 times the interquartile range. Means and population standard deviations use four trials for LNO and FNO at 4× and 5× downsampling and SDE at 6×; the other combinations use five trials. The CSV files contain all five trial values for every combination. The runnable random-observation notebook additionally displays all observation rates.

The downsampling table excludes values at or above the sentinel threshold 999.
Model loading and inference require the specified checkpoint, and the
evaluator reports metrics for completed runs.

## Notebook contents

The notebooks contain model-loading, input-preparation, sampling, measurement and plotting cells. Configuration paths resolve to the selected checkpoint directory. Baseline imports use the repository's model modules.

See [validation.md](validation.md) for the checked configurations. The notebooks
and stored tables include observation-rate and multi-seed comparisons.
