#!/usr/bin/env python
"""Plot Figure 4d at 385 Hz.

python scripts/plot_figure4.py --prediction results/tube/prediction_0000.npz \
    --data data/pu385.npy --config configs/figure4_tube.json --output-dir figures/figure4d
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def align_and_scale_signal(target_gt, source_pred):
    t_centered = target_gt - np.mean(target_gt)
    s_centered = source_pred - np.mean(source_pred)
    correlation = np.correlate(t_centered, s_centered, mode='full')
    lags = np.arange(1 - len(target_gt), len(target_gt))
    best_lag = lags[np.argmax(correlation)]
    source_shifted = np.roll(source_pred, best_lag)
    scale, bias = np.polyfit(source_shifted, target_gt, 1)
    calibrated_pred = source_shifted * scale + bias
    return calibrated_pred, best_lag, scale


def fit_velocity_waveforms(reference_velocity, raw_prediction_velocity):
    if reference_velocity.shape != raw_prediction_velocity.shape:
        raise ValueError("reference and predicted velocity must have the same shape")
    processed = np.empty_like(raw_prediction_velocity)
    sensor_count = reference_velocity.shape[1]
    lag_samples = np.empty(sensor_count, dtype=int)
    gain = np.empty(sensor_count)
    bias = np.empty(sensor_count)
    for index in range(sensor_count):
        reference = reference_velocity[:, index]
        raw = raw_prediction_velocity[:, index]
        processed[:, index], lag_samples[index], gain[index] = align_and_scale_signal(reference, raw)
        bias[index] = np.polyfit(np.roll(raw, lag_samples[index]), reference, 1)[1]
    return {
        "reference_velocity": reference_velocity,
        "raw_prediction_velocity": raw_prediction_velocity,
        "processed_prediction_velocity": processed,
        "lag_samples": lag_samples,
        "gain": gain,
        "bias": bias,
    }


def compact_tick(value, _position):
    return "0" if value == 0 else f"{value:g}"


def plot_waveforms(plot_data, output_dir):
    font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 6, "axes.labelsize": 6,
        "axes.titlesize": 6, "xtick.labelsize": 5, "ytick.labelsize": 5,
        "legend.fontsize": 6, "axes.linewidth": 0.6,
        "xtick.major.width": 0.4, "ytick.major.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "svg.fonttype": "none",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.default": "regular",
    })
    fig, axes = plt.subplots(3, 4, figsize=(12 / 2.54, 6 / 2.54), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15, wspace=0.15, hspace=0.35)
    frames = plot_data["physical_frame"]
    for index, ax in enumerate(axes.flat):
        ax.plot(frames, plot_data["reference_velocity"][:, index],
                color="black", linewidth=0.5, label="Ref.")
        ax.plot(frames, plot_data["processed_prediction_velocity"][:, index],
                color="#A61D24", linestyle=(0, (4, 3)), linewidth=0.8, label="PI-ISS")
        ax.set_title(f"Sensor {plot_data['sensor_id_one_based'][index]}", pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks([-0.003, 0, 0.003])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: compact_tick(value / 0.001, pos)))
        ax.xaxis.set_major_formatter(FuncFormatter(compact_tick))
        ax.set_ylim(-0.0035, 0.0035)
        ax.set_xlim(0, len(frames))
        if index % 4 == 0:
            ax.set_ylabel("Velocity (m/s)")
            ax.text(0, 1.02, r"$\times 10^{-3}$", transform=ax.transAxes, fontsize=5, ha="left", va="bottom")
        if index // 4 == 2:
            ax.set_xlabel("Time steps")
    fig.suptitle("Particle velocity waveform reconstruction for 385 Hz", fontsize=6, y=0.99)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07),
               ncol=2, frameon=False, handlelength=1.5)
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "plot_data.npz", **plot_data)
    for extension in ("svg", "png"):
        fig.savefig(output_dir / f"figure4d.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/pu385.npy")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/figure4_tube.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if settings["excitation_frequency_hz"] != 385:
        raise ValueError("Figure 4d uses the 385 Hz measurement")
    sensor_indices = np.asarray(settings["sensor_indices_zero_based"])
    if len(sensor_indices) != 12:
        raise ValueError("Figure 4d requires 12 measurement sensors")
    with np.load(args.prediction) as saved:
        prediction = saved["prediction"]
    if prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[2:] != (2, 64):
        raise ValueError("prediction must have shape [1, time, 2, 64] in physical units")
    measured = np.load(args.data, mmap_mode="r")
    start = settings["window_start"]
    reference = measured[0, start:start + prediction.shape[1], 1, :][:, sensor_indices]
    raw = prediction[0, :, 1, :][:, sensor_indices * settings["sensor_grid_stride"]]
    plot_data = fit_velocity_waveforms(reference, raw)
    plot_data.update({
        "sensor_id_one_based": sensor_indices + 1,
        "physical_frame": np.arange(prediction.shape[1]),
        "time_s": np.arange(prediction.shape[1]) * settings["physics_params"]["physics_dt"],
    })
    plot_waveforms(plot_data, args.output_dir)


if __name__ == "__main__":
    main()
