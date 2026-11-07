#!/usr/bin/env python
"""Paired reconstruction evaluation before and after FPE fine-tuning.

This script uses the case-specific notebook settings. Results are saved after
every trial to support resuming long evaluations. Reconstruction metrics
compare checkpoints on matched observations and random seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.signal import hilbert


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_ema, get_model  # noqa: E402
from sampler import VESDE, setup_seed  # noqa: E402
from sampler.utils import (  # noqa: E402
    NoneCorrector,
    mask_gen,
    s3gm_sample_1d,
    s3gm_sample_1d_ISS,
    s3gm_sample_1d_ISS_1dtube_res,
    s3gm_sample_1d_ISS_1dtube_res_p,
    s3gm_sample_2d,
    s3gm_sample_2d_ISS_2dsound,
    s3gm_sample_2d_ISS_2dsound_res,
    s3gm_sample_2d_ISS_ERA5,
)


ALLOWED_GPU_IDS = {0, 1, 2, 3}
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=(
            "2dsound", "tube-pressure-heldout", "tube", "line-sim",
            "line-measured", "era5",
        ),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="NAME=PATH",
        help="repeat to compare checkpoints using paired inputs and random seeds",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("screen", "full"), default="full")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--methods", default=None, help="comma-separated override")
    parser.add_argument("--power-ceiling-w", type=float, default=250.0)
    parser.add_argument("--power-poll-seconds", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--reverse-steps", type=int, default=None, help="override the number of reverse sampling steps")
    parser.add_argument("--max-trials", type=int, default=None, help="evaluate the first N trials")
    parser.add_argument(
        "--temporal-microbatch-target", type=int, default=26,
        help="target frames per temporal microbatch for high-power 2D cases",
    )
    parser.add_argument(
        "--forward-sleep-seconds", type=float, default=0.0,
        help="delay after each score-network call to reduce GPU duty cycle",
    )
    parser.add_argument(
        "--line-noise-seed", type=int, default=20260906,
        help="AWGN realization for the measured line-array protocol",
    )
    parser.add_argument(
        "--line-sampling-seed", type=int, default=3,
        help="reverse-diffusion seed for the measured line-array protocol",
    )
    return parser.parse_args()


def load_config(path: Path, device: str) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["device"] = device
    payload["gpu"] = device.split(":", 1)[1]
    payload["num_channels"] = payload.get(
        "num_channels", payload["num_components"] + payload["num_conditions"]
    )
    return SimpleNamespace(**payload)


def parse_checkpoint(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"checkpoint must be NAME=PATH, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    path = Path(raw_path)
    if not name or not path.is_file():
        raise FileNotFoundError(f"invalid checkpoint specification: {spec}")
    return name, path


def normalise_model_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(state)
    module_flags = [key.startswith("module.") for key in keys]
    if keys and all(module_flags):
        return {key[len("module."):]: value for key, value in state.items()}
    if any(module_flags):
        raise RuntimeError("checkpoint contains mixed DataParallel and bare keys")
    return state


def load_model(cfg: SimpleNamespace, checkpoint_path: Path, weights: str,
               device: torch.device) -> tuple[torch.nn.Module, dict]:
    model = get_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(normalise_model_state(state["model"]), strict=True)
    if weights == "ema":
        if "ema" not in state:
            raise RuntimeError(f"checkpoint has no EMA state: {checkpoint_path}")
        ema = get_ema(model.parameters(), decay=float(state["ema"]["decay"]))
        ema.load_state_dict(state["ema"])
        ema.shadow_params = [parameter.to(device=device) for parameter in ema.shadow_params]
        ema.copy_to(model.parameters())
    model.eval()
    return model, state


def query_power(gpu_id: int) -> float:
    output = subprocess.check_output(
        [
            "nvidia-smi", "-i", str(gpu_id), "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    )
    return float(output.strip().splitlines()[0])


class PowerWatchdog:
    def __init__(self, gpu_id: int, ceiling_w: float, poll_seconds: float, marker: Path):
        self.gpu_id = gpu_id
        self.ceiling_w = ceiling_w
        self.poll_seconds = poll_seconds
        self.marker = marker
        self.peak_w = query_power(gpu_id)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self.peak_w >= self.ceiling_w:
            raise RuntimeError(
                f"GPU {self.gpu_id} launch power {self.peak_w:.2f} W is at or above ceiling"
            )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                power_w = query_power(self.gpu_id)
                self.peak_w = max(self.peak_w, power_w)
                if power_w >= self.ceiling_w:
                    payload = {
                        "event": "power_ceiling_stop",
                        "time_unix": time.time(),
                        "gpu_id": self.gpu_id,
                        "power_draw_w": power_w,
                        "power_ceiling_w": self.ceiling_w,
                    }
                    self.marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
            except Exception as exc:  # The evaluation keeps running but records monitor failure.
                self.marker.write_text(
                    json.dumps({"event": "power_monitor_error", "error": repr(exc)}, indent=2),
                    encoding="utf-8",
                )
                return


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temp, path)


def pearson_flat(reference: np.ndarray, prediction: np.ndarray) -> float:
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(prediction, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def mean_available(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def metric_bundle(reference: np.ndarray, prediction: np.ndarray,
                  channel_names: list[str]) -> dict:
    """Return manuscript-global and historical notebook metric definitions."""
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim < 3:
        raise ValueError(f"metric shape mismatch: {reference.shape} vs {prediction.shape}")
    if reference.shape[1] != len(channel_names):
        raise ValueError("channel name count does not match channel dimension")

    channels = {}
    eps = np.finfo(np.float64).eps
    for channel_index, channel_name in enumerate(channel_names):
        gt = reference[:, channel_index]
        pred = prediction[:, channel_index]
        diff = pred - gt
        frame_denominator = np.linalg.norm(gt.reshape(len(gt), -1), axis=1)
        frame_numerator = np.linalg.norm(diff.reshape(len(diff), -1), axis=1)
        valid = frame_denominator > eps
        frame_ratios = frame_numerator[valid] / frame_denominator[valid]
        data_range = float(np.max(gt) - np.min(gt))
        frame_rmse = np.sqrt(np.mean(diff.reshape(len(diff), -1) ** 2, axis=1))
        gt_rms = float(np.sqrt(np.mean(gt ** 2)))
        pred_rms = float(np.sqrt(np.mean(pred ** 2)))
        frame_correlations = []
        for gt_frame, pred_frame in zip(gt, pred):
            correlation = pearson_flat(gt_frame, pred_frame)
            if np.isfinite(correlation):
                frame_correlations.append(correlation)
        channels[channel_name] = {
            "global_l2_nrmse": finite_or_none(
                np.linalg.norm(diff) / np.linalg.norm(gt)
            ) if np.linalg.norm(gt) > eps else None,
            "notebook_frame_mean_l2_nrmse": finite_or_none(frame_ratios.mean()),
            "notebook_frame_std_l2_nrmse": finite_or_none(frame_ratios.std(ddof=1)) if len(frame_ratios) > 1 else 0.0,
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "global_range_normalized_rmse": finite_or_none(
                np.sqrt(np.mean(diff ** 2)) / data_range
            ) if data_range > eps else None,
            "notebook_frame_mean_range_normalized_rmse": finite_or_none(
                frame_rmse.mean() / data_range
            ) if data_range > eps else None,
            "pearson": finite_or_none(pearson_flat(gt, pred)),
            "notebook_frame_mean_pearson": finite_or_none(
                np.mean(frame_correlations)
            ) if frame_correlations else None,
            "notebook_frame_std_pearson": finite_or_none(
                np.std(frame_correlations, ddof=1)
            ) if len(frame_correlations) > 1 else 0.0,
            "reference_rms": gt_rms,
            "prediction_rms": pred_rms,
            "relative_rms_amplitude_error": finite_or_none(
                abs(pred_rms - gt_rms) / gt_rms
            ) if gt_rms > eps else None,
        }
    return {
        "per_channel": channels,
        "macro_global_l2_nrmse": mean_available(
            [x["global_l2_nrmse"] for x in channels.values()]
        ),
        "macro_notebook_frame_mean_l2_nrmse": mean_available(
            [x["notebook_frame_mean_l2_nrmse"] for x in channels.values()]
        ),
        "macro_notebook_frame_mean_range_normalized_rmse": mean_available(
            [x["notebook_frame_mean_range_normalized_rmse"] for x in channels.values()]
        ),
        "macro_pearson": mean_available([x["pearson"] for x in channels.values()]),
        "macro_notebook_frame_mean_pearson": mean_available(
            [x["notebook_frame_mean_pearson"] for x in channels.values()]
        ),
    }


def acoustic_residuals_2d(field: np.ndarray, dt: float = 1e-4,
                          dx: float = 1 / 21, dy: float = 1 / 21) -> dict:
    p, vx, vy = field[:, 0], field[:, 1], field[:, 2]
    rho0, sound_speed = 1.21, 343.0
    dp_dt = np.gradient(p, dt, axis=0)
    dp_dx = np.gradient(p, dx, axis=1)
    dp_dy = np.gradient(p, dy, axis=2)
    dvx_dt = np.gradient(vx, dt, axis=0)
    dvy_dt = np.gradient(vy, dt, axis=0)
    dvx_dx = np.gradient(vx, dx, axis=1)
    dvy_dy = np.gradient(vy, dy, axis=2)
    d2p_dt2 = np.gradient(dp_dt, dt, axis=0)
    laplacian_p = np.gradient(dp_dx, dx, axis=1) + np.gradient(dp_dy, dy, axis=2)
    residuals = {
        "wave": d2p_dt2 - sound_speed ** 2 * laplacian_p,
        "continuity": dp_dt + rho0 * sound_speed ** 2 * (dvx_dx + dvy_dy),
        "euler_x": rho0 * dvx_dt + dp_dx,
        "euler_y": rho0 * dvy_dt + dp_dy,
    }
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    return {
        name: {
            "mae": float(np.mean(np.abs(value[interior]))),
            "rms": float(np.sqrt(np.mean(value[interior] ** 2))),
        }
        for name, value in residuals.items()
    }


def tube_phase_amplitude(reference: np.ndarray, prediction: np.ndarray,
                         channel_names: list[str]) -> dict:
    """Evaluate each measured sensor trace; arrays are T,C,S."""
    output = {}
    eps = np.finfo(np.float64).eps
    for channel_index, channel_name in enumerate(channel_names):
        amplitude_errors = []
        phase_errors = []
        sensor_correlations = []
        for sensor_index in range(reference.shape[2]):
            gt = reference[:, channel_index, sensor_index]
            pred = prediction[:, channel_index, sensor_index]
            gt_analytic = hilbert(gt)
            pred_analytic = hilbert(pred)
            gt_rms = np.sqrt(np.mean(gt ** 2))
            pred_rms = np.sqrt(np.mean(pred ** 2))
            amplitude_errors.append(abs(pred_rms - gt_rms) / max(gt_rms, eps))
            phase_delta = np.angle(pred_analytic * np.conj(gt_analytic))
            reliable = np.abs(gt_analytic) > np.quantile(np.abs(gt_analytic), 0.1)
            phase_errors.append(np.mean(np.abs(phase_delta[reliable])) * 180 / np.pi)
            sensor_correlations.append(pearson_flat(gt, pred))
        output[channel_name] = {
            "mean_sensor_relative_rms_amplitude_error": float(np.mean(amplitude_errors)),
            "median_sensor_relative_rms_amplitude_error": float(np.median(amplitude_errors)),
            "mean_absolute_hilbert_phase_error_deg": float(np.mean(phase_errors)),
            "mean_sensor_pearson": finite_or_none(np.nanmean(sensor_correlations)),
            "per_sensor_relative_rms_amplitude_error": [float(x) for x in amplitude_errors],
            "per_sensor_phase_error_deg": [float(x) for x in phase_errors],
        }
    return output


def tube_pressure_heldout_metrics(reference: np.ndarray,
                                  prediction: np.ndarray) -> dict:
    """Return both the manuscript-defined and plotted held-out-sensor scores."""
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != 3:
        raise ValueError(
            f"held-out pressure shape mismatch: {reference.shape} vs {prediction.shape}"
        )
    if reference.shape[1] != 1:
        raise ValueError("held-out pressure metric expects one pressure channel")

    gt = reference[:, 0, :]
    pred = prediction[:, 0, :]
    diff = pred - gt
    eps = np.finfo(np.float64).eps

    sensor_denominator = np.linalg.norm(gt, axis=0)
    sensor_numerator = np.linalg.norm(diff, axis=0)
    per_sensor_l2 = np.divide(
        sensor_numerator,
        sensor_denominator,
        out=np.full_like(sensor_numerator, np.nan),
        where=sensor_denominator > eps,
    )

    frame_denominator = np.linalg.norm(gt, axis=1)
    frame_numerator = np.linalg.norm(diff, axis=1)
    valid_frames = frame_denominator > eps
    frame_l2 = frame_numerator[valid_frames] / frame_denominator[valid_frames]

    shared_range = float(np.max(gt) - np.min(gt))
    if shared_range > eps:
        absolute_error = np.abs(diff)
        plotted_mean = np.mean(absolute_error, axis=0) / shared_range
        plotted_errorbar = np.std(absolute_error, axis=0) / 5.0 / shared_range
    else:
        plotted_mean = np.full(gt.shape[1], np.nan)
        plotted_errorbar = np.full(gt.shape[1], np.nan)

    global_denominator = np.linalg.norm(gt)
    global_l2 = (
        float(np.linalg.norm(diff) / global_denominator)
        if global_denominator > eps else None
    )
    return {
        "manuscript_equation_6": {
            "global_l2_nrmse": global_l2,
            "per_sensor_l2_nrmse": [finite_or_none(value) for value in per_sensor_l2],
            "mean_per_sensor_l2_nrmse": finite_or_none(np.nanmean(per_sensor_l2)),
            "frame_mean_l2_nrmse": finite_or_none(np.mean(frame_l2)),
            "frame_std_l2_nrmse": (
                finite_or_none(np.std(frame_l2, ddof=1)) if len(frame_l2) > 1 else 0.0
            ),
        },
        "extended_data_figure_3_implementation": {
            "per_sensor_mean_absolute_error_over_shared_range": [
                finite_or_none(value) for value in plotted_mean
            ],
            "per_sensor_errorbar_std_absolute_error_div_5_over_shared_range": [
                finite_or_none(value) for value in plotted_errorbar
            ],
            "mean_sensor_score": finite_or_none(np.nanmean(plotted_mean)),
            "shared_reference_range": shared_range,
            "note": (
                "The plotted score is mean absolute error divided by the reference range "
                "shared across the six held-out sensors; the panel labels it nRMSE."
            ),
        },
    }


class ArrayScaler2D:
    def __init__(self, scalar_path: Path, device: torch.device):
        scalar = np.load(scalar_path, allow_pickle=True)
        self.mean_np = np.asarray(scalar[0])
        self.std_np = np.asarray(scalar[1])
        self.mean_torch = torch.as_tensor(self.mean_np).float().permute(0, 1, 4, 2, 3).to(device)
        self.std_torch = torch.as_tensor(self.std_np).float().permute(0, 1, 4, 2, 3).to(device)

    def normalize(self, array: np.ndarray) -> np.ndarray:
        return (array - self.mean_np.transpose(0, 1, 4, 2, 3)) / self.std_np.transpose(0, 1, 4, 2, 3)

    def unnormalize(self, array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(array, torch.Tensor):
            return array * self.std_torch + self.mean_torch
        return array * self.std_np.transpose(0, 1, 4, 2, 3) + self.mean_np.transpose(0, 1, 4, 2, 3)


class ChannelScaler1D:
    def __init__(self, mean: np.ndarray, std: np.ndarray, device: torch.device):
        self.mean_np = np.asarray(mean).reshape(1, 1, -1, 1)
        self.std_np = np.asarray(std).reshape(1, 1, -1, 1)
        self.mean_torch = torch.as_tensor(self.mean_np, dtype=torch.float32, device=device)
        self.std_torch = torch.as_tensor(self.std_np, dtype=torch.float32, device=device)

    def normalize(self, array: np.ndarray) -> np.ndarray:
        return (array - self.mean_np) / self.std_np

    def unnormalize(self, array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(array, torch.Tensor):
            return array * self.std_torch + self.mean_torch
        return array * self.std_np + self.mean_np


def make_net_fn(model: torch.nn.Module, cfg: SimpleNamespace, device: torch.device,
                forward_sleep_seconds: float = 0.0):
    if cfg.dims == 2:
        latent_mask = torch.ones([cfg.num_frames, 1, 1, 1], device=device)
        obs_mask = torch.zeros([cfg.num_frames, 1, 1, 1], device=device)
    else:
        latent_mask = torch.ones([cfg.num_frames, 1, 1], device=device)
        obs_mask = torch.zeros([cfg.num_frames, 1, 1], device=device)
    frame_indices = torch.arange(cfg.num_frames, device=device)

    def repeat(value: torch.Tensor, number: int) -> torch.Tensor:
        return torch.stack([value for _ in range(number)])

    def net_fn(x, t):
        result = model(
            x, x0=x, timesteps=t,
            latent_mask=repeat(latent_mask, len(x)),
            obs_mask=repeat(obs_mask, len(x)),
            frame_indices=repeat(frame_indices, len(x)),
        )[0]
        if forward_sleep_seconds > 0:
            time.sleep(forward_sleep_seconds)
        return result

    return net_fn


def methods_for(args: argparse.Namespace, defaults: list[str]) -> list[str]:
    methods = args.methods.split(",") if args.methods else defaults
    invalid = set(methods) - {"sde", "iss", "pi-iss"}
    if invalid:
        raise ValueError(f"unsupported methods: {sorted(invalid)}")
    return methods


def aligned_length(target: int, frames: int = 10, overlap: int = 2) -> int:
    blocks = int(math.ceil((target - overlap) / (frames - overlap)))
    return blocks * (frames - overlap) + overlap


def evaluate_2dsound(args: argparse.Namespace, cfg: SimpleNamespace, model: torch.nn.Module,
                     device: torch.device, trial_callback) -> None:
    test_path = Path(str(SOURCE_ROOT / 'data/anechoic_high_prec_dataset_last700_test_2.npy'))
    scalar_path = SOURCE_ROOT / "results/2DSound_v22/scalar.npy"
    data = np.load(test_path, mmap_mode="r")
    scaler = ArrayScaler2D(scalar_path, device)
    trials = [(0, 0)] if args.profile == "screen" else [
        (1, 50), (2, 150), (3, 250), (4, 350), (5, 450)
    ]
    if args.max_trials is not None:
        trials = trials[:args.max_trials]
    methods = methods_for(args, ["sde", "iss", "pi-iss"])
    scale, overlap = 6, 2
    target_sequence_length = 100
    temporal_chunk_length = aligned_length(
        min(args.temporal_microbatch_target, target_sequence_length),
        cfg.num_frames,
        overlap,
    )
    chunk_starts = list(
        range(0, target_sequence_length, temporal_chunk_length)
    )
    sequence_length = temporal_chunk_length * len(chunk_starts)
    reverse_steps = args.reverse_steps or 100
    sparsity = 1 - 1 / scale ** 2
    net_fn = make_net_fn(model, cfg, device)

    for frequency_index, start in trials:
        raw = np.asarray(
            data[frequency_index:frequency_index + 1, start:start + sequence_length]
        ).transpose(0, 1, 4, 2, 3)
        normalized = scaler.normalize(raw).astype(np.float32)
        for method in methods:
            trial = {"frequency_index": frequency_index, "start": start, "scale": scale,
                     "sequence_length": sequence_length,
                     "temporal_microbatch_length": temporal_chunk_length,
                     "temporal_microbatch_count": len(chunk_starts),
                     "sampling_seeds": [
                         123 + 100_000 * i for i in range(len(chunk_starts))
                     ]}
            if trial_callback.is_complete(method, trial):
                continue
            prediction_chunks = []
            for chunk_index, chunk_start in enumerate(chunk_starts):
                reference_chunk = normalized[
                    :, chunk_start:chunk_start + temporal_chunk_length
                ]

                def transform(x):
                    return x[
                        :, :temporal_chunk_length, :cfg.num_components,
                        ::scale, ::scale,
                    ]

                observation = transform(reference_chunk)
                setup_seed(123 + 100_000 * chunk_index)
                sde = VESDE(
                    cfg, sigma_min=cfg.beta_min, sigma_max=cfg.beta_max,
                    N=reverse_steps,
                )
                if method == "sde":
                    chunk_prediction = s3gm_sample_2d(
                        cfg, net_fn, sde, observation, transform,
                        alpha=0.5 / np.sqrt(1 - sparsity), beta=0.4, gamma=0.5,
                        T_prime_y=temporal_chunk_length,
                        T_prime=temporal_chunk_length, overlap=overlap,
                        corrector=NoneCorrector, n_steps=0, snr=0.128,
                        device=str(device), dtype="float", eps=1e-12,
                        probability_flow=False, continuous=True,
                    )
                elif method == "iss":
                    chunk_prediction = s3gm_sample_2d_ISS_2dsound(
                        cfg, net_fn, sde, observation, transform,
                        alpha=0.5 / np.sqrt(1 - sparsity), beta=0.4, gamma=0.5,
                        T_prime_y=temporal_chunk_length,
                        T_prime=temporal_chunk_length, overlap=overlap,
                        corrector=NoneCorrector, n_steps=0, eta=0,
                        reg_coef_pa=10 * temporal_chunk_length, reg_coef_ar=1e2,
                        denoising_steps=reverse_steps, snr=0.128,
                        device=str(device), dtype="float", eps=1e-12,
                        probability_flow=False, continuous=True,
                    )
                else:
                    chunk_prediction = s3gm_sample_2d_ISS_2dsound_res(
                        cfg, net_fn, sde, observation, transform,
                        alpha=1.0, beta=0.4, gamma=0.5,
                        T_prime_y=temporal_chunk_length,
                        T_prime=temporal_chunk_length, overlap=overlap,
                        corrector=NoneCorrector, n_steps=0, eta=0,
                        use_residual_objective=False,
                        reg_coef_pa=10 * temporal_chunk_length, reg_coef_ar=1e2,
                        denoising_steps=reverse_steps, snr=0.128,
                        device=str(device), dtype="float", eps=1e-12,
                        probability_flow=False, continuous=True,
                        data_scalar=scaler.unnormalize,
                        physics_start_step=reverse_steps, reg_coef_wave=5e-8,
                        reg_coef_cont=5e-8, reg_coef_euler=5e-8, physics_c=343.0,
                        physics_rho0=1.21, physics_dt=1e-4,
                        physics_dx=1 / 21, physics_dy=1 / 21,
                    )
                prediction_chunks.append(chunk_prediction)
            prediction = np.concatenate(prediction_chunks, axis=1)
            physical = np.asarray(scaler.unnormalize(prediction))
            metrics = metric_bundle(raw[0], physical[0], ["pressure", "vx", "vy"])
            observed = np.zeros((cfg.image_size, cfg.image_size), dtype=bool)
            observed[::scale, ::scale] = True
            metrics["unobserved_only"] = metric_bundle(
                raw[0][:, :, ~observed], physical[0][:, :, ~observed],
                ["pressure", "vx", "vy"],
            )
            metrics["acoustic_residuals"] = {
                "reference": acoustic_residuals_2d(raw[0]),
                "prediction": acoustic_residuals_2d(physical[0]),
            }
            trial_callback(method, trial, metrics, physical)


def evaluate_tube_pressure_heldout(args: argparse.Namespace, cfg: SimpleNamespace,
                                   model: torch.nn.Module, device: torch.device,
                                   trial_callback) -> None:
    data_root = Path(
        str(SOURCE_ROOT / "data")
    )
    scalar_path = SOURCE_ROOT / "results/exp_p_v21/scalar.npy"
    mean, std = np.load(scalar_path, allow_pickle=True)
    mean = float(np.asarray(mean))
    std = float(np.asarray(std))

    def normalize(x):
        return (x - mean) / std

    def unnormalize(x):
        return x * std + mean

    masked_indices = np.array([2, 4, 7, 9, 12, 14], dtype=np.int64)
    kept_indices = np.setdiff1d(np.arange(16), masked_indices)
    reconstructed_indices = masked_indices * 4
    kept_indices_torch = torch.as_tensor(kept_indices, dtype=torch.long, device=device)

    overlap = 2
    sequence_length = aligned_length(500, cfg.num_frames, overlap)
    reverse_steps = args.reverse_steps or 500
    sparsity = 1 - len(kept_indices) / cfg.image_size
    methods = methods_for(args, ["sde", "iss", "pi-iss"])
    frequencies = [395] if args.profile == "screen" else [175, 345, 395]
    if args.max_trials is not None:
        frequencies = frequencies[:args.max_trials]
    net_fn = make_net_fn(model, cfg, device)

    for frequency_hz in frequencies:
        data_path = data_root / f"tube{frequency_hz}_resampled_calibrated.npy"
        data = np.load(data_path, mmap_mode="r")
        sample_index = 0 if data.shape[0] < 5 else 4
        raw = np.asarray(
            data[sample_index:sample_index + 1, :sequence_length]
        )
        notebook_reference = normalize(raw).astype(np.float32)
        notebook_model_input = normalize(notebook_reference).astype(np.float32)
        observation = notebook_model_input[..., kept_indices]

        def transform(x):
            downsampled = x[:, :sequence_length, :cfg.num_components, ::4]
            if isinstance(downsampled, np.ndarray):
                return downsampled[..., kept_indices]
            return downsampled.index_select(-1, kept_indices_torch)

        for method in methods:
            trial = {
                "frequency_hz": frequency_hz,
                "data_path": str(data_path),
                "sample_index": sample_index,
                "start": 0,
                "sequence_length": sequence_length,
                "reverse_steps": reverse_steps,
                "sampling_seed": 123,
                "input_sensor_indices_zero_based": kept_indices.tolist(),
                "heldout_sensor_indices_zero_based": masked_indices.tolist(),
                "input_sensor_count": len(kept_indices),
                "heldout_sensor_count": len(masked_indices),
            }
            if trial_callback.is_complete(method, trial):
                continue

            setup_seed(123)
            sde = VESDE(
                cfg, sigma_min=cfg.beta_min, sigma_max=cfg.beta_max,
                N=reverse_steps,
            )
            if method == "sde":
                prediction = s3gm_sample_1d(
                    cfg, net_fn, sde, observation, transform,
                    alpha=0.2 / np.sqrt(1 - sparsity), beta=0.4, gamma=0.5,
                    T_prime_y=sequence_length, T_prime=sequence_length,
                    overlap=overlap, corrector=NoneCorrector, snr=0.128,
                    device=str(device), dtype="float", eps=1e-12,
                    probability_flow=False, continuous=True,
                )
            elif method == "iss":
                prediction = s3gm_sample_1d_ISS(
                    cfg, net_fn, sde, observation, transform,
                    alpha=0.2 / np.sqrt(1 - sparsity), beta=0.4, gamma=0.5,
                    T_prime_y=sequence_length, T_prime=sequence_length,
                    overlap=overlap, denoising_steps=reverse_steps, eta=0,
                    reg_coef_pa=0, corrector=NoneCorrector, snr=0.128,
                    device=str(device), dtype="float", eps=1e-12,
                    probability_flow=False, continuous=True,
                )
            else:
                prediction = s3gm_sample_1d_ISS_1dtube_res_p(
                    cfg, net_fn, sde, observation, transform,
                    alpha=0.5 / np.sqrt(1 - sparsity), beta=0.4, gamma=0.5,
                    T_prime_y=sequence_length, T_prime=sequence_length,
                    overlap=overlap, denoising_steps=reverse_steps, eta=0,
                    reg_coef_pa=0, data_scalar=unnormalize,
                    physics_start_step=reverse_steps, reg_coef_wave=0.1,
                    physics_c=343.0, physics_rho0=1.21,
                    physics_dt=1e-4, physics_dx=0.02,
                    normalized_velocity_zero=None, corrector=NoneCorrector,
                    snr=0.128, device=str(device), dtype="float", eps=1e-12,
                    probability_flow=False, continuous=True,
                )

            reported_prediction = np.asarray(unnormalize(prediction))
            reference_heldout = np.take(
                notebook_reference[0], masked_indices, axis=-1
            )
            prediction_heldout = np.take(
                reported_prediction[0], reconstructed_indices, axis=-1
            )
            metrics = metric_bundle(
                reference_heldout, prediction_heldout, ["pressure"]
            )
            metrics["heldout_pressure"] = tube_pressure_heldout_metrics(
                reference_heldout, prediction_heldout
            )
            metrics["evaluation_scope"] = {
                "paper_role": "Extended Data Figure 3",
                "input_sensor_count": 10,
                "heldout_sensor_count": 6,
                "heldout_sensor_indices_one_based": (masked_indices + 1).tolist(),
                "dense_reconstruction_nodes": 64,
                "normalization_lineage": (
                    "The reference uses data normalized once. Sampling inputs are normalized twice, "
                    "and one inverse transform is applied to predictions."
                ),
                "source_script": str(SOURCE_ROOT / "scripts/evaluate_fpe_sampling.py"),
                "source_checkpoint": str(
                    SOURCE_ROOT / "results/exp_p_v21/checkpoint.pth"
                ),
            }
            trial_callback(method, trial, metrics, reported_prediction)


def evaluate_tube(args: argparse.Namespace, cfg: SimpleNamespace, model: torch.nn.Module,
                  device: torch.device, trial_callback) -> None:
    data_path = Path(str(SOURCE_ROOT / 'data/pu385.npy'))
    data = np.load(data_path, mmap_mode="r")
    valid_sensors = np.array([1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14])
    measured = np.asarray(data[:, :, :, valid_sensors])
    mean = measured.mean(axis=(0, 1, 3))
    std = measured.std(axis=(0, 1, 3))
    scaler = ChannelScaler1D(mean, std, device)
    overlap = 2
    # The standing-wave comparison uses the window beginning at sample 100.
    starts = [100]
    sequence_length = aligned_length(200, cfg.num_frames, overlap)
    model_positions = torch.as_tensor(valid_sensors * 4, dtype=torch.long, device=device)
    channel_mask = torch.tensor([1.0, 0.0], device=device).view(1, 1, 2, 1)

    def transform(x):
        return x[:, :, :cfg.num_components, model_positions] * channel_mask

    spatial_sparsity = 1 - len(valid_sensors) / cfg.image_size
    sparsity = (spatial_sparsity + 1.0) / 2.0
    reverse_steps = args.reverse_steps or 100
    methods = methods_for(args, ["sde", "iss", "pi-iss"])
    net_fn = make_net_fn(model, cfg, device)

    for start in starts:
        normalized = scaler.normalize(
            measured[:, start:start + sequence_length]
        ).astype(np.float32)
        raw = measured[:, start:start + sequence_length]
        observation = normalized.copy()
        observation[:, :, 1, :] = 0.0
        for method in methods:
            trial = {
                "start": start,
                "sequence_length": sequence_length,
                "reverse_steps": reverse_steps,
                "sampling_seed": 123,
                "pressure_sensor_indices_zero_based": valid_sensors.tolist(),
                "pressure_sensor_indices_one_based": (valid_sensors + 1).tolist(),
                "pressure_sensor_count": 12,
                "velocity_observation_count": 0,
            }
            if trial_callback.is_complete(method, trial):
                continue
            setup_seed(123)
            sde = VESDE(cfg, sigma_min=cfg.beta_min, sigma_max=cfg.beta_max, N=reverse_steps)
            if method == "sde":
                prediction = s3gm_sample_1d(
                    cfg, net_fn, sde, observation, transform,
                    alpha=0.5 / np.sqrt(1 - sparsity), beta=0.4, gamma=0.5,
                    T_prime_y=sequence_length, T_prime=sequence_length, overlap=overlap,
                    corrector=NoneCorrector, snr=0.128, device=str(device), dtype="float",
                    eps=1e-12, probability_flow=False, continuous=True,
                )
            elif method == "iss":
                prediction = s3gm_sample_1d_ISS(
                    cfg, net_fn, sde, observation, transform,
                    alpha=0.5 / np.sqrt(1 - sparsity), beta=0.4, gamma=0.5,
                    T_prime_y=sequence_length, T_prime=sequence_length, overlap=overlap,
                    denoising_steps=reverse_steps, eta=0, corrector=NoneCorrector,
                    snr=0.128, device=str(device), dtype="float", eps=1e-12,
                    probability_flow=False, continuous=True, reg_coef_pa=0,
                )
            else:
                prediction = s3gm_sample_1d_ISS_1dtube_res(
                    cfg, net_fn, sde, observation, transform,
                    alpha=0.5 / np.sqrt(1 - sparsity), beta=0.4, gamma=0.5,
                    T_prime_y=sequence_length, T_prime=sequence_length, overlap=overlap,
                    denoising_steps=reverse_steps, eta=0, reg_coef_pa=0.1 * sequence_length,
                    data_scalar=scaler.unnormalize, physics_start_step=reverse_steps,
                    physics_c=343.0, physics_rho0=1.21, physics_dt=1e-4, physics_dx=0.02,
                    reg_coef_wave=0.1, reg_coef_cont=0.1, reg_coef_euler=0.1,
                    reg_coef_bc=0.0, normalized_velocity_zero=None,
                    corrector=NoneCorrector, snr=0.128, device=str(device), dtype="float",
                    eps=1e-12, probability_flow=False, continuous=True,
                )
            physical = np.asarray(scaler.unnormalize(prediction))
            physical_at_sensors = physical[:, :, :, valid_sensors * 4]
            metrics = metric_bundle(raw[0], physical_at_sensors[0], ["pressure", "velocity"])
            metrics["phase_amplitude_at_12_measured_sensors"] = tube_phase_amplitude(
                raw[0], physical_at_sensors[0], ["pressure", "velocity"]
            )
            metrics["evaluation_scope"] = {
                "paper_role": "Figure 4 cross-modal standing-wave-tube result",
                "source_notebook": str(
                    SOURCE_ROOT / "notebooks/tube/pressure_velocity.ipynb"
                ),
                "data_path": str(data_path),
                "excitation_frequency_hz": 385,
                "paper_displayed_method": "PI-ISS",
                "paper_numeric_anchor_available": False,
                "pressure": "observed at all 12 evaluation sensors",
                "velocity": "fully hidden at all 12 evaluation sensors",
                "dense_64_point_ground_truth_available": False,
                "pi_iss_physics_residual_active": False,
                "pi_iss_physics_note": (
                    "Physics residuals are evaluated for configurations with at least four "
                    "components. They are inactive for this two-component pressure/velocity case."
                ),
            }
            trial_callback(method, trial, metrics, physical)


def evaluate_line_sim(args: argparse.Namespace, cfg: SimpleNamespace, model: torch.nn.Module,
                      device: torch.device, trial_callback) -> None:
    data_path = Path(str(SOURCE_ROOT / 'data/fdtd_3d_data_an_100_test.npy'))
    data = np.load(data_path, mmap_mode="r")
    test_ids = [10] if args.profile == "screen" else [30, 40, 50, 60, 70]
    if args.max_trials is not None:
        test_ids = test_ids[:args.max_trials]
    methods = methods_for(args, ["pi-iss"])
    if methods != ["pi-iss"]:
        raise ValueError("line-sim supports the PI-ISS method")
    # v33 uses impedance-matched inputs (p, Z0*vx, Z0*vy, Z0*vz),
    # 1000 frames, 500 reverse steps and 7 of 32 regularly spaced sensors.
    impedance = 1.21 * 343.0
    inverse_scale = np.array(
        [1.0, 1 / impedance, 1 / impedance, 1 / impedance], dtype=np.float32
    ).reshape(1, 1, 4, 1)

    def to_physical(x):
        if isinstance(x, torch.Tensor):
            scale_tensor = torch.as_tensor(inverse_scale, dtype=x.dtype, device=x.device)
            return x * scale_tensor
        return x * inverse_scale

    overlap, scale = 2, 5
    sequence_length = aligned_length(1000, cfg.num_frames, overlap)
    reverse_steps = args.reverse_steps or 500
    sparsity = 1 - 1 / scale
    net_fn = make_net_fn(model, cfg, device)

    for test_id in test_ids:
        trial = {"test_id": test_id, "start": 0, "scale": scale,
                 "sequence_length": sequence_length, "sampling_seed": 3,
                 "network_units": "p,Z0*vx,Z0*vy,Z0*vz", "impedance": impedance}
        if trial_callback.is_complete("pi-iss", trial):
            continue
        raw_thc = np.asarray(data[test_id, :sequence_length])
        network_thc = raw_thc.copy()
        network_thc[..., 1:] *= impedance
        normalized_thc = network_thc[np.newaxis].transpose(0, 1, 3, 2).astype(np.float32)
        raw = raw_thc[np.newaxis].transpose(0, 1, 3, 2)
        transform = lambda x: x[:, :sequence_length, :cfg.num_components, ::scale]
        observation = transform(normalized_thc).astype(np.float32)
        setup_seed(3)
        sde = VESDE(cfg, sigma_min=cfg.beta_min, sigma_max=cfg.beta_max, N=reverse_steps)
        prediction = s3gm_sample_1d_ISS_1dtube_res(
            cfg, net_fn, sde, observation, transform,
            alpha=0.1 / np.sqrt(1 - sparsity), beta=0.1, gamma=0.5,
            T_prime_y=sequence_length, T_prime=sequence_length, overlap=overlap,
            denoising_steps=reverse_steps, eta=0, reg_coef_pa=1.0,
            data_scalar=to_physical, physics_start_step=reverse_steps,
            physics_c=343.0, physics_rho0=1.21, physics_dt=1e-5, physics_dx=0.032,
            reg_coef_wave=1e-3, reg_coef_cont=0.0, reg_coef_euler=0.0,
            reg_coef_bc=0.0, normalized_velocity_zero=None,
            corrector=NoneCorrector, snr=0.128, device=str(device), dtype="float",
            eps=1e-12, probability_flow=False, continuous=True,
        )
        physical = np.asarray(to_physical(prediction))
        metrics = {
            "normalized": metric_bundle(
                normalized_thc[0], prediction[0], ["pressure", "vx", "vy", "vz"]
            ),
            "physical": metric_bundle(raw[0], physical[0], ["pressure", "vx", "vy", "vz"]),
        }
        observed = np.zeros(cfg.image_size, dtype=bool)
        observed[::scale] = True
        metrics["normalized_unobserved_only"] = metric_bundle(
            normalized_thc[0][:, :, ~observed], prediction[0][:, :, ~observed],
            ["pressure", "vx", "vy", "vz"],
        )
        metrics["physical_unobserved_only"] = metric_bundle(
            raw[0][:, :, ~observed], physical[0][:, :, ~observed],
            ["pressure", "vx", "vy", "vz"],
        )
        metrics["evaluation_scope"] = {
            "pi_iss_physics_residual_active": True,
            "pi_iss_physics_note": (
                "The v33 protocol activates the wave-equation residual with coefficient "
                "1e-3; continuity, Euler and boundary coefficients remain zero."
            ),
        }
        trial_callback("pi-iss", trial, metrics, physical)


def add_awgn_per_channel(signal: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = np.zeros_like(signal)
    for sensor_index in range(signal.shape[0]):
        for channel_index in range(signal.shape[1]):
            trace = signal[sensor_index, channel_index]
            signal_power = float(np.mean(trace ** 2))
            if signal_power == 0:
                noisy[sensor_index, channel_index] = trace
            else:
                noise_std = np.sqrt(signal_power / (10 ** (snr_db / 10)))
                noisy[sensor_index, channel_index] = trace + rng.normal(
                    0.0, noise_std, trace.shape
                )
    return noisy


def music_pv_map(data_phys: np.ndarray, coordinates: np.ndarray, frequency: float,
                 scan_points: int = 150) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sound_speed, impedance = 343.0, 1.21 * 343.0
    wave_number = 2 * np.pi * frequency / sound_speed
    scaled = np.asarray(data_phys, dtype=np.float64).copy()
    scaled[:, 1:, :] *= impedance
    analytic = hilbert(scaled, axis=-1)
    snapshots = analytic.reshape(len(coordinates) * 4, -1)
    covariance = snapshots @ snapshots.conj().T / snapshots.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    noise_vectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]][:, 2:]

    scan_x = np.linspace(-1.5, 1.5, scan_points)
    scan_z = np.linspace(0.01, 1.8, scan_points)
    x_grid, z_grid = np.meshgrid(scan_x, scan_z)
    points = np.stack(
        [x_grid.reshape(-1), np.ones(x_grid.size), z_grid.reshape(-1)], axis=1
    )
    difference = coordinates[np.newaxis, :, :] - points[:, np.newaxis, :]
    distance = np.linalg.norm(difference, axis=2)
    direction = difference / np.maximum(distance[:, :, np.newaxis], 1e-12)
    propagation = np.exp(-1j * wave_number * distance) / np.maximum(distance, 1e-12)
    block = propagation[:, :, np.newaxis] * np.concatenate(
        [np.ones((*direction.shape[:2], 1)), direction], axis=2
    )
    steering = block.reshape(len(points), -1).T
    steering /= np.maximum(np.linalg.norm(steering, axis=0, keepdims=True), 1e-12)
    projection = noise_vectors.conj().T @ steering
    spectrum = 1.0 / (np.sum(np.abs(projection) ** 2, axis=0) + 1e-12)
    spectrum = spectrum.reshape(x_grid.shape)
    spectrum_db = 10 * np.log10(spectrum / np.max(spectrum))
    return x_grid, z_grid, spectrum_db


def two_source_localization(x_grid: np.ndarray, z_grid: np.ndarray,
                            spectrum_db: np.ndarray,
                            true_positions: np.ndarray) -> dict:
    order = np.argsort(spectrum_db.reshape(-1))[::-1]
    estimates = []
    for flat_index in order:
        point = np.array([x_grid.reshape(-1)[flat_index], z_grid.reshape(-1)[flat_index]])
        if all(np.linalg.norm(point - previous) >= 0.15 for previous in estimates):
            estimates.append(point)
        if len(estimates) == 2:
            break
    estimates_array = np.asarray(estimates)
    cost = np.linalg.norm(
        true_positions[:, np.newaxis, :] - estimates_array[np.newaxis, :, :], axis=2
    )
    true_indices, estimate_indices = linear_sum_assignment(cost)
    errors = cost[true_indices, estimate_indices]
    return {
        "estimated_xz_m": estimates_array[estimate_indices].tolist(),
        "true_xz_m": true_positions[true_indices].tolist(),
        "per_source_error_m": errors.tolist(),
        "mean_localization_error_m": float(errors.mean()),
        "max_localization_error_m": float(errors.max()),
    }


def interference_peak(x_grid: np.ndarray, z_grid: np.ndarray,
                      spectrum_db: np.ndarray) -> float:
    region = (
        (x_grid >= -0.6) & (x_grid <= 0.6) &
        (z_grid >= 1.6) & (z_grid <= 1.8)
    )
    return float(np.max(spectrum_db[region]))


def evaluate_line_measured(args: argparse.Namespace, cfg: SimpleNamespace,
                           model: torch.nn.Module, device: torch.device,
                           trial_callback) -> None:
    data_path = Path(
        str(SOURCE_ROOT / 'data/Aligned_2900_3000_pos5.npy')
    )
    raw = np.load(data_path)
    impedance = 1.21 * 343.0
    network_units = raw.copy()
    network_units[:, 1:, :] *= impedance
    noise_seed, snr_db = args.line_noise_seed, 10.0
    sampling_seed = args.line_sampling_seed
    noisy_network_units = add_awgn_per_channel(network_units, snr_db, noise_seed)
    overlap, start = 2, 2000
    sequence_length = aligned_length(1000, cfg.num_frames, overlap)
    observation = noisy_network_units[:, :, start:start + sequence_length].transpose(
        2, 1, 0
    )[np.newaxis].astype(np.float32)
    scale, reverse_steps = 5, args.reverse_steps or 500
    sparsity = 1 - 1 / scale
    transform = lambda x: x[:, :sequence_length, :4, ::scale]
    net_fn = make_net_fn(model, cfg, device)
    methods = methods_for(args, ["pi-iss"])
    if methods != ["pi-iss"]:
        raise ValueError("line-measured supports the PI-ISS method")
    trial = {
        "measurement_file": str(data_path), "position": "pos5",
        "frequency_band_hz": [2900, 3000], "start": start,
        "sequence_length": sequence_length, "snr_db": snr_db,
        "noise_seed": noise_seed, "sampling_seed": sampling_seed,
    }
    if trial_callback.is_complete("pi-iss", trial):
        return

    inverse_scale = np.array([1.0, 1 / impedance, 1 / impedance, 1 / impedance]).reshape(
        1, 1, 4, 1
    )

    def to_physical(x):
        if isinstance(x, torch.Tensor):
            scale_tensor = torch.as_tensor(inverse_scale, dtype=x.dtype, device=x.device)
            return x * scale_tensor
        return x * inverse_scale

    setup_seed(sampling_seed)
    sde = VESDE(cfg, sigma_min=cfg.beta_min, sigma_max=cfg.beta_max, N=reverse_steps)
    prediction = s3gm_sample_1d_ISS_1dtube_res(
        cfg, net_fn, sde, observation, transform,
        alpha=0.1 / np.sqrt(1 - sparsity), beta=0.1, gamma=0.5,
        T_prime_y=sequence_length, T_prime=sequence_length, overlap=overlap,
        denoising_steps=reverse_steps, eta=0, reg_coef_pa=1.0,
        data_scalar=to_physical, physics_start_step=reverse_steps,
        physics_c=343.0, physics_rho0=1.21, physics_dt=1e-5, physics_dx=0.032,
        reg_coef_wave=1e-3, reg_coef_cont=0.0, reg_coef_euler=0.0,
        reg_coef_bc=0.0, normalized_velocity_zero=None,
        corrector=NoneCorrector, snr=0.128, device=str(device), dtype="float",
        eps=1e-12, probability_flow=False, continuous=True,
    )
    prediction_phys = np.asarray(to_physical(prediction))
    reconstruction_32 = prediction_phys[0].transpose(2, 1, 0)
    clean_7 = raw[:, :, start:start + sequence_length]
    noisy_7 = np.asarray(to_physical(observation))[0].transpose(2, 1, 0)
    coordinates_7 = np.zeros((7, 3))
    coordinates_7[:, 0] = 0.48 - np.arange(7) * 0.16
    coordinates_32 = np.zeros((32, 3))
    coordinates_32[:, 0] = 0.48 - np.arange(32) * 0.032
    frequency = 2950.0
    true_positions = np.array([[0.94, 1.17], [-0.98, 1.27]])
    maps = {}
    for name, field, coordinates in (
        ("clean_7_sensor", clean_7, coordinates_7),
        ("noisy_7_sensor", noisy_7, coordinates_7),
        ("reconstructed_32_sensor", reconstruction_32, coordinates_32),
    ):
        x_grid, z_grid, spectrum_db = music_pv_map(field, coordinates, frequency)
        maps[name] = {
            "localization": two_source_localization(
                x_grid, z_grid, spectrum_db, true_positions
            ),
            "interference_region_peak_db": interference_peak(
                x_grid, z_grid, spectrum_db
            ),
        }
    maps["interference_suppression_db_vs_clean_7"] = (
        maps["clean_7_sensor"]["interference_region_peak_db"] -
        maps["reconstructed_32_sensor"]["interference_region_peak_db"]
    )
    maps["interference_suppression_db_vs_noisy_7"] = (
        maps["noisy_7_sensor"]["interference_region_peak_db"] -
        maps["reconstructed_32_sensor"]["interference_region_peak_db"]
    )
    metrics = {
        "dense_ground_truth_available": False,
        "nrmse_reported": False,
        "music_p_plus_v": maps,
    }
    trial_callback("pi-iss", trial, metrics, prediction_phys)


def era5_physical(array: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    mean = scalars[:, 0].reshape(1, 1, -1, 1, 1)
    std = scalars[:, 1].reshape(1, 1, -1, 1, 1)
    return array * std + mean


def evaluate_era5(args: argparse.Namespace, cfg: SimpleNamespace, model: torch.nn.Module,
                  device: torch.device, trial_callback) -> None:
    data_path = SOURCE_ROOT / "data/era5/era5_test.npy"
    scalar_path = SOURCE_ROOT / "results/era5_v0/scalars_era5.npy"
    data = np.load(data_path, mmap_mode="r")
    scalars = np.load(scalar_path)
    normalized = np.asarray(data).transpose(0, 1, 4, 2, 3).astype(np.float32)
    overlap = 2
    sequence_length = aligned_length(100, cfg.num_frames, overlap)
    normalized = normalized[:, :sequence_length]
    ratios = [0.98] if args.profile == "screen" else [
        0.98, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80,
    ]
    seeds = [1] if args.profile == "screen" else [1, 42, 123, 456, 789]
    trials = [(ratio, seed) for ratio in ratios for seed in seeds]
    if args.max_trials is not None:
        trials = trials[:args.max_trials]
    methods = methods_for(args, ["sde", "iss"])
    reverse_steps = args.reverse_steps or 50
    net_fn = make_net_fn(model, cfg, device, args.forward_sleep_seconds)
    channel_names = ["u", "v", "temperature", "pressure"]
    mask_seed = 1

    for missing_ratio, seed in trials:
        mask = np.zeros([cfg.num_channels, cfg.image_size, cfg.image_size])
        spatial_mask = mask_gen(
            [1, cfg.image_size, cfg.image_size], missing_ratio, seed=mask_seed
        )
        mask[:cfg.num_components] = spatial_mask
        mask_torch = torch.as_tensor(mask, device=device)
        sparsity = missing_ratio

        def transform(x):
            if isinstance(x, np.ndarray):
                return x[:, :sequence_length] * mask[np.newaxis, np.newaxis]
            return x[:, :sequence_length] * mask_torch[None, None].to(x.device)

        observation = transform(normalized)
        for method in methods:
            trial = {"missing_ratio": missing_ratio, "observed_ratio": 1 - missing_ratio,
                     "realized_observed_ratio": float(mask[0].mean()),
                     "mask_seed": mask_seed, "sampling_seed": seed, "start": 0,
                     "sequence_length": sequence_length,
                     "reverse_steps": reverse_steps,
                     "forward_sleep_seconds": args.forward_sleep_seconds}
            if trial_callback.is_complete(method, trial):
                continue
            setup_seed(seed)
            sde = VESDE(
                cfg, sigma_min=cfg.beta_min, sigma_max=cfg.beta_max, N=reverse_steps
            )
            if method == "sde":
                prediction = s3gm_sample_2d(
                    cfg, net_fn, sde, observation, transform,
                    alpha=0.1 / np.sqrt(1 - sparsity), beta=0.4, gamma=0.5,
                    T_prime_y=sequence_length, T_prime=sequence_length,
                    overlap=overlap, corrector=NoneCorrector, snr=0.128,
                    device=str(device), dtype="float", eps=1e-12,
                    probability_flow=False, continuous=True,
                )
            else:
                prediction = s3gm_sample_2d_ISS_ERA5(
                    cfg, net_fn, sde, observation, transform,
                    alpha=0.3 / np.sqrt(1 - sparsity), beta=0.4, gamma=0.5,
                    T_prime_y=sequence_length, T_prime=sequence_length,
                    overlap=overlap, corrector=NoneCorrector, n_steps=0, eta=0,
                    reg_coef_pa=0, reg_coef_ar=0, denoising_steps=reverse_steps,
                    snr=0.128, device=str(device), dtype="float", eps=1e-12,
                    probability_flow=False, continuous=True,
                )
            physical_reference = era5_physical(normalized, scalars)
            physical_prediction = era5_physical(prediction, scalars)
            metrics = {
                "normalized": metric_bundle(normalized[0], prediction[0], channel_names),
                "physical": metric_bundle(
                    physical_reference[0], physical_prediction[0], channel_names
                ),
            }
            unobserved = mask[0] == 0
            metrics["normalized_unobserved_only"] = metric_bundle(
                normalized[0][:, :, unobserved], prediction[0][:, :, unobserved],
                channel_names,
            )
            metrics["physical_unobserved_only"] = metric_bundle(
                physical_reference[0][:, :, unobserved],
                physical_prediction[0][:, :, unobserved], channel_names,
            )
            metrics["evaluation_scope"] = {
                "paper_role": "Figure 6f-g",
                "paper_nrmse": "normalized.per_channel.*.notebook_frame_mean_l2_nrmse",
                "paper_pearson": "normalized.per_channel.*.notebook_frame_mean_pearson",
                "mask_fixed_across_sampling_seeds": True,
                "source_notebook": str(SOURCE_ROOT / "notebooks/era5/nrmse.ipynb"),
            }
            trial_callback(method, trial, metrics, physical_prediction)


EVALUATORS = {
    "2dsound": evaluate_2dsound,
    "tube-pressure-heldout": evaluate_tube_pressure_heldout,
    "tube": evaluate_tube,
    "line-sim": evaluate_line_sim,
    "line-measured": evaluate_line_measured,
    "era5": evaluate_era5,
}


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:"):
        raise ValueError("an explicit CUDA device is required")
    gpu_id = int(args.device.split(":", 1)[1])
    if gpu_id not in ALLOWED_GPU_IDS:
        raise ValueError(f"GPU {gpu_id} is outside the campaign allow-list")
    torch.cuda.set_device(gpu_id)
    device = torch.device(args.device)
    checkpoints = [parse_checkpoint(spec) for spec in args.checkpoint]
    if len({name for name, _ in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint names must be unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "sampling_metrics.json"
    if report_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {report_path}; use --resume")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {
            "protocol": {
                "case": args.case,
                "config": str(args.config.resolve()),
                "profile": args.profile,
                "weights": args.weights,
                "device": args.device,
                "power_ceiling_w": args.power_ceiling_w,
                "reverse_steps_override": args.reverse_steps,
                "checkpoint_paths": {name: str(path.resolve()) for name, path in checkpoints},
                "nrmse_definitions": {
                    "global_l2_nrmse": "||prediction-reference||_2 / ||reference||_2 over a channel and full trial",
                    "notebook_frame_mean_l2_nrmse": "mean_t ||prediction_t-reference_t||_2 / ||reference_t||_2",
                    "global_range_normalized_rmse": "RMSE(prediction-reference) / (max(reference)-min(reference))",
                    "notebook_frame_mean_range_normalized_rmse": "mean_t RMSE_space(prediction_t-reference_t) / (max(reference)-min(reference))",
                },
            },
            "results": [],
            "errors": [],
        }
        atomic_json(report_path, report)

    completed = {
        (entry["checkpoint"], entry["method"], json.dumps(entry["trial"], sort_keys=True))
        for entry in report["results"]
    }
    watchdog = PowerWatchdog(
        gpu_id, args.power_ceiling_w, args.power_poll_seconds,
        args.output_dir / "power_guard_event.json",
    )
    watchdog.start()
    try:
        cfg = load_config(args.config, args.device)
        for checkpoint_name, checkpoint_path in checkpoints:
            model, state = load_model(cfg, checkpoint_path, args.weights, device)

            def save_trial(method: str, trial: dict, metrics: dict,
                           prediction: np.ndarray) -> None:
                key = (checkpoint_name, method, json.dumps(trial, sort_keys=True))
                if key in completed:
                    return
                prediction_path = None
                if args.save_predictions:
                    token = len(report["results"])
                    prediction_path = args.output_dir / f"prediction_{token:04d}.npz"
                    np.savez_compressed(prediction_path, prediction=np.asarray(prediction))
                report["results"].append({
                    "checkpoint": checkpoint_name,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_epoch": state.get("epoch"),
                    "checkpoint_iteration": state.get("iteration"),
                    "fine_tune_iteration": state.get("fine_tune_iteration"),
                    "method": method,
                    "trial": trial,
                    "metrics": metrics,
                    "prediction_path": str(prediction_path) if prediction_path else None,
                })
                completed.add(key)
                report["protocol"]["peak_power_draw_w"] = watchdog.peak_w
                atomic_json(report_path, report)

            save_trial.is_complete = lambda method, trial: (
                checkpoint_name, method, json.dumps(trial, sort_keys=True)
            ) in completed

            try:
                EVALUATORS[args.case](args, cfg, model, device, save_trial)
            except Exception as exc:
                report["errors"].append({
                    "checkpoint": checkpoint_name,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
                atomic_json(report_path, report)
                raise
            finally:
                del model
                torch.cuda.empty_cache()
    finally:
        watchdog.stop()
        report["protocol"]["peak_power_draw_w"] = watchdog.peak_w
        atomic_json(report_path, report)
    print(json.dumps({
        "output": str(report_path),
        "results": len(report["results"]),
        "errors": len(report["errors"]),
        "peak_power_draw_w": watchdog.peak_w,
    }))


if __name__ == "__main__":
    main()
