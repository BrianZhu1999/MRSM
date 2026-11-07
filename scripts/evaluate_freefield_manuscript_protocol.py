#!/usr/bin/env python
"""Evaluate the free-field downsampling free-field nRMSE settings.

The calculation uses the source notebook's frequency, temporal windows,
downsampling factors, sampler settings, random seed and frame-wise nRMSE.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1]
MRSM_ROOT = SOURCE_ROOT
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from models import get_ema, get_model  # noqa: E402
from sampler import VESDE, setup_seed  # noqa: E402
from sampler.utils import (  # noqa: E402
    NoneCorrector,
    s3gm_sample_2d,
    s3gm_sample_2d_ISS_2dsound,
    s3gm_sample_2d_ISS_2dsound_res,
)
import sampler.utils as sampler_utils  # noqa: E402


EXPECTED_CSV = Path(str(SOURCE_ROOT / 'source_data/baselines_comparison_20260331_213441.csv'))
TEST_DATA = Path(str(SOURCE_ROOT / 'data/anechoic_high_prec_dataset_last700_test_2.npy'))
SCALAR_PATH = SOURCE_ROOT / "results/2DSound_v22/scalar.npy"
FREQUENCY_INDEX = 6
FREQUENCY_HZ = 375
STARTS = (95, 15, 30, 158, 128)
SCALES = (2, 3, 4, 5, 6)
METHODS = ("SDE", "ISS", "PI-ISS")
CHANNELS = ("pressure", "vx", "vy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--config", type=Path, default=SOURCE_ROOT / "results/2DSound_v22/config.json")
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", default="2,3,4,5,6")
    parser.add_argument("--starts", default="95,15,30,158,128")
    parser.add_argument(
        "--pairs",
        help="optional comma-separated SCALE:START pairs for non-overlapping GPU manifests",
    )
    parser.add_argument("--methods", default="SDE,ISS,PI-ISS")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--model-mode",
        choices=("manuscript", "eval"),
        default="manuscript",
        help="use the source notebook train-mode dropout during sampling",
    )
    parser.add_argument(
        "--source-parity-tolerance",
        type=float,
        default=5e-4,
        help="maximum absolute nRMSE difference from the free-field downsampling CSV for checkpoint NAME=source",
    )
    parser.add_argument("--power-ceiling-w", type=float, default=250.0)
    parser.add_argument("--power-poll-seconds", type=float, default=0.25)
    parser.add_argument(
        "--step-sleep-seconds",
        type=float,
        default=0.0,
        help="wall-clock throttle after each reverse step; tensor operations are unchanged",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def parse_checkpoint(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"checkpoint must be NAME=PATH: {spec}")
    name, raw_path = spec.split("=", 1)
    path = Path(raw_path)
    if not name or not path.is_file():
        raise FileNotFoundError(spec)
    return name, path


def load_config(path: Path, device: str) -> SimpleNamespace:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["num_channels"] = payload["num_components"] + payload["num_conditions"]
    return SimpleNamespace(**payload)


def normalise_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(state)
    if keys and all(key.startswith("module.") for key in keys):
        return {key[len("module."):]: value for key, value in state.items()}
    if any(key.startswith("module.") for key in keys):
        raise RuntimeError("mixed DataParallel prefixes in checkpoint")
    return state


def load_model(
    cfg: SimpleNamespace,
    path: Path,
    device: torch.device,
    model_mode: str,
) -> tuple[torch.nn.Module, dict]:
    model = get_model(cfg).to(device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(normalise_state_dict(state["model"]), strict=True)
    ema_state = state.get("ema")
    if ema_state is None:
        raise RuntimeError(f"checkpoint has no EMA state: {path}")
    ema = get_ema(model.parameters(), decay=float(ema_state["decay"]))
    ema.load_state_dict(ema_state)
    ema.shadow_params = [parameter.to(device=device) for parameter in ema.shadow_params]
    ema.copy_to(model.parameters())
    if model_mode == "manuscript":
        model.train()
    else:
        model.eval()
    return model, state


def load_paper_references() -> dict[tuple[str, int, int], float]:
    if not EXPECTED_CSV.is_file():
        raise FileNotFoundError(f"free-field downsampling source CSV is missing: {EXPECTED_CSV}")
    references: dict[tuple[str, int, int], float] = {}
    with EXPECTED_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scale = int(row["Scale"])
            for method in METHODS:
                for trial_index in range(1, len(STARTS) + 1):
                    references[(method, scale, trial_index)] = float(
                        row[f"{method}_trial_{trial_index}"]
                    )
    return references


class DataScaler:
    def __init__(self, path: Path, device: torch.device):
        mean, std = np.load(path, allow_pickle=True)
        self.mean_np = np.asarray(mean).transpose(0, 1, 4, 2, 3)
        self.std_np = np.asarray(std).transpose(0, 1, 4, 2, 3)
        self.mean_torch = torch.as_tensor(self.mean_np, dtype=torch.float32, device=device)
        self.std_torch = torch.as_tensor(self.std_np, dtype=torch.float32, device=device)

    def normalize(self, array: np.ndarray) -> np.ndarray:
        return (array - self.mean_np) / self.std_np

    def unnormalize(self, array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(array, torch.Tensor):
            return array * self.std_torch + self.mean_torch
        return array * self.std_np + self.mean_np


def make_net_fn(model: torch.nn.Module, cfg: SimpleNamespace, device: torch.device):
    latent_mask = torch.ones([cfg.num_frames, 1, 1, 1], device=device)
    obs_mask = torch.zeros([cfg.num_frames, 1, 1, 1], device=device)
    frame_indices = torch.arange(cfg.num_frames, device=device)

    def repeat(value: torch.Tensor, number: int) -> torch.Tensor:
        return torch.stack([value for _ in range(number)])

    return lambda x, t: model(
        x,
        x0=x,
        timesteps=t,
        latent_mask=repeat(latent_mask, len(x)),
        obs_mask=repeat(obs_mask, len(x)),
        frame_indices=repeat(frame_indices, len(x)),
    )[0]


def frame_l2_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    per_channel = {}
    for channel_index, channel_name in enumerate(CHANNELS):
        gt = reference[:, channel_index].reshape(len(reference), -1).astype(np.float64)
        pred = prediction[:, channel_index].reshape(len(prediction), -1).astype(np.float64)
        ratios = np.linalg.norm(pred - gt, axis=1) / np.linalg.norm(gt, axis=1)
        per_channel[channel_name] = {
            "mean": float(ratios.mean()),
            "std_over_frames": float(ratios.std()),
            "per_frame": ratios.tolist(),
        }
    macro = float(np.mean([per_channel[name]["mean"] for name in CHANNELS]))
    return {"per_channel": per_channel, "macro_frame_mean_l2_nrmse": macro}


def query_power(gpu_id: int) -> float:
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        text=True,
        timeout=5,
    )
    return float(output.strip())


class PowerWatchdog:
    def __init__(self, gpu_id: int, ceiling: float, poll_seconds: float, marker: Path):
        self.gpu_id = gpu_id
        self.ceiling = ceiling
        self.poll_seconds = poll_seconds
        self.marker = marker
        self.peak_w = query_power(gpu_id)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        if self.peak_w >= self.ceiling:
            raise RuntimeError(f"GPU {self.gpu_id} is already at {self.peak_w:.2f} W")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)

    def run(self) -> None:
        while not self.stop_event.wait(self.poll_seconds):
            try:
                power = query_power(self.gpu_id)
                self.peak_w = max(self.peak_w, power)
                if power >= self.ceiling:
                    atomic_json(self.marker, {"gpu": self.gpu_id, "power_w": power, "ceiling_w": self.ceiling})
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
            except Exception as exc:
                atomic_json(self.marker, {"monitor_error": repr(exc)})
                return


class ThrottledProgress:
    """Proxy tqdm while inserting a synchronized, computation-free pause."""

    def __init__(self, original_tqdm, sleep_seconds: float, *args, **kwargs):
        self.progress = original_tqdm(*args, **kwargs)
        self.sleep_seconds = sleep_seconds

    def _pause(self) -> None:
        if self.sleep_seconds > 0:
            torch.cuda.synchronize()
            time.sleep(self.sleep_seconds)

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.progress.__exit__(exc_type, exc_value, traceback)

    def __iter__(self):
        for item in self.progress:
            yield item
            self._pause()

    def update(self, count: int = 1):
        result = self.progress.update(count)
        self._pause()
        return result

    def __getattr__(self, name):
        return getattr(self.progress, name)


def sample_method(
    method: str,
    cfg: SimpleNamespace,
    net_fn,
    normalized: np.ndarray,
    scaler: DataScaler,
    scale: int,
    device: torch.device,
) -> np.ndarray:
    overlap = 2
    target = 100
    sequence_length = int(np.ceil((target - overlap) / (cfg.num_frames - overlap))) * (cfg.num_frames - overlap) + overlap
    sparsity = 1 - 1 / scale ** 2
    transform = lambda x: x[:, :sequence_length, :cfg.num_components, ::scale, ::scale]
    observation = transform(normalized)
    setup_seed(123)
    sde = VESDE(cfg, sigma_min=cfg.beta_min, sigma_max=cfg.beta_max, N=100)
    common = dict(
        T_prime_y=sequence_length,
        T_prime=sequence_length,
        overlap=overlap,
        corrector=NoneCorrector,
        n_steps=0,
        snr=0.128,
        device=str(device),
        dtype="float",
        eps=1e-12,
        probability_flow=False,
        continuous=True,
    )
    if method == "SDE":
        prediction = s3gm_sample_2d(
            cfg,
            net_fn,
            sde,
            observation,
            transform,
            alpha=0.5 / np.sqrt(1 - sparsity),
            beta=0.4,
            gamma=0.5,
            **common,
        )
    elif method == "ISS":
        prediction = s3gm_sample_2d_ISS_2dsound(
            cfg,
            net_fn,
            sde,
            observation,
            transform,
            alpha=0.5 / np.sqrt(1 - sparsity),
            beta=0.4,
            gamma=0.5,
            eta=0,
            reg_coef_pa=10 * sequence_length,
            reg_coef_ar=1e2,
            denoising_steps=100,
            **common,
        )
    elif method == "PI-ISS":
        prediction = s3gm_sample_2d_ISS_2dsound_res(
            cfg,
            net_fn,
            sde,
            observation,
            transform,
            alpha=1.4 / np.sqrt(1 - sparsity),
            beta=0.4,
            gamma=0.5,
            eta=0,
            use_residual_objective=False,
            reg_coef_pa=10 * sequence_length,
            reg_coef_ar=1e2,
            denoising_steps=100,
            data_scalar=scaler.unnormalize,
            physics_start_step=100,
            reg_coef_wave=5e-8,
            reg_coef_cont=5.0,
            reg_coef_euler=5.0,
            physics_c=343.0,
            physics_rho0=1.21,
            physics_dt=1e-4,
            physics_dx=1 / 21,
            physics_dy=1 / 21,
            **common,
        )
    else:
        raise ValueError(method)
    return np.asarray(scaler.unnormalize(prediction))


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:"):
        raise ValueError("an explicit CUDA device is required")
    gpu_id = int(args.device.split(":", 1)[1])
    torch.cuda.set_device(gpu_id)
    device = torch.device(args.device)
    if args.step_sleep_seconds < 0:
        raise ValueError("step sleep must be non-negative")
    if args.step_sleep_seconds:
        original_tqdm = sampler_utils.tqdm
        sampler_utils.tqdm = lambda *positional, **keywords: ThrottledProgress(
            original_tqdm, args.step_sleep_seconds, *positional, **keywords
        )
    scales = tuple(int(item) for item in args.scales.split(","))
    starts = tuple(int(item) for item in args.starts.split(","))
    requested_pairs = (
        tuple(tuple(int(value) for value in item.split(":")) for item in args.pairs.split(","))
        if args.pairs
        else tuple((scale, start) for scale in scales for start in starts)
    )
    methods = tuple(item.strip() for item in args.methods.split(","))
    if (
        any(len(pair) != 2 or pair[0] not in SCALES or pair[1] not in STARTS for pair in requested_pairs)
        or not set(methods).issubset(METHODS)
    ):
        raise ValueError("requested scales, starts or methods fall outside the frozen protocol")

    checkpoints = [parse_checkpoint(spec) for spec in args.checkpoint]
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --resume")
    if args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        report = {
            "protocol": {
                "source_notebook": str(SOURCE_ROOT / "2d_sound_Error_evaluation.ipynb"),
                "source_csv": str(EXPECTED_CSV),
                "frequency_index": FREQUENCY_INDEX,
                "frequency_hz": FREQUENCY_HZ,
                "starts": list(STARTS),
                "scales": list(SCALES),
                "requested_pairs": [list(pair) for pair in requested_pairs],
                "sampling_seed": 123,
                "reverse_steps": 100,
                "target_frames": 100,
                "actual_frames": 106,
                "step_sleep_seconds": args.step_sleep_seconds,
                "model_mode": args.model_mode,
                "source_parity_tolerance": args.source_parity_tolerance,
                "nrmse": "mean_t ||prediction_t-reference_t||_2 / ||reference_t||_2, then mean over pressure,vx,vy",
                "checkpoint_paths": {name: str(path) for name, path in checkpoints},
            },
            "results": [],
            "errors": [],
        }
        atomic_json(args.output, report)

    completed = {
        (entry["checkpoint"], entry["method"], entry["scale"], entry["start"])
        for entry in report["results"]
    }
    raw_data = np.load(TEST_DATA, mmap_mode="r")
    paper_references = load_paper_references()
    scaler = DataScaler(SCALAR_PATH, device)
    cfg = load_config(args.config, args.device)
    watchdog = PowerWatchdog(gpu_id, args.power_ceiling_w, args.power_poll_seconds, args.output.with_name("power_guard_event.json"))
    watchdog.start()
    try:
        for checkpoint_name, checkpoint_path in checkpoints:
            model, state = load_model(cfg, checkpoint_path, device, args.model_mode)
            net_fn = make_net_fn(model, cfg, device)
            for scale in scales:
                for start in starts:
                    if (scale, start) not in requested_pairs:
                        continue
                    sequence_length = 106
                    reference = np.asarray(
                        raw_data[FREQUENCY_INDEX:FREQUENCY_INDEX + 1, start:start + sequence_length]
                    ).transpose(0, 1, 4, 2, 3)
                    normalized = scaler.normalize(reference).astype(np.float32)
                    for method in methods:
                        key = (checkpoint_name, method, scale, start)
                        if key in completed:
                            continue
                        started = time.time()
                        try:
                            prediction = sample_method(method, cfg, net_fn, normalized, scaler, scale, device)
                            metrics = frame_l2_metrics(reference[0], prediction[0])
                            result_entry = {
                                "checkpoint": checkpoint_name,
                                "checkpoint_path": str(checkpoint_path),
                                "checkpoint_epoch": state.get("epoch"),
                                "checkpoint_iteration": state.get("iteration"),
                                "fine_tune_iteration": state.get("fine_tune_iteration"),
                                "method": method,
                                "scale": scale,
                                "start": start,
                                "metrics": metrics,
                                "runtime_seconds": time.time() - started,
                            }
                            if checkpoint_name == "source":
                                trial_index = STARTS.index(start) + 1
                                paper_value = paper_references[(method, scale, trial_index)]
                                measured = metrics["macro_frame_mean_l2_nrmse"]
                                absolute_delta = abs(measured - paper_value)
                                result_entry["paper_reference_nrmse"] = paper_value
                                result_entry["paper_absolute_delta"] = absolute_delta
                                result_entry["source_parity_pass"] = (
                                    absolute_delta <= args.source_parity_tolerance
                                )
                            report["results"].append(result_entry)
                            completed.add(key)
                            atomic_json(args.output, report)
                            if checkpoint_name == "source" and not result_entry["source_parity_pass"]:
                                raise RuntimeError(
                                    "reference comparison exceeded tolerance for "
                                    f"method={method} scale={scale} start={start}: "
                                    f"measured={measured:.9f}, paper={paper_value:.9f}, "
                                    f"delta={absolute_delta:.9f}"
                                )
                            print(
                                f"DONE checkpoint={checkpoint_name} method={method} scale={scale} "
                                f"start={start} nRMSE={metrics['macro_frame_mean_l2_nrmse']:.9f}",
                                flush=True,
                            )
                        except Exception as exc:
                            report["errors"].append({
                                "checkpoint": checkpoint_name,
                                "method": method,
                                "scale": scale,
                                "start": start,
                                "error": repr(exc),
                            })
                            atomic_json(args.output, report)
                            raise
                    torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()
    finally:
        watchdog.stop()
        report["peak_power_w"] = watchdog.peak_w
        atomic_json(args.output, report)


if __name__ == "__main__":
    main()
