#!/usr/bin/env python
"""Evaluate free-field reconstruction at random observation rates and compare FPE checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_freefield_manuscript_protocol_full_metrics import (  # noqa: E402
    CHANNELS,
    MRSM_ROOT,
    SOURCE_ROOT,
    DataScaler,
    PowerWatchdog,
    ThrottledProgress,
    aggregate_metrics,
    atomic_json,
    load_config,
    load_model,
    make_net_fn,
    parse_checkpoint,
)

from sampler import VESDE, setup_seed  # noqa: E402
from sampler.utils import (  # noqa: E402
    NoneCorrector,
    s3gm_sample_2d,
    s3gm_sample_2d_ISS_2dsound,
    s3gm_sample_2d_ISS_2dsound_res,
)
import sampler.utils as sampler_utils  # noqa: E402


SOURCE_NOTEBOOK = SOURCE_ROOT / "notebooks/free_field/random_observations.ipynb"
SOURCE_CSV = Path(str(SOURCE_ROOT / 'source_data/experiment_results_20260331_204711.csv'))
TEST_DATA = Path(str(SOURCE_ROOT / 'data/anechoic_high_prec_dataset_last700_test_2.npy'))
SCALAR_PATH = SOURCE_ROOT / "results/2DSound_v22/scalar.npy"
FREQUENCY_INDEX = 6
FREQUENCY_HZ = 375
START = 0
SEEDS = (1, 42, 123, 456, 789)
OBSERVABLE_PCTS = (2, 3, 5, 8, 10, 15, 20)
METHODS = ("SDE", "ISS", "PI-ISS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--config", type=Path, default=SOURCE_ROOT / "results/2DSound_v22/config.json")
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observable-pcts", default="3,5,8,10,15,20")
    parser.add_argument("--methods", default="SDE,ISS,PI-ISS")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-parity-atol", type=float, default=5e-4)
    parser.add_argument("--source-parity-rtol", type=float, default=5e-6)
    parser.add_argument(
        "--allow-source-parity-failure",
        action="store_true",
        help="finish evaluation and record out-of-tolerance reference comparisons",
    )
    parser.add_argument("--power-ceiling-w", type=float, default=250.0)
    parser.add_argument("--power-poll-seconds", type=float, default=0.25)
    parser.add_argument("--step-sleep-seconds", type=float, default=0.2)
    return parser.parse_args()


def load_paper_rows() -> dict[int, dict[str, str]]:
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(f"free-field random-observation source CSV is missing: {SOURCE_CSV}")
    with SOURCE_CSV.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["observable_pct"]): row for row in csv.DictReader(handle)}


def metric_bundle_random(
    reference: np.ndarray,
    prediction: np.ndarray,
    observed_mask: np.ndarray,
) -> dict:
    metrics = aggregate_metrics(reference, prediction)
    unobserved = ~observed_mask.astype(bool)
    metrics["unobserved_only"] = aggregate_metrics(
        reference[..., unobserved], prediction[..., unobserved]
    )
    return metrics


def make_observation(
    cfg,
    normalized: np.ndarray,
    observable_pct: int,
    seed: int,
    device: torch.device,
):
    missing_ratio = 1.0 - observable_pct / 100.0
    setup_seed(seed)
    mask = np.zeros(
        [cfg.num_components + cfg.num_conditions, cfg.image_size, cfg.image_size],
        dtype=np.float32,
    )
    mask[:cfg.num_components] = sampler_utils.mask_gen(
        [1, cfg.image_size, cfg.image_size], missing_ratio, seed=seed
    )
    mask_torch = torch.from_numpy(mask).to(device)

    def transform(value):
        if isinstance(value, np.ndarray):
            return value[:, :106] * mask[np.newaxis, np.newaxis]
        return value[:, :106] * mask_torch[None, None].to(value.device)

    return transform, transform(normalized), mask[0].astype(bool), missing_ratio


def sample_method(
    method: str,
    cfg,
    net_fn,
    normalized: np.ndarray,
    scaler: DataScaler,
    observable_pct: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    transform, observation, observed_mask, sparsity = make_observation(
        cfg, normalized, observable_pct, seed, device
    )
    setup_seed(seed)
    sde = VESDE(cfg, sigma_min=cfg.beta_min, sigma_max=cfg.beta_max, N=100)
    common = dict(
        T_prime_y=106,
        T_prime=106,
        overlap=2,
        corrector=NoneCorrector,
        n_steps=0,
        snr=0.128,
        device=str(device),
        dtype="float",
        eps=1e-12,
        probability_flow=False,
        continuous=True,
    )
    with torch.no_grad():
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
                reg_coef_pa=1060,
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
                reg_coef_pa=1060,
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
    return np.asarray(scaler.unnormalize(prediction)), observed_mask


def validate_source_parity(
    report: dict,
    observable_pcts: tuple[int, ...],
    methods: tuple[str, ...],
    paper_rows: dict[int, dict[str, str]],
    atol: float,
    rtol: float,
) -> list[dict]:
    checks = []
    method_prefix = {"SDE": "sde", "ISS": "iss", "PI-ISS": "pi_iss"}
    channel_csv = {"pressure": "p", "vx": "vx", "vy": "vy"}
    for observable_pct in observable_pcts:
        for method in methods:
            entries = [
                item for item in report["results"]
                if item["checkpoint"] == "source"
                and item["observable_pct"] == observable_pct
                and item["method"] == method
            ]
            if len(entries) != len(SEEDS):
                raise RuntimeError(
                    f"Reference comparison requires {len(SEEDS)} seeds for {observable_pct}% {method}; got {len(entries)}"
                )
            for channel in CHANNELS:
                values = np.asarray([
                    item["metrics"]["per_channel"][channel]["frame_mean_l2_nrmse"]
                    for item in entries
                ])
                correlations = np.asarray([
                    item["metrics"]["per_channel"][channel]["frame_mean_pearson"]
                    for item in entries
                ])
                prefix = f"{method_prefix[method]}_{channel_csv[channel]}"
                for metric_name, measured_values, suffix in (
                    ("nrmse_mean", values, "rmse_mean"),
                    ("nrmse_std", values, "rmse_std"),
                    ("pearson_mean", correlations, "corr_mean"),
                    ("pearson_std", correlations, "corr_std"),
                ):
                    measured = float(
                        measured_values.mean() if metric_name.endswith("mean") else measured_values.std()
                    )
                    paper_value = float(paper_rows[observable_pct][f"{prefix}_{suffix}"])
                    delta = abs(measured - paper_value)
                    tolerance = atol + rtol * abs(paper_value)
                    checks.append({
                        "observable_pct": observable_pct,
                        "method": method,
                        "channel": channel,
                        "metric": metric_name,
                        "measured": measured,
                        "paper": paper_value,
                        "absolute_delta": delta,
                        "tolerance": tolerance,
                        "pass": delta <= tolerance,
                    })
    return checks


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda:"):
        raise ValueError("an explicit CUDA device is required")
    gpu_id = int(args.device.split(":", 1)[1])
    torch.cuda.set_device(gpu_id)
    device = torch.device(args.device)
    observable_pcts = tuple(int(item) for item in args.observable_pcts.split(","))
    methods = tuple(item.strip() for item in args.methods.split(","))
    if not set(observable_pcts).issubset(OBSERVABLE_PCTS) or not set(methods).issubset(METHODS):
        raise ValueError("requested observable proportions or methods fall outside the frozen protocol")
    checkpoints = [parse_checkpoint(spec) for spec in args.checkpoint]
    if not checkpoints or checkpoints[0][0] != "source":
        raise ValueError("the source checkpoint must be listed first as NAME=source")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --resume")

    if args.step_sleep_seconds:
        original_tqdm = sampler_utils.tqdm
        sampler_utils.tqdm = lambda *positional, **keywords: ThrottledProgress(
            original_tqdm, args.step_sleep_seconds, *positional, **keywords
        )

    if args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        report = {
            "protocol": {
                "source_notebook": str(SOURCE_NOTEBOOK),
                "source_csv": str(SOURCE_CSV),
                "frequency_index": FREQUENCY_INDEX,
                "frequency_hz": FREQUENCY_HZ,
                "start": START,
                "frames": 106,
                "observable_pcts": list(observable_pcts),
                "random_seeds": list(SEEDS),
                "methods": list(methods),
                "reverse_steps": 100,
                "model_mode": "eval",
                "source_parity_atol": args.source_parity_atol,
                "source_parity_rtol": args.source_parity_rtol,
                "checkpoint_paths": {name: str(path) for name, path in checkpoints},
            },
            "results": [],
            "source_parity": [],
            "errors": [],
        }
        atomic_json(args.output, report)

    completed = {
        (item["checkpoint"], item["observable_pct"], item["seed"], item["method"])
        for item in report["results"]
    }
    raw = np.load(TEST_DATA, mmap_mode="r")
    reference = np.asarray(raw[FREQUENCY_INDEX:FREQUENCY_INDEX + 1, START:START + 106]).transpose(0, 1, 4, 2, 3)
    scaler = DataScaler(SCALAR_PATH, device)
    normalized = scaler.normalize(reference).astype(np.float32)
    cfg = load_config(args.config, args.device)
    paper_rows = load_paper_rows()
    watchdog = PowerWatchdog(
        gpu_id,
        args.power_ceiling_w,
        args.power_poll_seconds,
        args.output.with_name("power_guard_event.json"),
    )
    watchdog.start()
    try:
        for checkpoint_name, checkpoint_path in checkpoints:
            model, state = load_model(cfg, checkpoint_path, device, "eval")
            net_fn = make_net_fn(model, cfg, device)
            for observable_pct in observable_pcts:
                for seed in SEEDS:
                    for method in methods:
                        key = (checkpoint_name, observable_pct, seed, method)
                        if key in completed:
                            continue
                        started = time.time()
                        try:
                            prediction, observed_mask = sample_method(
                                method,
                                cfg,
                                net_fn,
                                normalized,
                                scaler,
                                observable_pct,
                                seed,
                                device,
                            )
                            metrics = metric_bundle_random(
                                reference[0], prediction[0], observed_mask
                            )
                            report["results"].append({
                                "checkpoint": checkpoint_name,
                                "checkpoint_path": str(checkpoint_path),
                                "checkpoint_epoch": state.get("epoch"),
                                "checkpoint_iteration": state.get("iteration"),
                                "fine_tune_iteration": state.get("fine_tune_iteration"),
                                "observable_pct": observable_pct,
                                "seed": seed,
                                "method": method,
                                "metrics": metrics,
                                "runtime_seconds": time.time() - started,
                            })
                            completed.add(key)
                            atomic_json(args.output, report)
                            print(
                                f"DONE checkpoint={checkpoint_name} method={method} "
                                f"obs={observable_pct} seed={seed} "
                                f"nRMSE={metrics['macro_frame_mean_l2_nrmse']:.9f}",
                                flush=True,
                            )
                        except Exception as exc:
                            report["errors"].append({
                                "checkpoint": checkpoint_name,
                                "observable_pct": observable_pct,
                                "seed": seed,
                                "method": method,
                                "error": repr(exc),
                            })
                            atomic_json(args.output, report)
                            raise
                    torch.cuda.empty_cache()
            if checkpoint_name == "source":
                report["source_parity"] = validate_source_parity(
                    report,
                    observable_pcts,
                    methods,
                    paper_rows,
                    args.source_parity_atol,
                    args.source_parity_rtol,
                )
                atomic_json(args.output, report)
                failures = [item for item in report["source_parity"] if not item["pass"]]
                if failures and not args.allow_source_parity_failure:
                    raise RuntimeError(f"Free-field random-observation differences exceeded tolerance: {failures[:3]}")
            del model
            torch.cuda.empty_cache()
    finally:
        watchdog.stop()
        report["peak_power_w"] = watchdog.peak_w
        atomic_json(args.output, report)


if __name__ == "__main__":
    main()
