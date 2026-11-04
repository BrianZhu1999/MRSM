#!/usr/bin/env python
"""Train the score prior or fine-tune it with Score FPE regularization.

--total-steps counts optimizer updates, independently of reverse sampling steps.
Relative paths are resolved from the repository root.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import get_model, get_ema
from sampler import VESDE, setup_seed
from trainer import get_dataset, loss_fn_video, fpe_regularizer_loss
from trainer.utils import restore_checkpoint, save_checkpoint


def repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args(stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if stage is None:
        parser.add_argument("--stage", choices=("dsm", "fpe"), required=True)
    else:
        parser.set_defaults(stage=stage)
    parser.add_argument("--config", type=repo_path, required=True)
    parser.add_argument("--data", type=repo_path, help="override config.data_location")
    parser.add_argument("--checkpoint", type=repo_path, help="base score checkpoint for FPE fine-tuning")
    parser.add_argument("--output-dir", type=repo_path, required=True)
    parser.add_argument("--total-steps", type=int, required=True, help="target number of optimizer updates")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fpe-alpha", type=float, default=1e-4)
    parser.add_argument("--fpe-beta", type=float, default=0.0)
    parser.add_argument("--fpe-m", type=int, default=1)
    parser.add_argument("--h-s", type=float, default=0.001)
    parser.add_argument("--h-d", type=float, default=0.0005)
    parser.add_argument("--grad-clip", type=float, help="gradient norm limit; default: 1 for FPE, disabled for DSM")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true", help="continue output-dir/checkpoint.pth")
    parser.add_argument("--power-ceiling-w", type=float, help="stop and save if the selected physical GPU reaches this power")
    args = parser.parse_args()
    if args.total_steps < 1 or args.checkpoint_every < 1:
        parser.error("total-steps and checkpoint-every must be positive")
    if args.stage == "fpe" and not args.resume and args.checkpoint is None:
        parser.error("FPE fine-tuning requires --checkpoint")
    if args.stage == "dsm" and args.checkpoint is not None:
        parser.error("use --resume to continue DSM, or train_FT.py to fine-tune a base checkpoint")
    if args.fpe_alpha < 0 or args.fpe_beta < 0 or args.fpe_m < 1 or min(args.h_s, args.h_d) <= 0:
        parser.error("FPE weights must be nonnegative, m positive, and offsets positive")
    if args.grad_clip is not None and args.grad_clip <= 0:
        parser.error("grad-clip must be positive")
    if args.power_ceiling_w is not None and args.power_ceiling_w <= 0:
        parser.error("power-ceiling-w must be positive")
    return args


def rng_state(device):
    result = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device)
    return result


def restore_rng(saved, device):
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda"].cpu(), device)


def gpu_power(device):
    # nvidia-smi uses the physical GPU UUID, independent of CUDA_VISIBLE_DEVICES.
    gpu_uuid = "GPU-" + str(torch.cuda.get_device_properties(device).uuid).removeprefix("GPU-")
    value = subprocess.check_output(["nvidia-smi", "-i", gpu_uuid,
        "--query-gpu=power.draw", "--format=csv,noheader,nounits"], text=True, timeout=5)
    return float(value.strip())


def main(stage=None):
    args = parse_args(stage)
    cfg = SimpleNamespace(**json.loads(args.config.read_text(encoding="utf-8")))
    cfg.device = args.device
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.power_ceiling_w is not None:
        if device.type != "cuda":
            raise ValueError("GPU power monitoring requires a CUDA device")
        if gpu_power(device) >= args.power_ceiling_w:
            raise RuntimeError("GPU power is already at or above the requested ceiling")
    cfg.seed = args.seed if args.seed is not None else getattr(cfg, "seed", 42)
    cfg.batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    cfg.lr = args.lr if args.lr is not None else cfg.lr
    cfg.num_channels = cfg.num_components + cfg.num_conditions
    cfg.data_location = str(args.data or repo_path(cfg.data_location))
    cfg.results_path = str(args.output_dir)
    if cfg.batch_size < 1 or cfg.lr <= 0:
        raise ValueError("batch size and learning rate must be positive")
    if not Path(cfg.data_location).is_file():
        raise FileNotFoundError(cfg.data_location)
    clip = args.grad_clip if args.grad_clip is not None else (1.0 if args.stage == "fpe" else None)
    settings = dict(stage=args.stage, config=vars(cfg), fpe_alpha=args.fpe_alpha,
        fpe_beta=args.fpe_beta, fpe_m=args.fpe_m, h_s=args.h_s, h_d=args.h_d, grad_clip=clip)
    checkpoint_path = args.output_dir / "checkpoint.pth"
    if args.resume:
        recorded = json.loads((args.output_dir / "training_settings.json").read_text())
        if settings != recorded:
            raise ValueError("resume settings differ from training_settings.json")
    else:
        # A new directory keeps source weights and earlier training runs intact.
        args.output_dir.mkdir(parents=True, exist_ok=False)
    setup_seed(cfg.seed)
    model = get_model(cfg).to(device)
    ema = get_ema(model.parameters(), decay=cfg.ema_rate)
    if args.stage == "fpe" and not args.resume:
        restore_checkpoint(args.checkpoint, dict(model=model, ema=ema), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    state = dict(model=model, ema=ema, optimizer=optimizer, epoch=0, iteration=0,
                 loss_train=[], loss_val=[], training_settings=settings)
    if args.resume:
        restore_checkpoint(checkpoint_path, state, device)
        if "rng_state" not in state:
            raise ValueError("resume checkpoint has no random-generator state")
        if state["iteration"] >= args.total_steps:
            raise ValueError("checkpoint has already reached the requested total steps")
    dataset = get_dataset(cfg, np.load(cfg.data_location, mmap_mode="r"), train=True)
    # Dataset items draw random windows; isolate the iterator seed from model RNG.
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, generator=generator)
    iterator = iter(loader)
    if args.resume:
        restore_rng(state["rng_state"], device)
    else:
        (args.output_dir / "config.json").write_text(json.dumps(vars(cfg), indent=2))
        (args.output_dir / "training_settings.json").write_text(json.dumps(settings, indent=2))
        np.save(args.output_dir / "scalar.npy", np.stack([dataset.mean, dataset.std]))
    sde = VESDE(cfg, sigma_min=cfg.beta_min, sigma_max=cfg.beta_max, N=cfg.num_scales)
    model.train()
    with (args.output_dir / "loss.csv").open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "dsm", "fpe", "total"])
        if not args.resume:
            writer.writeheader()
        while state["iteration"] < args.total_steps:
            try:
                batch, indices, observed, latent = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch, indices, observed, latent = next(iterator)
            batch = batch.to(device=device, dtype=torch.float32)
            kwargs = dict(frame_indices=indices.to(device), obs_mask=observed.to(device=device, dtype=torch.float32),
                          latent_mask=latent.to(device=device, dtype=torch.float32))
            optimizer.zero_grad(set_to_none=True)
            dsm = loss_fn_video(model, sde, batch, **kwargs)
            fpe = fpe_regularizer_loss(model, sde, batch, alpha=args.fpe_alpha, beta=args.fpe_beta,
                m=args.fpe_m, h_s=args.h_s, h_d=args.h_d, **kwargs) if args.stage == "fpe" else dsm.new_zeros(())
            loss = dsm + fpe
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss is not finite")
            loss.backward()
            max_norm = clip if clip is not None else float("inf")
            clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
            optimizer.step()
            ema.update(model.parameters())
            state["iteration"] += 1
            state["epoch"] = state["iteration"] // len(loader)
            row = dict(step=state["iteration"], dsm=float(dsm.detach()), fpe=float(fpe.detach()), total=float(loss.detach()))
            writer.writerow(row)
            handle.flush()
            state["loss_train"].append(row["total"])
            over_power = args.power_ceiling_w is not None and gpu_power(device) >= args.power_ceiling_w
            if over_power or state["iteration"] % args.checkpoint_every == 0 or state["iteration"] == args.total_steps:
                state["rng_state"] = rng_state(device)
                save_checkpoint(checkpoint_path, state)
                save_checkpoint(args.output_dir / f"checkpoint_step_{state['iteration']:06d}.pth", state)
                print(json.dumps(row), flush=True)
            if over_power:
                raise RuntimeError("GPU reached the power ceiling; checkpoint saved and training stopped")


if __name__ == "__main__":
    main()
