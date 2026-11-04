"""Checkpoint loading shared by training and sampling."""
from pathlib import Path
import os

import torch


def load_model_state(model, saved):
    """Load all parameters, accepting the historical DataParallel prefix."""
    if not isinstance(saved, dict) or not saved:
        raise ValueError("checkpoint model state must be a nonempty dictionary")
    prefixes = [key.startswith("module.") for key in saved]
    if any(prefixes) and not all(prefixes):
        raise ValueError("checkpoint mixes DataParallel and unprefixed model keys")
    if all(prefixes):
        saved = {key[7:]: value for key, value in saved.items()}
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    expected = target.state_dict()
    if saved.keys() != expected.keys():
        missing = sorted(expected.keys() - saved.keys())
        extra = sorted(saved.keys() - expected.keys())
        raise ValueError(f"checkpoint keys do not match: missing={missing[:10]}, extra={extra[:10]}")
    for key, value in saved.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"checkpoint parameter shape does not match: {key}")
    target.load_state_dict(saved, strict=True)


def load_ema_state(ema, saved, model):
    """Validate EMA order and shapes, then place shadows beside model parameters."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    shadows = saved.get("shadow_params", [])
    if len(shadows) != len(parameters):
        raise ValueError("checkpoint EMA parameter count does not match the model")
    if not 0 <= saved.get("decay", -1) <= 1:
        raise ValueError("checkpoint EMA decay must be between zero and one")
    for index, (shadow, parameter) in enumerate(zip(shadows, parameters)):
        if not isinstance(shadow, torch.Tensor) or shadow.shape != parameter.shape:
            raise ValueError(f"checkpoint EMA parameter shape does not match at {index}")
    saved = dict(saved)
    saved["shadow_params"] = [s.detach().to(device=p.device, dtype=p.dtype).clone()
                              for s, p in zip(shadows, parameters)]
    ema.load_state_dict(saved)


def restore_checkpoint(ckpt_dir, state, device):
    """Restore a required checkpoint; missing or incompatible weights are errors."""
    path = Path(ckpt_dir)
    if not path.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {path}")
    loaded = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(loaded, dict) or "model" not in loaded:
        raise ValueError(f"checkpoint must contain a model state: {path}")
    load_model_state(state["model"], loaded["model"])
    if state.get("ema") is not None:
        if "ema" not in loaded:
            raise ValueError(f"checkpoint has no EMA parameters: {path}")
        load_ema_state(state["ema"], loaded["ema"], state["model"])
    if state.get("optimizer") is not None:
        if "optimizer" not in loaded:
            raise ValueError(f"checkpoint has no optimizer state: {path}")
        state["optimizer"].load_state_dict(loaded["optimizer"])
    for key in ("epoch", "iteration", "loss_train", "loss_val", "rng_state", "training_settings"):
        if key in loaded:
            state[key] = loaded[key]
    return state


def save_checkpoint(ckpt_dir, state):
    """Write a checkpoint atomically so an interrupted save preserves the last one."""
    path = Path(ckpt_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = dict(state)
    for key in ("model", "optimizer", "ema"):
        if saved.get(key) is not None:
            saved[key] = saved[key].state_dict()
    temporary = path.with_name(path.name + ".tmp")
    torch.save(saved, temporary)
    os.replace(temporary, path)
