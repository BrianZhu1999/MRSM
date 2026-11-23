"""Channel scaling and pressure observations for the standing-wave tube."""

from pathlib import Path

import numpy as np
import torch


class ChannelScaler1D:
    def __init__(self, mean, std, device):
        self.mean_np = np.asarray(mean).reshape(1, 1, -1, 1)
        self.std_np = np.asarray(std).reshape(1, 1, -1, 1)
        if (self.mean_np.shape != self.std_np.shape
                or not np.isfinite(self.mean_np).all()
                or not np.isfinite(self.std_np).all()
                or np.any(self.std_np <= 0)):
            raise ValueError("channel means and positive standard deviations must be finite")
        self.mean_torch = torch.as_tensor(self.mean_np, dtype=torch.float32, device=device)
        self.std_torch = torch.as_tensor(self.std_np, dtype=torch.float32, device=device)

    def normalize(self, array):
        return (array - self.mean_np) / self.std_np

    def unnormalize(self, array):
        if isinstance(array, torch.Tensor):
            return array * self.std_torch + self.mean_torch
        return array * self.std_np + self.mean_np


def tube_scaler(measured, device, scale_source="record", statistics_path=None):
    """Use full-record channels or saved training means and standard deviations."""
    if scale_source == "record":
        mean = measured.mean(axis=(0, 1, 3))
        std = measured.std(axis=(0, 1, 3))
    elif scale_source == "training":
        if statistics_path is None:
            raise ValueError("training scaling requires a statistics file")
        statistics = np.load(Path(statistics_path), allow_pickle=False)
        if statistics.size != 4 or statistics.shape[0] != 2:
            raise ValueError("training statistics must contain two channel means and standard deviations")
        mean, std = statistics.reshape(2, 2)
    else:
        raise ValueError(f"unknown tube scale source: {scale_source}")
    return ChannelScaler1D(mean, std, device)


def pressure_only_observation(window, scaler):
    observation = np.zeros(window.shape, dtype=np.float32)
    observation[:, :, 0, :] = (
        (window[:, :, 0, :] - scaler.mean_np[0, 0, 0, 0])
        / scaler.std_np[0, 0, 0, 0]
    )
    return observation
