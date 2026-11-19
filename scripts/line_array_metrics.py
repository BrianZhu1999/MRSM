"""Metrics shared by the measured line-array examples."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_source_positions(true_positions: np.ndarray,
                           estimated_positions: np.ndarray):
    """Match source positions by the minimum sum of squared distances."""
    true_positions = np.asarray(true_positions, dtype=float)
    estimated_positions = np.asarray(estimated_positions, dtype=float)
    distances = np.linalg.norm(
        true_positions[:, np.newaxis, :] - estimated_positions[np.newaxis, :, :],
        axis=2,
    )
    true_indices, estimate_indices = linear_sum_assignment(distances ** 2)
    return true_indices, estimate_indices, distances[true_indices, estimate_indices]


def matched_source_errors(true_positions: np.ndarray,
                          estimated_positions: np.ndarray) -> np.ndarray:
    """Return matched source distances."""
    return match_source_positions(true_positions, estimated_positions)[2]
