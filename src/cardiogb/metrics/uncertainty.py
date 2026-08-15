"""Ensemble uncertainty and calibration summaries."""

from __future__ import annotations

import numpy as np


def ensemble_summary(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions)
    if predictions.ndim < 2:
        raise ValueError("predictions must include an ensemble dimension")
    return predictions.mean(axis=0), predictions.std(axis=0, ddof=0)


def interval_coverage(
    mean: np.ndarray,
    std: np.ndarray,
    observed: np.ndarray,
    z: float = 1.96,
) -> float:
    lower, upper = mean - z * std, mean + z * std
    return float(np.mean((observed >= lower) & (observed <= upper)))


def uncertainty_error_correlation(std: np.ndarray, absolute_error: np.ndarray) -> float:
    x, y = np.asarray(std).reshape(-1), np.asarray(absolute_error).reshape(-1)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def distribution_mean_calibration(
    ensemble_predictions: np.ndarray,
    observed: np.ndarray,
    z: float = 1.96,
) -> dict[str, object]:
    """Calibrate ensemble uncertainty on state means for unmatched populations."""
    predictions, observed = np.asarray(ensemble_predictions), np.asarray(observed)
    if predictions.ndim != 3 or observed.ndim != 2:
        raise ValueError("expected [members, nodes, states] and [nodes, states]")
    member_means = predictions.mean(axis=1)
    mean, std = member_means.mean(axis=0), member_means.std(axis=0)
    target = observed.mean(axis=0)
    covered = (target >= mean - z * std) & (target <= mean + z * std)
    return {
        "coverage": float(covered.mean()),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "observed_mean": target.tolist(),
        "covered": covered.tolist(),
    }
