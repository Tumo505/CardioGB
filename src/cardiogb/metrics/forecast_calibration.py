"""Validation-only calibration for forecasts beyond the observed horizon."""

from __future__ import annotations

import numpy as np


def fit_displacement_scale(
    source_means: np.ndarray,
    predicted_means: np.ndarray,
    observed_means: np.ndarray,
) -> float:
    """Fit a scalar prediction-displacement multiplier on validation means.

    The constrained least-squares coefficient lies in [0, 1], so calibration
    can only retain or shrink the modelled displacement from persistence.
    """
    source = np.asarray(source_means, dtype=float)
    predicted = np.asarray(predicted_means, dtype=float)
    observed = np.asarray(observed_means, dtype=float)
    if source.shape != predicted.shape or source.shape != observed.shape:
        raise ValueError("source, predicted, and observed means must have matching shapes")
    if source.ndim != 2 or not np.isfinite([source, predicted, observed]).all():
        raise ValueError("validation means must be finite two-dimensional arrays")
    displacement = predicted - source
    target = observed - source
    denominator = float(np.square(displacement).sum())
    if denominator <= np.finfo(float).eps:
        return 0.0
    coefficient = float((displacement * target).sum() / denominator)
    return float(np.clip(coefficient, 0.0, 1.0))


def horizon_displacement_scale(
    horizon: float,
    maximum_validated_horizon: float,
    validation_scale: float,
) -> float:
    """Shrink displacement once the forecast exceeds validation support."""
    if horizon <= 0 or maximum_validated_horizon <= 0:
        raise ValueError("forecast and validation horizons must be positive")
    if not 0 <= validation_scale <= 1:
        raise ValueError("validation_scale must lie in [0, 1]")
    return float(validation_scale * min(1.0, maximum_validated_horizon / horizon))


def calibrate_forecast(
    source: np.ndarray,
    predicted: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Return a bounded forecast between persistence and the raw prediction."""
    source = np.asarray(source, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if source.shape != predicted.shape:
        raise ValueError("source and predicted states must have matching shapes")
    if not 0 <= scale <= 1:
        raise ValueError("scale must lie in [0, 1]")
    return np.clip(source + scale * (predicted - source), 0.0, 1.0)
