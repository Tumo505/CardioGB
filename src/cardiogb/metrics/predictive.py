"""Point-summary predictive metrics."""

from __future__ import annotations

import numpy as np


def rmse(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(observed)) ** 2)))


def mae(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(observed))))


def statewise_rmse(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    return np.sqrt(np.mean((predicted - observed) ** 2, axis=0))
