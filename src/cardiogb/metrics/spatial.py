"""Spatial autocorrelation metrics."""

from __future__ import annotations

import numpy as np


def morans_i(values: np.ndarray, edge_index: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    source, target = np.asarray(edge_index)
    centered = values - values.mean()
    denominator = np.square(centered).sum()
    if denominator == 0 or len(source) == 0:
        return 0.0
    numerator = np.sum(centered[source] * centered[target])
    return float(len(values) / len(source) * numerator / denominator)
