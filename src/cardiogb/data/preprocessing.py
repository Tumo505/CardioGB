"""Expression normalization and stable state scaling."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def library_size_log1p(
    expression: np.ndarray | sparse.spmatrix,
    target_sum: float = 10_000.0,
) -> np.ndarray | sparse.spmatrix:
    """Normalize observations by library size and apply log1p."""
    if target_sum <= 0:
        raise ValueError("target_sum must be positive")
    totals = np.asarray(expression.sum(axis=1)).reshape(-1)
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    if sparse.issparse(expression):
        result = sparse.diags(scale) @ expression
        result = result.tocsr(copy=False)
        result.data = np.log1p(result.data)
        return result
    return np.log1p(np.asarray(expression, dtype=float) * scale[:, None])


def robust_minmax(values: np.ndarray, lower_q: float = 0.01, upper_q: float = 0.99) -> np.ndarray:
    """Clip each state to robust quantiles and scale it to [0, 1]."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be two-dimensional")
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    lower = np.quantile(values, lower_q, axis=0)
    upper = np.quantile(values, upper_q, axis=0)
    width = upper - lower
    width[width == 0] = 1.0
    return np.clip((values - lower) / width, 0.0, 1.0).astype(np.float32)
