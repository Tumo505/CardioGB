"""Statistics that treat biological units, rather than spots, as replicates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def grouped_bootstrap(
    values: np.ndarray,
    groups: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap whole biological units and return a percentile interval."""
    values, groups = np.asarray(values), np.asarray(groups)
    if len(values) != len(groups) or len(values) == 0:
        raise ValueError("values and groups must have equal, non-zero length")
    units = np.unique(groups)
    if len(units) < 2:
        raise ValueError("at least two biological units are required")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    for iteration in range(n_resamples):
        sampled = rng.choice(units, size=len(units), replace=True)
        replicate = np.concatenate([values[groups == unit] for unit in sampled])
        estimates[iteration] = statistic(replicate)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(statistic(values)),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "n_biological_units": int(len(units)),
    }


def paired_group_permutation_test(
    first: np.ndarray,
    second: np.ndarray,
    *,
    n_permutations: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired sign-flip test for one metric per biological unit."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1 or len(first) < 2:
        raise ValueError("paired inputs must be one-dimensional and have length >= 2")
    difference = first - second
    observed = float(difference.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, len(difference)))
    null = (signs * difference).mean(axis=1)
    p_value = (np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (n_permutations + 1)
    return {"mean_difference": observed, "p_value": float(p_value)}

def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Control false-discovery rate while preserving the input order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a one-dimensional array in [0, 1]")
    if len(values) == 0:
        return values.copy()
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output