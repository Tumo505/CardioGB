"""Validation-only simplex weighting for deep ensembles."""

from __future__ import annotations

import numpy as np


def fit_simplex_weights(
    member_state_means: np.ndarray,
    observed_state_means: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Fit nonnegative, sum-to-one member weights on validation state means."""
    design = np.asarray(member_state_means, dtype=float)
    observed = np.asarray(observed_state_means, dtype=float)
    if design.ndim != 2 or observed.shape != (len(design),):
        raise ValueError("member_state_means must be [samples, members] and observed [samples]")
    if design.shape[1] < 2 or not np.isfinite(design).all() or not np.isfinite(observed).all():
        raise ValueError("at least two finite ensemble members are required")
    members = design.shape[1]
    uniform = np.repeat(1.0 / members, members)

    def project_simplex(values: np.ndarray) -> np.ndarray:
        ordered = np.sort(values)[::-1]
        cumulative = np.cumsum(ordered) - 1.0
        eligible = ordered - cumulative / np.arange(1, len(values) + 1) > 0
        rho = np.flatnonzero(eligible)[-1]
        threshold = cumulative[rho] / (rho + 1)
        return np.maximum(values - threshold, 0.0)

    spectral = np.linalg.norm(design, ord=2)
    lipschitz = 2.0 * spectral**2 / len(design) + 2.0 * ridge
    step = 1.0 / max(lipschitz, np.finfo(float).eps)
    weights = uniform.copy()
    for _ in range(5000):
        gradient = (
            2.0 * design.T @ (design @ weights - observed) / len(design)
            + 2.0 * ridge * (weights - uniform)
        )
        updated = project_simplex(weights - step * gradient)
        if np.linalg.norm(updated - weights) <= 1e-12:
            weights = updated
            break
        weights = updated
    return weights


def weighted_summary(stacked: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted mean and standard deviation over ensemble axis zero."""
    values = np.asarray(stacked, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim < 2 or weights.shape != (values.shape[0],):
        raise ValueError("weights must match ensemble axis zero")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    mean = np.tensordot(weights, values, axes=(0, 0))
    variance = np.tensordot(weights, (values - mean) ** 2, axes=(0, 0))
    return mean, np.sqrt(np.maximum(variance, 0.0))
