"""Validation-selected robust aggregation for deep ensembles."""

from __future__ import annotations

import numpy as np

from cardiogb.training.ensemble_weighting import fit_simplex_weights, weighted_summary


AGGREGATION_METHODS = ("median", "trimmed_mean", "simplex", "equal_mean")


def aggregate_members(
    values: np.ndarray,
    method: str,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate an array whose first axis indexes ensemble members."""
    values = np.asarray(values, dtype=float)
    if values.ndim < 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite ensemble members are required")
    if method == "simplex":
        if weights is None:
            raise ValueError("simplex aggregation requires weights")
        return weighted_summary(values, weights)[0]
    if method == "equal_mean":
        return values.mean(axis=0)
    if method == "median":
        return np.median(values, axis=0)
    if method == "trimmed_mean":
        ordered = np.sort(values, axis=0)
        return ordered[1:-1].mean(axis=0) if len(ordered) >= 5 else np.median(ordered, axis=0)
    raise ValueError(f"unknown aggregation method: {method}")


def select_aggregation(
    member_state_means: np.ndarray,
    observed_state_means: np.ndarray,
    groups: np.ndarray,
) -> tuple[str, dict[str, float]]:
    """Select aggregation by leave-one-validation-group-out mean squared error."""
    design = np.asarray(member_state_means, dtype=float)
    observed = np.asarray(observed_state_means, dtype=float)
    groups = np.asarray(groups)
    if design.ndim != 2 or observed.shape != (len(design),) or groups.shape != (len(design),):
        raise ValueError("design, observations, and validation groups have incompatible shapes")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two validation groups are required for aggregation selection")
    predictions = {method: np.empty_like(observed) for method in AGGREGATION_METHODS}
    for group in unique:
        held_out = groups == group
        fitted = fit_simplex_weights(design[~held_out], observed[~held_out])
        predictions["simplex"][held_out] = design[held_out] @ fitted
        held_out_members = design[held_out].T
        for method in ("median", "trimmed_mean", "equal_mean"):
            predictions[method][held_out] = aggregate_members(held_out_members, method)
    scores = {
        method: float(np.mean(np.square(predictions[method] - observed)))
        for method in AGGREGATION_METHODS
    }
    selected = min(AGGREGATION_METHODS, key=lambda method: scores[method])
    return selected, scores
