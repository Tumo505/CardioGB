"""Leakage-safe experiment masks and transition evaluation protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from cardiogb.data.state_dataset import StateDataset
from cardiogb.interpretation.mechanistic_insufficiency import mechanistic_insufficiency
from cardiogb.metrics.distributional import distribution_metrics
from cardiogb.ode.integration import integrate_model
from cardiogb.training.trainer import CrossSectionalTransition


def group_holdout_masks(
    dataset: StateDataset,
    *,
    validation_groups: Sequence[str],
    test_groups: Sequence[str],
) -> dict[str, np.ndarray]:
    """Create mutually exclusive masks from biological-unit identifiers."""
    groups = dataset.groups.astype(str)
    validation = np.isin(groups, np.asarray(validation_groups, dtype=str))
    test = np.isin(groups, np.asarray(test_groups, dtype=str))
    if np.any(validation & test):
        raise ValueError("validation and test biological units overlap")
    train = ~(validation | test)
    if not train.any() or not validation.any() or not test.any():
        raise ValueError("each split must contain observations")
    return {"train": train, "validation": validation, "test": test}


def interpolation_masks(
    dataset: StateDataset, held_out_time: float, validation_time: float | None = None
) -> dict[str, np.ndarray]:
    """Hold out a non-terminal test stage and a distinct validation stage."""
    times = np.unique(dataset.times)
    if held_out_time not in times or held_out_time in (times.min(), times.max()):
        raise ValueError("interpolation stage must be observed and non-terminal")
    candidates = times[(times != held_out_time) & (times != times.min()) & (times != times.max())]
    if validation_time is None:
        if not len(candidates):
            raise ValueError("a distinct intermediate validation stage is required")
        validation_time = float(candidates[np.argmin(np.abs(candidates - held_out_time))])
    if validation_time == held_out_time or validation_time not in times:
        raise ValueError("validation stage must be observed and distinct from test")
    return {
        "train": (dataset.times != held_out_time) & (dataset.times != validation_time),
        "validation": dataset.times == validation_time,
        "test": dataset.times == held_out_time,
    }


def extrapolation_masks(dataset: StateDataset, cutoff_time: float) -> dict[str, np.ndarray]:
    """Separate future stages without using them for fitting."""
    train = dataset.times < cutoff_time
    validation = dataset.times == cutoff_time
    test = dataset.times > cutoff_time
    if not train.any() or not validation.any() or not test.any():
        raise ValueError("cutoff must leave earlier, cutoff, and future observations")
    return {"train": train, "validation": validation, "test": test}


@torch.no_grad()
def evaluate_transitions(
    model: nn.Module,
    transitions: Sequence[CrossSectionalTransition],
    *,
    device: str,
    step_size: float = 0.05,
    solver: str = "rk4",
    max_steps_per_transition: int | None = 16,
) -> list[dict[str, Any]]:
    """Evaluate unmatched distributions, aggregating bounded source patches."""
    model = model.to(device).eval()
    grouped: dict[str, dict[str, Any]] = {}
    for transition in transitions:
        graph = transition.graph.to(device) if hasattr(transition.graph, "to") else transition.graph
        source = transition.source_states.to(device)
        effective_step_size = step_size
        if max_steps_per_transition is not None:
            effective_step_size = max(
                step_size, abs(transition.t1 - transition.t0) / max_steps_per_transition
            )
        prediction = integrate_model(
            model,
            source,
            graph,
            transition.t0,
            transition.t1,
            step_size=effective_step_size,
            method=solver,
        )
        key = transition.evaluation_group or transition.name
        entry = grouped.setdefault(
            key,
            {
                "t0": transition.t0,
                "t1": transition.t1,
                "predictions": [],
                "target": transition.target_states.numpy(),
                "mi": [],
            },
        )
        entry["predictions"].append(prediction.cpu().numpy())
        if hasattr(model, "vector_field"):
            terms = model.vector_field(transition.t1, prediction, graph)
            if "mechanistic" in terms and "residual" in terms:
                mi = mechanistic_insufficiency(terms["mechanistic"], terms["residual"])
                entry["mi"].append(mi.cpu().numpy())
    rows = []
    for name, entry in grouped.items():
        prediction = np.concatenate(entry["predictions"], axis=0)
        row: dict[str, Any] = {
            "transition": name,
            "t0": entry["t0"],
            "t1": entry["t1"],
            **distribution_metrics(prediction, entry["target"]),
        }
        if entry["mi"]:
            mi = np.concatenate(entry["mi"])
            row.update(mi_mean=float(mi.mean()), mi_median=float(np.median(mi)))
        rows.append(row)
    return rows
