"""Model-aware integration helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from cardiogb.ode.solvers import integrate_fixed_step


def integrate_model(
    model: nn.Module,
    states: Tensor,
    graph: Any,
    t0: float,
    t1: float,
    *,
    step_size: float,
    method: str = "rk4",
) -> Tensor:
    return integrate_fixed_step(
        lambda t, x: model(t, x, graph), states, t0, t1, step_size=step_size, method=method
    )


def integrate_trajectory(
    model: nn.Module,
    states: Tensor,
    graph: Any,
    times: Tensor,
    *,
    step_size: float,
    method: str = "rk4",
) -> Tensor:
    """Return states at configured times without assuming paired observations."""
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times must be a non-empty one-dimensional tensor")
    outputs = [states]
    current = states
    for start, end in zip(times[:-1], times[1:]):
        current = integrate_model(
            model,
            current,
            graph,
            float(start.item()),
            float(end.item()),
            step_size=step_size,
            method=method,
        )
        outputs.append(current)
    return torch.stack(outputs)
