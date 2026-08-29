"""Model-aware integration helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from cardiogb.ode.solvers import integrate_fixed_step


def _model_field(
    model: nn.Module, t: Tensor | float, states: Tensor, graph: Any, horizon: float
) -> Tensor:
    if hasattr(model, "persistence_gate"):
        return model(t, states, graph, forecast_horizon=horizon)
    return model(t, states, graph)


def _model_components(
    model: nn.Module, t: Tensor | float, states: Tensor, graph: Any, horizon: float
) -> dict[str, Tensor]:
    return model.vector_field(t, states, graph, forecast_horizon=horizon)


def integrate_model(
    model: nn.Module,
    states: Tensor,
    graph: Any,
    t0: float,
    t1: float,
    *,
    step_size: float,
    method: str = "rk4",
    checkpoint_steps: bool | int = False,
) -> Tensor:
    horizon = abs(t1 - t0)
    return integrate_fixed_step(
        lambda t, x: _model_field(model, t, x, graph, horizon),
        states,
        t0,
        t1,
        step_size=step_size,
        method=method,
        projector=getattr(model, "project_state", None),
        checkpoint_steps=checkpoint_steps,
    )


def integrate_model_with_residual_energy(
    model: nn.Module,
    states: Tensor,
    graph: Any,
    t0: float,
    t1: float,
    *,
    step_size: float,
    method: str = "rk4",
    checkpoint_steps: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Integrate states and the full-trajectory residual energy together."""
    if not hasattr(model, "vector_field"):
        return integrate_fixed_step(
            lambda t, x: model(t, x, graph),
            states,
            t0,
            t1,
            step_size=step_size,
            method=method,
            projector=getattr(model, "project_state", None),
            checkpoint_steps=checkpoint_steps,
        ), None

    state_width = states.shape[-1]
    horizon = abs(t1 - t0)
    augmented = torch.cat((states, states.new_zeros((len(states), 1))), dim=-1)

    def field(t: Tensor, value: Tensor) -> Tensor:
        x = value[:, :state_width]
        components = _model_components(model, t, x, graph, horizon)
        residual = components.get("residual")
        node_energy = (
            residual.square().mean(dim=-1, keepdim=True)
            if residual is not None
            else x.new_zeros((len(x), 1))
        )
        return torch.cat((components["total"], node_energy), dim=-1)

    state_projector = getattr(model, "project_state", None)

    def projector(value: Tensor) -> Tensor:
        if state_projector is None:
            return value
        return torch.cat((state_projector(value[:, :state_width]), value[:, state_width:]), dim=-1)

    result = integrate_fixed_step(
        field,
        augmented,
        t0,
        t1,
        step_size=step_size,
        method=method,
        projector=projector,
        checkpoint_steps=checkpoint_steps,
    )
    duration = max(abs(t1 - t0), torch.finfo(states.dtype).eps)
    return result[:, :state_width], result[:, state_width].mean() / duration


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
