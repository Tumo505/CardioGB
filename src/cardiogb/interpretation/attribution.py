"""Gradient and perturbation attribution for residual hypotheses."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def input_gradient_attribution(
    residual_model: nn.Module,
    t: float,
    states: Tensor,
    graph: Any,
    *,
    target_state: int,
) -> Tensor:
    inputs = states.detach().clone().requires_grad_(True)
    output = residual_model(t, inputs, graph)[..., target_state].sum()
    gradient = torch.autograd.grad(output, inputs)[0]
    return gradient.detach()


def integrated_gradients(
    residual_model: nn.Module,
    t: float,
    states: Tensor,
    graph: Any,
    *,
    target_state: int,
    baseline: Tensor | None = None,
    steps: int = 32,
) -> Tensor:
    if steps < 2:
        raise ValueError("steps must be at least two")
    baseline = torch.zeros_like(states) if baseline is None else baseline
    total = torch.zeros_like(states)
    for alpha in torch.linspace(0, 1, steps, device=states.device, dtype=states.dtype):
        point = (baseline + alpha * (states - baseline)).detach().requires_grad_(True)
        output = residual_model(t, point, graph)[..., target_state].sum()
        total += torch.autograd.grad(output, point)[0]
    return ((states - baseline) * total / steps).detach()
