"""Small deterministic fixed-step solvers for debugging and experiments."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


VectorField = Callable[[Tensor, Tensor], Tensor]
StateProjector = Callable[[Tensor], Tensor]


def _step_euler(field: VectorField, t: Tensor, x: Tensor, dt: Tensor) -> Tensor:
    return x + dt * field(t, x)


def _step_rk4(field: VectorField, t: Tensor, x: Tensor, dt: Tensor) -> Tensor:
    half = dt / 2
    k1 = field(t, x)
    k2 = field(t + half, x + half * k1)
    k3 = field(t + half, x + half * k2)
    k4 = field(t + dt, x + dt * k3)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_fixed_step(
    field: VectorField,
    x0: Tensor,
    t0: float,
    t1: float,
    *,
    step_size: float,
    method: str = "rk4",
    projector: StateProjector | None = None,
    checkpoint_steps: bool | int = False,
) -> Tensor:
    """Integrate a vector field while preserving gradients and device placement."""
    if t1 < t0:
        raise ValueError("t1 must not precede t0")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be 'euler' or 'rk4'")
    duration = t1 - t0
    if duration == 0:
        return x0
    steps = max(1, math.ceil(duration / step_size))
    dt = torch.as_tensor(duration / steps, dtype=x0.dtype, device=x0.device)
    t = torch.as_tensor(t0, dtype=x0.dtype, device=x0.device)
    x = x0
    step = _step_rk4 if method == "rk4" else _step_euler
    def advance(current: Tensor, current_time: Tensor) -> Tensor:
        updated = step(field, current_time, current, dt)
        return projector(updated) if projector is not None else updated

    checkpoint_interval = (
        1 if checkpoint_steps is True else int(checkpoint_steps or 0)
    )
    for index in range(steps):
        should_checkpoint = (
            checkpoint_interval > 0
            and (index + 1) % checkpoint_interval == 0
            and torch.is_grad_enabled()
        )
        if should_checkpoint:
            x = checkpoint(advance, x, t, use_reentrant=False)
        else:
            x = advance(x, t)
        t = t + dt
    return x

