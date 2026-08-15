"""Parameter and state constraint functions."""

from __future__ import annotations

import torch
from torch import Tensor


def positive(raw: Tensor, transform: str = "softplus") -> Tensor:
    if transform == "softplus":
        return torch.nn.functional.softplus(raw)
    if transform == "exp":
        return torch.exp(raw)
    raise ValueError(f"Unsupported positive-parameter transform: {transform}")


def bounds_penalty(states: Tensor, lower: float = 0.0, upper: float = 1.0) -> Tensor:
    """Mean soft violation of a closed state interval."""
    if lower >= upper:
        raise ValueError("lower must be less than upper")
    return (torch.relu(lower - states) + torch.relu(states - upper)).mean()


def inverse_softplus(value: float) -> float:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if value <= 0:
        raise ValueError("softplus initial values must be positive")
    return float(torch.log(torch.expm1(tensor)))

