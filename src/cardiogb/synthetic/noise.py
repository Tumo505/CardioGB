"""Observation noise models."""

from __future__ import annotations

import torch
from torch import Tensor


def add_gaussian_noise(
    values: Tensor,
    std: float,
    *,
    generator: torch.Generator | None = None,
    clamp: tuple[float, float] | None = (0.0, 1.0),
) -> Tensor:
    if std < 0:
        raise ValueError("noise standard deviation must be non-negative")
    noisy = values + torch.randn(values.shape, dtype=values.dtype, device=values.device, generator=generator) * std
    return noisy if clamp is None else noisy.clamp(*clamp)
