"""Deliberately omitted nonlinear mechanisms for recovery experiments."""

from __future__ import annotations

from torch import Tensor


def inflammation_remodelling_synergy(states: Tensor, strength: float = 0.4) -> Tensor:
    """Hidden I×F contribution to cardiomyocyte-regeneration dynamics."""
    correction = states.new_zeros(states.shape)
    correction[..., 3] = strength * states[..., 0] * states[..., 2]
    return correction
