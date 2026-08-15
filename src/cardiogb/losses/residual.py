"""Regularization that prefers mechanistic explanations when sufficient."""

from __future__ import annotations

from torch import Tensor


def residual_penalty(residual: Tensor, norm: str = "l2") -> Tensor:
    if norm == "l2":
        return residual.square().mean()
    if norm == "l1":
        return residual.abs().mean()
    raise ValueError("norm must be 'l1' or 'l2'")
