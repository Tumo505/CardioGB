"""Differentiable losses for unmatched cross-sectional state distributions."""

from __future__ import annotations

import torch
from torch import Tensor


def rbf_mmd(
    predicted: Tensor,
    observed: Tensor,
    bandwidths: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0),
) -> Tensor:
    """Biased multi-scale RBF MMD supporting unequal sample counts."""
    _validate_samples(predicted, observed)
    if not bandwidths or any(value <= 0 for value in bandwidths):
        raise ValueError("bandwidths must contain positive values")
    xx = torch.cdist(predicted, predicted).square()
    yy = torch.cdist(observed, observed).square()
    xy = torch.cdist(predicted, observed).square()
    loss = predicted.new_zeros(())
    for bandwidth in bandwidths:
        denominator = 2 * bandwidth**2
        loss = loss + torch.exp(-xx / denominator).mean()
        loss = loss + torch.exp(-yy / denominator).mean()
        loss = loss - 2 * torch.exp(-xy / denominator).mean()
    return loss / len(bandwidths)


def moment_matching(predicted: Tensor, observed: Tensor) -> Tensor:
    """Match mean and covariance of two unmatched samples."""
    _validate_samples(predicted, observed)
    mean_loss = (predicted.mean(0) - observed.mean(0)).square().mean()
    pred_centered = predicted - predicted.mean(0)
    obs_centered = observed - observed.mean(0)
    pred_cov = pred_centered.T @ pred_centered / max(len(predicted) - 1, 1)
    obs_cov = obs_centered.T @ obs_centered / max(len(observed) - 1, 1)
    return mean_loss + (pred_cov - obs_cov).square().mean()


def sliced_wasserstein(
    predicted: Tensor,
    observed: Tensor,
    *,
    num_projections: int = 64,
    num_quantiles: int = 128,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Approximate W2 using random one-dimensional projections and quantiles."""
    _validate_samples(predicted, observed)
    directions = torch.randn(
        predicted.shape[1], num_projections,
        dtype=predicted.dtype, device=predicted.device, generator=generator,
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    pred_projection = predicted @ directions
    obs_projection = observed @ directions
    q = torch.linspace(0, 1, num_quantiles, dtype=predicted.dtype, device=predicted.device)
    return (
        torch.quantile(pred_projection, q, dim=0) - torch.quantile(obs_projection, q, dim=0)
    ).square().mean()


def _validate_samples(predicted: Tensor, observed: Tensor) -> None:
    if predicted.ndim != 2 or observed.ndim != 2:
        raise ValueError("distribution samples must be [observations, features]")
    if predicted.shape[1] != observed.shape[1]:
        raise ValueError("predicted and observed feature dimensions differ")
    if len(predicted) == 0 or len(observed) == 0:
        raise ValueError("distribution samples must be non-empty")
