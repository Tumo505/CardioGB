"""Differentiable losses for unmatched cross-sectional state distributions."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def rbf_mmd(
    predicted: Tensor,
    observed: Tensor,
    bandwidths: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0),
    *,
    chunk_size: int = 256,
) -> Tensor:
    """Biased multi-scale RBF MMD with exact checkpointed kernel chunks."""
    _validate_samples(predicted, observed)
    if not bandwidths or any(value <= 0 for value in bandwidths):
        raise ValueError("bandwidths must contain positive values")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    denominators = predicted.new_tensor([2 * value**2 for value in bandwidths])

    def kernel_means(left: Tensor, right: Tensor) -> Tensor:
        total = left.new_zeros(len(bandwidths))
        for start_index in range(0, len(left), chunk_size):
            left_chunk = left[start_index : start_index + chunk_size]

            def kernel_sums(chunk: Tensor, reference: Tensor) -> Tensor:
                distances = torch.cdist(chunk, reference).square()
                return torch.stack(
                    [torch.exp(-distances / denominator).sum() for denominator in denominators]
                )

            if torch.is_grad_enabled() and (left_chunk.requires_grad or right.requires_grad):
                chunk_sums = checkpoint(
                    kernel_sums, left_chunk, right, use_reentrant=False
                )
            else:
                chunk_sums = kernel_sums(left_chunk, right)
            total = total + chunk_sums
        return total / (len(left) * len(right))

    xx = kernel_means(predicted, predicted)
    yy = kernel_means(observed, observed)
    xy = kernel_means(predicted, observed)
    return (xx + yy - 2 * xy).mean()

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
    # torch.quantile does not accept float16; promotion preserves gradients.
    if predicted.dtype not in (torch.float32, torch.float64):
        predicted = predicted.float()
    if observed.dtype != predicted.dtype:
        observed = observed.to(dtype=predicted.dtype)
    directions = torch.randn(
        predicted.shape[1], num_projections,
        dtype=predicted.dtype, device=predicted.device, generator=generator,
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    pred_projection = (predicted @ directions).float()
    obs_projection = (observed @ directions).float()
    q = torch.linspace(0, 1, num_quantiles, dtype=torch.float32, device=predicted.device)
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
