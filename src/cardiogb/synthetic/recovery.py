"""Optimization routines for synthetic system-identification experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cardiogb.metrics.identifiability import parameter_recovery
from cardiogb.ode.integration import integrate_trajectory


@dataclass(frozen=True)
class RecoveryResult:
    true_parameters: dict[str, float]
    inferred_parameters: dict[str, float]
    metrics: dict[str, float]
    loss_history: tuple[float, ...]


@dataclass(frozen=True)
class HiddenMechanismRecoveryResult:
    correlation: float
    rmse: float
    loss_history: tuple[float, ...]


def recover_hidden_mechanism(
    residual_model: torch.nn.Module,
    states: torch.Tensor,
    hidden_values: torch.Tensor,
    *,
    epochs: int = 200,
    learning_rate: float = 1e-3,
) -> HiddenMechanismRecoveryResult:
    """Synthetic oracle diagnostic: fit and compare a deliberately omitted term.

    This direct vector-field supervision is valid only because the simulator
    exposes ground truth. Real-data residuals are never treated as causal truth.
    """
    flat_states = states.reshape(-1, states.shape[-1]).detach()
    flat_hidden = hidden_values.reshape(-1, hidden_values.shape[-1]).detach()
    optimizer = torch.optim.Adam(residual_model.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        predicted = residual_model(0.0, flat_states, None)
        loss = (predicted - flat_hidden).square().mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        predicted = residual_model(0.0, flat_states, None)
        target = flat_hidden[:, 3]
        recovered = predicted[:, 3]
        centered_target = target - target.mean()
        centered_recovered = recovered - recovered.mean()
        denominator = centered_target.norm() * centered_recovered.norm()
        correlation = float((centered_target @ centered_recovered) / denominator.clamp_min(1e-12))
        rmse = float(torch.sqrt((predicted - flat_hidden).square().mean()))
    return HiddenMechanismRecoveryResult(correlation, rmse, tuple(history))


def recover_mechanistic_parameters(
    model: torch.nn.Module,
    times: torch.Tensor,
    observations: torch.Tensor,
    true_parameters: dict[str, float],
    *,
    epochs: int = 200,
    learning_rate: float = 0.03,
    step_size: float = 0.05,
) -> RecoveryResult:
    """Fit known synthetic trajectories; this paired loss is synthetic-only."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    initial = observations[0]
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        predicted = integrate_trajectory(model, initial, None, times, step_size=step_size)
        loss = (predicted - observations).square().mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    inferred = {
        name: float(value.detach()) for name, value in model.constrained_parameters().items()
    }
    return RecoveryResult(
        true_parameters,
        inferred,
        parameter_recovery(true_parameters, inferred),
        tuple(history),
    )
