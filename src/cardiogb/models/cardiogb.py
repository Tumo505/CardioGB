"""Grey-box composition exposing mechanistic and residual vector fields."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from cardiogb.models.constraints import inverse_softplus
from cardiogb.ode.solvers import integrate_fixed_step


class CardioGB(nn.Module):
    def __init__(
        self,
        mechanistic_model: nn.Module,
        residual_model: nn.Module,
        *,
        state_min: float = 0.0,
        state_max: float = 1.0,
        residual_scale_max: float = 0.25,
        residual_scale_initial: float = 0.05,
        mechanistic_gate_min: float = 0.05,
        mechanistic_gate_initial: float = 0.5,
        learn_mechanistic_gate: bool = False,
        orthogonal_residual: bool = True,
        orthogonal_projection_strength: float = 1.0,
        velocity_limit: float | None = 0.5,
        persistence_gate: bool = True,
        persistence_gate_initial: float = 0.95,
        persistence_horizon_slope_initial: float = 0.05,
    ) -> None:
        super().__init__()
        if state_min >= state_max:
            raise ValueError("state_min must be less than state_max")
        if not 0 < residual_scale_initial < residual_scale_max:
            raise ValueError("residual_scale_initial must lie within (0, residual_scale_max)")
        self.mechanistic_model = mechanistic_model
        self.residual_model = residual_model
        self.state_min = float(state_min)
        self.state_max = float(state_max)
        self.residual_scale_max = float(residual_scale_max)
        ratio = residual_scale_initial / residual_scale_max
        raw_initial = torch.logit(torch.tensor(ratio, dtype=torch.float32))
        state_dim = len(getattr(mechanistic_model, "state_names", ()))
        if state_dim == 0:
            raise ValueError("mechanistic_model must expose non-empty state_names")
        self.raw_residual_scale = nn.Parameter(raw_initial.repeat(state_dim))
        if not 0 <= mechanistic_gate_min < mechanistic_gate_initial <= 1:
            raise ValueError("mechanistic gate must satisfy 0 <= min < initial <= 1")
        self.mechanistic_gate_min = float(mechanistic_gate_min)
        gate_ratio = (mechanistic_gate_initial - mechanistic_gate_min) / (1.0 - mechanistic_gate_min)
        self.raw_mechanistic_gate = nn.Parameter(
            torch.logit(torch.tensor(gate_ratio, dtype=torch.float32)).repeat(state_dim)
        )
        self.learn_mechanistic_gate = bool(learn_mechanistic_gate)
        if not self.learn_mechanistic_gate:
            self.raw_mechanistic_gate.requires_grad_(False)
        if not 0 <= orthogonal_projection_strength <= 1:
            raise ValueError("orthogonal_projection_strength must lie in [0, 1]")
        if velocity_limit is not None and velocity_limit <= 0:
            raise ValueError("velocity_limit must be positive")
        if not 0 < persistence_gate_initial < 1:
            raise ValueError("persistence_gate_initial must lie in (0, 1)")
        if persistence_horizon_slope_initial <= 0:
            raise ValueError("persistence_horizon_slope_initial must be positive")
        self.orthogonal_residual = bool(orthogonal_residual)
        self.orthogonal_projection_strength = float(orthogonal_projection_strength)
        self.velocity_limit = None if velocity_limit is None else float(velocity_limit)
        self.persistence_gate_enabled = bool(persistence_gate)
        self.raw_persistence_gate_bias = nn.Parameter(
            torch.logit(torch.tensor(persistence_gate_initial, dtype=torch.float32))
        )
        self.raw_persistence_horizon_slope = nn.Parameter(
            torch.tensor(inverse_softplus(persistence_horizon_slope_initial), dtype=torch.float32)
        )
        self.mechanistic_enabled = True
        self.residual_enabled = True

    def residual_scale(self) -> Tensor:
        return self.residual_scale_max * torch.sigmoid(self.raw_residual_scale)

    def mechanistic_gate(self) -> Tensor:
        if not self.learn_mechanistic_gate:
            return torch.ones_like(self.raw_mechanistic_gate)
        return self.mechanistic_gate_min + (1.0 - self.mechanistic_gate_min) * torch.sigmoid(
            self.raw_mechanistic_gate
        )

    def persistence_gate(
        self,
        forecast_horizon: Tensor | float | None,
        uncertainty: Tensor | float | None = None,
    ) -> Tensor:
        """Return learned confidence in departing from persistence."""
        if not self.persistence_gate_enabled:
            return torch.ones_like(self.raw_persistence_gate_bias)
        horizon = torch.as_tensor(
            0.0 if forecast_horizon is None else forecast_horizon,
            dtype=self.raw_persistence_gate_bias.dtype,
            device=self.raw_persistence_gate_bias.device,
        ).abs()
        slope = torch.nn.functional.softplus(self.raw_persistence_horizon_slope)
        logit = self.raw_persistence_gate_bias - slope * torch.log1p(horizon)
        if uncertainty is not None:
            value = torch.as_tensor(uncertainty, dtype=logit.dtype, device=logit.device)
            logit = logit - value.clamp_min(0.0)
        return torch.sigmoid(logit)

    def project_state(self, states: Tensor) -> Tensor:
        """Project numerical solver steps into the registered biological range."""
        return states.clamp(min=self.state_min, max=self.state_max)

    def vector_field(
        self,
        t: Tensor | float,
        states: Tensor,
        graph: Any = None,
        *,
        forecast_horizon: Tensor | float | None = None,
        uncertainty: Tensor | float | None = None,
    ) -> dict[str, Tensor]:
        if self.mechanistic_enabled:
            mechanistic_raw = self.mechanistic_model(t, states)
            mechanistic = self.mechanistic_gate().to(dtype=states.dtype) * mechanistic_raw
        else:
            mechanistic = torch.zeros_like(states)
        if self.residual_enabled:
            residual_raw = self.residual_model(t, states, graph)
            residual = self.residual_scale().to(dtype=states.dtype) * torch.tanh(residual_raw)
        else:
            residual = torch.zeros_like(mechanistic)
        if mechanistic.shape != residual.shape:
            raise ValueError("mechanistic and residual vector fields must have identical shapes")
        if self.orthogonal_residual and self.mechanistic_enabled and self.residual_enabled:
            denominator = mechanistic.square().sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(states.dtype).eps
            )
            parallel = (
                (residual * mechanistic).sum(dim=-1, keepdim=True) / denominator
            ) * mechanistic
            residual = residual - self.orthogonal_projection_strength * parallel
        confidence = self.persistence_gate(forecast_horizon, uncertainty).to(dtype=states.dtype)
        mechanistic = confidence * mechanistic
        residual = confidence * residual
        total = mechanistic + residual
        if self.velocity_limit is not None:
            magnitude = total.abs().amax(dim=-1, keepdim=True)
            ratio = magnitude / self.velocity_limit
            scale = torch.where(
                ratio > torch.finfo(total.dtype).eps,
                torch.tanh(ratio) / ratio,
                torch.ones_like(ratio),
            )
            mechanistic = mechanistic * scale
            residual = residual * scale
            total = mechanistic + residual
        return {
            "total": total,
            "mechanistic": mechanistic,
            "residual": residual,
        }

    def forward(
        self,
        t: Tensor | float,
        states: Tensor,
        graph: Any = None,
        *,
        forecast_horizon: Tensor | float | None = None,
        uncertainty: Tensor | float | None = None,
    ) -> Tensor:
        return self.vector_field(
            t, states, graph, forecast_horizon=forecast_horizon, uncertainty=uncertainty
        )["total"]

    def integrate(
        self,
        states: Tensor,
        graph: Any,
        t0: float,
        t1: float,
        *,
        step_size: float = 0.05,
        method: str = "rk4",
    ) -> Tensor:
        horizon = abs(t1 - t0)
        return integrate_fixed_step(
            lambda t, x: self(
                t,
                x,
                graph,
                forecast_horizon=horizon,
            ),
            states,
            t0,
            t1,
            step_size=step_size,
            method=method,
            projector=self.project_state,
        )
