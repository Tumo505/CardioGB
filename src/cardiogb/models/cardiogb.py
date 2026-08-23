"""Grey-box composition exposing mechanistic and residual vector fields."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

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
        self.residual_enabled = True

    def residual_scale(self) -> Tensor:
        return self.residual_scale_max * torch.sigmoid(self.raw_residual_scale)

    def mechanistic_gate(self) -> Tensor:
        return self.mechanistic_gate_min + (1.0 - self.mechanistic_gate_min) * torch.sigmoid(
            self.raw_mechanistic_gate
        )

    def project_state(self, states: Tensor) -> Tensor:
        """Project numerical solver steps into the registered biological range."""
        return states.clamp(min=self.state_min, max=self.state_max)

    def vector_field(self, t: Tensor | float, states: Tensor, graph: Any = None) -> dict[str, Tensor]:
        mechanistic_raw = self.mechanistic_model(t, states)
        mechanistic = self.mechanistic_gate().to(dtype=states.dtype) * mechanistic_raw
        if self.residual_enabled:
            residual_raw = self.residual_model(t, states, graph)
            residual = self.residual_scale().to(dtype=states.dtype) * torch.tanh(residual_raw)
        else:
            residual = torch.zeros_like(mechanistic)
        if mechanistic.shape != residual.shape:
            raise ValueError("mechanistic and residual vector fields must have identical shapes")
        return {
            "total": mechanistic + residual,
            "mechanistic": mechanistic,
            "residual": residual,
        }

    def forward(self, t: Tensor | float, states: Tensor, graph: Any = None) -> Tensor:
        return self.vector_field(t, states, graph)["total"]

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
        return integrate_fixed_step(
            lambda t, x: self(t, x, graph),
            states,
            t0,
            t1,
            step_size=step_size,
            method=method,
            projector=self.project_state,
        )
