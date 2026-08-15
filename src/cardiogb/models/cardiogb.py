"""Grey-box composition exposing mechanistic and residual vector fields."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from cardiogb.ode.solvers import integrate_fixed_step


class CardioGB(nn.Module):
    def __init__(self, mechanistic_model: nn.Module, residual_model: nn.Module) -> None:
        super().__init__()
        self.mechanistic_model = mechanistic_model
        self.residual_model = residual_model

    def vector_field(self, t: Tensor | float, states: Tensor, graph: Any = None) -> dict[str, Tensor]:
        mechanistic = self.mechanistic_model(t, states)
        residual = self.residual_model(t, states, graph)
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
            lambda t, x: self(t, x, graph), states, t0, t1, step_size=step_size, method=method
        )
