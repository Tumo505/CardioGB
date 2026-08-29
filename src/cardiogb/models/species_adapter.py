"""Species-specific observation maps around conserved CardioGB dynamics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from cardiogb.models.constraints import inverse_softplus
from cardiogb.ode.solvers import integrate_fixed_step


class SpeciesAdapter(nn.Module):
    """Near-identity species encoders/decoders with positive biological-time scales."""

    def __init__(self, state_dim: int, species: Sequence[str]) -> None:
        super().__init__()
        names = tuple(dict.fromkeys(map(str, species)))
        if state_dim < 1 or len(names) < 2:
            raise ValueError("state_dim must be positive and at least two species are required")
        self.state_dim = int(state_dim)
        self.species = names
        self.encoders = nn.ModuleDict({name: nn.Linear(state_dim, state_dim, bias=False) for name in names})
        self.decoders = nn.ModuleDict({name: nn.Linear(state_dim, state_dim, bias=False) for name in names})
        initial = torch.tensor(inverse_softplus(1.0), dtype=torch.float32)
        self.raw_time_scales = nn.ParameterDict(
            {name: nn.Parameter(initial.clone()) for name in names}
        )
        identity = torch.eye(state_dim)
        with torch.no_grad():
            for name in names:
                self.encoders[name].weight.copy_(identity)
                self.decoders[name].weight.copy_(identity)

    def _check(self, species: str) -> str:
        name = str(species)
        if name not in self.encoders:
            raise KeyError(f"unknown species: {name}")
        return name

    def encode(self, states: Tensor, species: str) -> Tensor:
        return self.encoders[self._check(species)](states)

    def decode(self, latent: Tensor, species: str) -> Tensor:
        return self.decoders[self._check(species)](latent)

    def time_scale(self, species: str) -> Tensor:
        return torch.nn.functional.softplus(self.raw_time_scales[self._check(species)])

    def regularization(self) -> Tensor:
        """Penalize departure from invertible near-identity observation maps."""
        identity = torch.eye(
            self.state_dim,
            dtype=next(self.parameters()).dtype,
            device=next(self.parameters()).device,
        )
        penalties = []
        for name in self.species:
            encoder = self.encoders[name].weight
            decoder = self.decoders[name].weight
            penalties.extend(((encoder - identity).square().mean(), (decoder @ encoder - identity).square().mean()))
        return torch.stack(penalties).mean()


class SpeciesAdaptedForecaster(nn.Module):
    """Wrap shared conserved dynamics with species-specific measurement and time maps."""

    def __init__(self, shared_model: nn.Module, adapter: SpeciesAdapter) -> None:
        super().__init__()
        self.shared_model = shared_model
        self.adapter = adapter

    def freeze_shared_dynamics(self, frozen: bool = True) -> None:
        for parameter in self.shared_model.parameters():
            parameter.requires_grad_(not frozen)

    def forecast(
        self,
        states: Tensor,
        graph: Any,
        t0: float,
        t1: float,
        *,
        species: str,
        step_size: float,
        method: str = "rk4",
        checkpoint_steps: bool | int = False,
    ) -> Tensor:
        if t1 < t0:
            raise ValueError("t1 must not precede t0")
        scale = self.adapter.time_scale(species)
        duration = float(t1 - t0)
        latent = self.adapter.encode(states, species)

        def field(unit_time: Tensor, value: Tensor) -> Tensor:
            biological_time = (t0 + unit_time * duration) * scale
            horizon = duration * scale
            if hasattr(self.shared_model, "persistence_gate"):
                velocity = self.shared_model(
                    biological_time, value, graph, forecast_horizon=horizon
                )
            else:
                velocity = self.shared_model(biological_time, value, graph)
            return duration * scale * velocity

        normalized_step = min(1.0, step_size / max(duration, step_size))
        predicted = integrate_fixed_step(
            field,
            latent,
            0.0,
            1.0,
            step_size=normalized_step,
            method=method,
            projector=getattr(self.shared_model, "project_state", None),
            checkpoint_steps=checkpoint_steps,
        )
        return self.adapter.decode(predicted, species)
