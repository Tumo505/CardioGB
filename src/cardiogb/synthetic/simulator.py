"""Synthetic population dynamics for identifiability and omission tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from cardiogb.ode.integration import integrate_trajectory
from cardiogb.synthetic.hidden_mechanism import inflammation_remodelling_synergy
from cardiogb.synthetic.noise import add_gaussian_noise


@dataclass(frozen=True)
class SyntheticResult:
    times: Tensor
    latent_clean: Tensor
    observations: Tensor
    initial_states: Tensor
    hidden_mechanism_values: Tensor
    true_parameters: dict[str, float]


class SyntheticVectorField(nn.Module):
    def __init__(self, base: nn.Module, hidden_strength: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.hidden_strength = hidden_strength

    def forward(self, t: Tensor | float, states: Tensor, graph: object = None) -> Tensor:
        del graph
        return self.base(t, states) + inflammation_remodelling_synergy(
            states, self.hidden_strength
        )


def simulate_system(
    model: nn.Module,
    *,
    observation_times: list[float],
    num_entities: int = 256,
    noise_std: float = 0.02,
    hidden_mechanism: bool = False,
    hidden_strength: float = 0.4,
    seed: int = 0,
    step_size: float = 0.02,
) -> SyntheticResult:
    if sorted(observation_times) != observation_times or len(observation_times) < 2:
        raise ValueError("observation_times must contain at least two sorted values")
    generator = torch.Generator().manual_seed(seed)
    initial = torch.rand(num_entities, 6, generator=generator) * 0.25
    field = SyntheticVectorField(model, hidden_strength if hidden_mechanism else 0.0)
    times = torch.tensor(observation_times, dtype=initial.dtype)
    with torch.no_grad():
        clean = integrate_trajectory(field, initial, None, times, step_size=step_size).clamp(0, 1)
        observed = add_gaussian_noise(clean, noise_std, generator=generator)
        hidden_values = torch.stack(
            [inflammation_remodelling_synergy(state, hidden_strength) for state in clean]
        ) if hidden_mechanism else torch.zeros_like(clean)
    clean = clean.detach()
    observed = observed.detach()
    hidden_values = hidden_values.detach()
    if hasattr(model, "constrained_parameters"):
        parameters = {
            name: float(value.detach()) for name, value in model.constrained_parameters().items()
        }
    else:
        parameters = {}
    return SyntheticResult(times, clean, observed, initial, hidden_values, parameters)
