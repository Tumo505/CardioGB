"""Configurable interaction-based mechanistic cardiac-regeneration ODE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from cardiogb.models.constraints import inverse_softplus, positive


@dataclass(frozen=True)
class Interaction:
    source: int
    target: int
    parameter: str
    sign: float


@dataclass(frozen=True)
class Decay:
    state: int
    parameter: str
    relaxation_target: float | None


class MechanisticODE(nn.Module):
    """A modular six-state mechanistic vector field.

    Interactions and decay terms are driven entirely by configuration. Learned
    rate parameters are transformed to be non-negative, while inhibitory signs
    must be declared explicitly in the interaction topology.
    """

    def __init__(
        self,
        state_names: Sequence[str],
        interactions: Sequence[Interaction],
        decays: Sequence[Decay],
        parameter_initials: Mapping[str, float],
        *,
        parameter_transform: str = "softplus",
        activation_state: str = "A",
        injury_parameter: str = "alpha_A",
        injury_decay: float = 2.0,
    ) -> None:
        super().__init__()
        self.state_names = tuple(state_names)
        if len(set(self.state_names)) != len(self.state_names):
            raise ValueError("state names must be unique")
        if activation_state not in self.state_names:
            raise ValueError(f"activation state {activation_state!r} is absent")
        self.state_index = {name: index for index, name in enumerate(self.state_names)}
        self.interactions = tuple(interactions)
        self.decays = tuple(decays)
        self.parameter_transform = parameter_transform
        self.activation_index = self.state_index[activation_state]
        self.injury_parameter = injury_parameter
        self.injury_decay = float(injury_decay)

        raw = {}
        for name, initial in parameter_initials.items():
            if parameter_transform == "softplus":
                value = inverse_softplus(float(initial))
            elif parameter_transform == "exp":
                if initial <= 0:
                    raise ValueError("exponential initial values must be positive")
                value = float(torch.log(torch.tensor(float(initial))))
            else:
                raise ValueError(f"Unsupported parameter transform: {parameter_transform}")
            raw[name] = nn.Parameter(torch.tensor(value, dtype=torch.float32))
        required = {item.parameter for item in interactions} | {item.parameter for item in decays}
        required.add(injury_parameter)
        missing = required - set(raw)
        if missing:
            raise ValueError(f"Missing initial values for parameters: {sorted(missing)}")
        self.raw_parameters = nn.ParameterDict(raw)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MechanisticODE":
        states = tuple(config["states"])
        index = {name: i for i, name in enumerate(states)}
        initials: dict[str, float] = {}
        interactions = []
        for item in config.get("interactions", []):
            source, target = item["source"], item["target"]
            if source not in index or target not in index:
                raise ValueError(f"Unknown interaction state: {source}->{target}")
            interactions.append(
                Interaction(index[source], index[target], item["parameter"], float(item["sign"]))
            )
            initials[item["parameter"]] = float(item["initial"])
        decays = []
        for state, item in config.get("decay", {}).items():
            if state not in index:
                raise ValueError(f"Unknown decay state: {state}")
            target = item.get("relaxation_target")
            decays.append(
                Decay(index[state], item["parameter"], None if target is None else float(target))
            )
            initials[item["parameter"]] = float(item["initial"])
        injury = config.get("injury_input", {})
        injury_parameter = injury.get("parameter", "alpha_A")
        initials[injury_parameter] = float(injury.get("initial", 1.0))
        return cls(
            states,
            interactions,
            decays,
            initials,
            parameter_transform=config.get("parameter_transform", "softplus"),
            injury_parameter=injury_parameter,
            injury_decay=float(injury.get("decay", 2.0)),
        )

    def constrained_parameter(self, name: str) -> Tensor:
        return positive(self.raw_parameters[name], self.parameter_transform)

    def constrained_parameters(self) -> dict[str, Tensor]:
        return {name: self.constrained_parameter(name) for name in self.raw_parameters}

    def injury_input(self, t: Tensor | float, reference: Tensor) -> Tensor:
        time = torch.as_tensor(t, dtype=reference.dtype, device=reference.device)
        return torch.exp(-self.injury_decay * torch.clamp_min(time, 0.0))

    def forward(self, t: Tensor | float, states: Tensor, graph: object = None) -> Tensor:
        del graph
        if states.shape[-1] != len(self.state_names):
            raise ValueError(
                f"Expected final state dimension {len(self.state_names)}, got {states.shape[-1]}"
            )
        derivative = torch.zeros_like(states)
        for item in self.interactions:
            rate = self.constrained_parameter(item.parameter)
            contribution = item.sign * rate * states[..., item.source]
            derivative[..., item.target] = derivative[..., item.target] + contribution
        for item in self.decays:
            rate = self.constrained_parameter(item.parameter)
            state = states[..., item.state]
            if item.relaxation_target is None:
                contribution = -rate * state
            else:
                contribution = rate * (item.relaxation_target - state)
            derivative[..., item.state] = derivative[..., item.state] + contribution
        injury_rate = self.constrained_parameter(self.injury_parameter)
        derivative[..., self.activation_index] = (
            derivative[..., self.activation_index] + injury_rate * self.injury_input(t, states)
        )
        return derivative
