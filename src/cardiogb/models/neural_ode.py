"""Non-spatial black-box neural ODE vector field."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NeuralODEFunc(nn.Module):
    def __init__(
        self,
        state_dim: int = 6,
        hidden_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
        time_dependent: bool = True,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.state_dim = state_dim
        self.time_dependent = time_dependent
        input_dim = state_dim + int(time_dependent)
        modules: list[nn.Module] = []
        for layer in range(layers):
            modules.extend(
                [nn.Linear(input_dim if layer == 0 else hidden_dim, hidden_dim), nn.SiLU()]
            )
            if dropout:
                modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(hidden_dim, state_dim))
        self.network = nn.Sequential(*modules)

    def forward(self, t: Tensor | float, states: Tensor, graph: object = None) -> Tensor:
        del graph
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}")
        features = states
        if self.time_dependent:
            time = torch.as_tensor(t, dtype=states.dtype, device=states.device)
            time_feature = torch.ones_like(states[..., :1]) * time
            features = torch.cat((states, time_feature), dim=-1)
        return self.network(features)

