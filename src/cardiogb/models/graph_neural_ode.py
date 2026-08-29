"""Pure black-box graph neural ODE vector field."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cardiogb.models.message_passing import EdgeMessagePassing


class GraphNeuralODEFunc(nn.Module):
    def __init__(
        self,
        state_dim: int = 6,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
        time_dependent: bool = True,
        edge_gating: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.time_dependent = time_dependent
        message_state_dim = state_dim + int(time_dependent)
        self.message_passing = EdgeMessagePassing(
            state_dim=message_state_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            edge_gating=edge_gating,
        )
        self.projection = nn.Linear(message_state_dim, state_dim)

    def forward(self, t: Tensor | float, states: Tensor, graph: object) -> Tensor:
        features = states
        if self.time_dependent:
            time = torch.as_tensor(t, dtype=states.dtype, device=states.device)
            features = torch.cat((states, torch.ones_like(states[:, :1]) * time), dim=-1)
        return self.projection(self.message_passing(features, graph))

