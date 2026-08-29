"""Portable edge-aware message passing without mandatory PyG kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TorchGraph:
    edge_index: Tensor
    edge_attr: Tensor

    def to(self, device: torch.device | str) -> "TorchGraph":
        return TorchGraph(self.edge_index.to(device), self.edge_attr.to(device))

    def induced_subgraph(self, nodes: Tensor) -> "TorchGraph":
        """Return an index-remapped subgraph over the requested global nodes."""
        nodes = nodes.to(device=self.edge_index.device, dtype=torch.long)
        if len(nodes) == 0:
            raise ValueError("nodes must be non-empty")
        if self.edge_index.shape[1] == 0:
            return TorchGraph(
                torch.empty((2, 0), dtype=torch.long, device=self.edge_index.device),
                self.edge_attr[:0],
            )
        size = max(int(self.edge_index.max()), int(nodes.max())) + 1
        mapping = torch.full(
            (size,), -1, dtype=torch.long, device=self.edge_index.device
        )
        mapping[nodes] = torch.arange(len(nodes), device=self.edge_index.device)
        source, target = self.edge_index
        keep = (mapping[source] >= 0) & (mapping[target] >= 0)
        remapped = torch.stack((mapping[source[keep]], mapping[target[keep]]))
        return TorchGraph(remapped, self.edge_attr[keep])


def unpack_graph(graph: TorchGraph | Mapping[str, Tensor] | Any) -> tuple[Tensor, Tensor]:
    if isinstance(graph, Mapping):
        return graph["edge_index"], graph["edge_attr"]
    if hasattr(graph, "edge_index") and hasattr(graph, "edge_attr"):
        return graph.edge_index, graph.edge_attr
    raise TypeError("graph must expose edge_index and edge_attr")


class EdgeMessagePassing(nn.Module):
    """Mean-aggregated edge MLP followed by a node residual MLP."""

    def __init__(
        self,
        state_dim: int = 6,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.0,
        edge_gating: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.edge_dim = edge_dim
        self.edge_gating = bool(edge_gating)
        self.message_network = _mlp(2 * state_dim + edge_dim, hidden_dim, hidden_dim, layers, dropout)
        self.edge_gate_network = (
            _mlp(2 * state_dim + edge_dim, hidden_dim, 1, max(1, layers - 1), dropout)
            if self.edge_gating
            else None
        )
        if self.edge_gate_network is not None:
            final = next(module for module in reversed(self.edge_gate_network) if isinstance(module, nn.Linear))
            with torch.no_grad():
                final.weight.normal_(mean=0.0, std=1e-3)
                final.bias.fill_(2.0)
        self.update_network = _mlp(state_dim + hidden_dim, hidden_dim, state_dim, layers, dropout)

    def forward(self, states: Tensor, graph: TorchGraph | Mapping[str, Tensor] | Any) -> Tensor:
        if states.ndim != 2:
            raise ValueError("graph message passing expects [nodes, states]")
        edge_index, edge_attr = unpack_graph(graph)
        edge_index = edge_index.to(device=states.device, dtype=torch.long)
        edge_attr = edge_attr.to(device=states.device, dtype=states.dtype)
        source, target = edge_index
        edge_features = torch.cat((states[target], states[source], edge_attr), dim=-1)
        messages = self.message_network(edge_features)
        gates = (
            torch.sigmoid(self.edge_gate_network(edge_features))
            if self.edge_gate_network is not None
            else torch.ones((len(target), 1), dtype=states.dtype, device=states.device)
        )
        messages = messages * gates
        # Scatter accumulation stays in the state dtype for numerical stability
        # and because index_add_ requires matching dtypes under CUDA autocast.
        messages = messages.to(dtype=states.dtype)
        aggregate = torch.zeros(
            states.shape[0], messages.shape[-1], dtype=states.dtype, device=states.device
        )
        aggregate.index_add_(0, target, messages)
        degree = torch.zeros(states.shape[0], 1, dtype=states.dtype, device=states.device)
        degree.index_add_(0, target, gates.to(dtype=states.dtype))
        aggregate = aggregate / degree.clamp_min(1.0)
        return self.update_network(torch.cat((states, aggregate), dim=-1))


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
    dropout: float,
) -> nn.Sequential:
    if layers < 1:
        raise ValueError("layers must be positive")
    modules: list[nn.Module] = []
    for layer in range(layers):
        modules.extend(
            [nn.Linear(input_dim if layer == 0 else hidden_dim, hidden_dim), nn.SiLU()]
        )
        if dropout:
            modules.append(nn.Dropout(dropout))
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)
