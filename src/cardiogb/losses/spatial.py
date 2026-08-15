"""Spatial regularization with distance-aware graph weights."""

from __future__ import annotations

import torch
from torch import Tensor

from cardiogb.models.message_passing import unpack_graph


def graph_smoothness(states: Tensor, graph: object, distance_scale: float = 1.0) -> Tensor:
    if distance_scale <= 0:
        raise ValueError("distance_scale must be positive")
    edge_index, edge_attr = unpack_graph(graph)
    edge_index = edge_index.to(states.device)
    edge_attr = edge_attr.to(states.device, states.dtype)
    if edge_index.shape[1] == 0:
        return states.new_zeros(())
    source, target = edge_index
    weights = torch.exp(-edge_attr[:, 0] / distance_scale)
    difference = (states[source] - states[target]).square().sum(dim=-1)
    return (weights * difference).mean()
