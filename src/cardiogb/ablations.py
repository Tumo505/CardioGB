"""Controlled ablations for spatial and mechanistic model components."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch

from cardiogb.models.message_passing import TorchGraph


def shuffle_graph(graph: TorchGraph, *, seed: int = 0) -> TorchGraph:
    """Relabel nodes while preserving edge count and edge-feature rows."""
    if graph.edge_index.numel() == 0:
        return graph
    n_nodes = int(graph.edge_index.max()) + 1
    generator = torch.Generator(device=graph.edge_index.device).manual_seed(seed)
    permutation = torch.randperm(n_nodes, generator=generator, device=graph.edge_index.device)
    return TorchGraph(permutation[graph.edge_index], graph.edge_attr.clone())


def disable_interaction(
    mechanistic_config: Mapping[str, Any], interaction_name: str
) -> dict[str, Any]:
    """Remove a named or ``source->target`` interaction from a copied config."""
    result = deepcopy(dict(mechanistic_config))
    interactions = result.get("interactions", [])
    kept = [
        item
        for item in interactions
        if item.get("name", f"{item.get('source')}->{item.get('target')}") != interaction_name
    ]
    if len(kept) == len(interactions):
        raise KeyError(f"unknown mechanistic interaction: {interaction_name}")
    result["interactions"] = kept
    return result


def ablation_registry() -> dict[str, dict[str, Any]]:
    """Canonical experiment factors; values are config overrides or operations."""
    return {
        "full": {},
        "shuffled_graph": {"graph_operation": "shuffle"},
        "no_residual_penalty": {"loss.lambda_residual": 0.0},
        "no_spatial_penalty": {"loss.lambda_spatial": 0.0},
        "small_residual": {"gnn.hidden_dim": 16, "gnn.layers": 1},
        "large_residual": {"gnn.hidden_dim": 128, "gnn.layers": 3},
    }
