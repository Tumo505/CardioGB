"""Configuration-driven model construction."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from cardiogb.models.cardiogb import CardioGB
from cardiogb.models.graph_neural_ode import GraphNeuralODEFunc
from cardiogb.models.mechanistic import MechanisticODE
from cardiogb.models.neural_ode import NeuralODEFunc
from cardiogb.models.persistence import PersistenceBaseline


def build_model(
    name: str,
    model_config: Mapping[str, Any],
    mechanistic_config: Mapping[str, Any],
) -> nn.Module:
    states = model_config.get("states", ("I", "A", "F", "C", "V", "M"))
    state_dim = len(states)
    gnn = model_config.get("gnn", {})
    common = {
        "state_dim": state_dim,
        "hidden_dim": int(gnn.get("hidden_dim", 64)),
        "layers": int(gnn.get("layers", 2)),
        "dropout": float(gnn.get("dropout", 0.0)),
    }
    if name == "persistence":
        return PersistenceBaseline()
    if name == "mechanistic_ode":
        return MechanisticODE.from_config(mechanistic_config)
    if name == "neural_ode":
        return NeuralODEFunc(**common)
    if name == "graph_neural_ode":
        return GraphNeuralODEFunc(edge_dim=3, **common)
    if name == "cardiogb":
        return CardioGB(
            MechanisticODE.from_config(mechanistic_config),
            GraphNeuralODEFunc(edge_dim=3, **common),
        )
    raise ValueError(f"Unknown model: {name}")
