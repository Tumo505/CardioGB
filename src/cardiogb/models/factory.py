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
        constraints = model_config.get("constraints", {})
        residual = model_config.get("residual", {})
        gate = model_config.get("mechanistic_gate", {})
        return CardioGB(
            MechanisticODE.from_config(mechanistic_config),
            GraphNeuralODEFunc(edge_dim=3, **common),
            state_min=float(constraints.get("state_min", 0.0)),
            state_max=float(constraints.get("state_max", 1.0)),
            residual_scale_max=float(residual.get("scale_max", 0.25)),
            residual_scale_initial=float(residual.get("scale_initial", 0.05)),
            mechanistic_gate_min=float(gate.get("minimum", 0.05)),
            mechanistic_gate_initial=float(gate.get("initial", 0.5)),
        )
    raise ValueError(f"Unknown model: {name}")
