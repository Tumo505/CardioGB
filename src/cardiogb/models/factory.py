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
        return GraphNeuralODEFunc(
            edge_dim=3,
            edge_gating=bool(gnn.get("state_dependent_edge_gate", True)),
            **common,
        )
    if name == "cardiogb":
        constraints = model_config.get("constraints", {})
        residual = model_config.get("residual", {})
        gate = model_config.get("mechanistic_gate", {})
        stability = model_config.get("stability", {})
        persistence = model_config.get("persistence_gate", {})
        residual_model = GraphNeuralODEFunc(
            edge_dim=3,
            edge_gating=bool(gnn.get("state_dependent_edge_gate", True)),
            **common,
        )
        if bool(residual.get("spectral_normalization", False)):
            for module in residual_model.modules():
                if isinstance(module, nn.Linear):
                    nn.utils.parametrizations.spectral_norm(module)
        return CardioGB(
            MechanisticODE.from_config(mechanistic_config),
            residual_model,
            state_min=float(constraints.get("state_min", 0.0)),
            state_max=float(constraints.get("state_max", 1.0)),
            residual_scale_max=float(residual.get("scale_max", 0.25)),
            residual_scale_initial=float(residual.get("scale_initial", 0.05)),
            mechanistic_gate_min=float(gate.get("minimum", 0.05)),
            mechanistic_gate_initial=float(gate.get("initial", 0.5)),
            learn_mechanistic_gate=bool(gate.get("learned", False)),
            orthogonal_residual=bool(residual.get("orthogonal", True)),
            orthogonal_projection_strength=float(residual.get("orthogonal_strength", 1.0)),
            velocity_limit=stability.get("velocity_limit", 0.5),
            persistence_gate=bool(persistence.get("enabled", True)),
            persistence_gate_initial=float(persistence.get("initial", 0.95)),
            persistence_horizon_slope_initial=float(
                persistence.get("horizon_slope_initial", 0.05)
            ),
        )
    raise ValueError(f"Unknown model: {name}")
