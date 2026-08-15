"""Experiment registry and synthetic experiment execution."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from cardiogb.models.mechanistic import MechanisticODE
from cardiogb.models.neural_ode import NeuralODEFunc
from cardiogb.synthetic.recovery import recover_hidden_mechanism, recover_mechanistic_parameters
from cardiogb.synthetic.simulator import simulate_system
from cardiogb.utils.config import load_yaml
from cardiogb.utils.io import atomic_json


CORE_EXPERIMENTS = {
    "e1_benchmark",
    "e2_interpolation",
    "e3_extrapolation",
    "e4_group_cv",
    "e5_parameter_recovery",
    "e6_hidden_mechanism",
    "e7_mechanistic_insufficiency",
    "e8_ablations",
    "e9_uncertainty",
    "e10_mouse_validation",
}


def run_synthetic_parameter_recovery(
    mechanistic_config: Mapping[str, Any],
    *,
    epochs: int = 100,
    output: str | Path | None = None,
) -> dict[str, Any]:
    true_model = MechanisticODE.from_config(mechanistic_config)
    simulated = simulate_system(
        true_model,
        observation_times=[0.0, 0.25, 0.5, 1.0, 3.0],
        num_entities=64,
        noise_std=0.01,
        step_size=0.05,
    )
    fitted = MechanisticODE.from_config(mechanistic_config)
    with torch.no_grad():
        for parameter in fitted.raw_parameters.values():
            parameter.add_(torch.randn_like(parameter) * 0.35)
    recovered = recover_mechanistic_parameters(
        fitted,
        simulated.times,
        simulated.observations,
        simulated.true_parameters,
        epochs=epochs,
        step_size=0.05,
    )
    result = asdict(recovered)
    if output is not None:
        atomic_json(result, output)
    return result


def run_synthetic_hidden_recovery(
    mechanistic_config: Mapping[str, Any],
    *,
    epochs: int = 200,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Recover a known omitted I×F term in a synthetic-only oracle test."""
    true_model = MechanisticODE.from_config(mechanistic_config)
    simulated = simulate_system(
        true_model,
        observation_times=[0.0, 0.25, 0.5, 1.0, 3.0],
        num_entities=64,
        noise_std=0.01,
        hidden_mechanism=True,
        hidden_strength=0.4,
        step_size=0.05,
    )
    residual = NeuralODEFunc(state_dim=6, hidden_dim=32, layers=2, time_dependent=False)
    recovered = recover_hidden_mechanism(
        residual,
        simulated.latent_clean,
        simulated.hidden_mechanism_values,
        epochs=epochs,
        learning_rate=3e-3,
    )
    result = asdict(recovered)
    result["interpretation"] = "synthetic oracle recovery; not a causal real-data claim"
    if output is not None:
        atomic_json(result, output)
    return result


def experiment_readiness(name: str, processed_data: str | Path) -> dict[str, Any]:
    if name not in CORE_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}")
    synthetic = name in {"e5_parameter_recovery", "e6_hidden_mechanism"}
    processed_exists = Path(processed_data).is_file()
    return {
        "name": name,
        "ready": synthetic or processed_exists,
        "synthetic": synthetic,
        "processed_data_exists": processed_exists,
        "blocker": None if synthetic or processed_exists else "processed zebrafish state dataset missing",
    }
