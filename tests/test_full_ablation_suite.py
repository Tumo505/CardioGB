from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_real_ablations import RESIDUAL_PENALTY_MULTIPLIERS, VARIANTS, make_model

from cardiogb.models import CardioGB
from cardiogb.models.constraints import positive
from cardiogb.protocols import evaluate_transitions
from cardiogb.training.trainer import CrossSectionalTransition
from cardiogb.utils.config import load_yaml


def test_full_ablation_registry_covers_registered_components() -> None:
    assert {
        "no_mechanism",
        "no_graph",
        "no_constraints",
        "shuffled_graph",
        "mechanistic_misspecification",
        "residual_penalty_zero",
        "residual_penalty_low",
        "residual_penalty_high",
        "state_definition",
    } == set(VARIANTS)
    assert RESIDUAL_PENALTY_MULTIPLIERS == {
        "residual_penalty_zero": 0.0,
        "residual_penalty_low": 0.25,
        "residual_penalty_high": 4.0,
    }


def test_no_constraint_and_misspecification_models_change_registered_structure() -> None:
    model_config = load_yaml(ROOT / "configs" / "model.yaml")
    mechanistic_config = load_yaml(ROOT / "configs" / "mechanistic_model.yaml")
    unconstrained = make_model("no_constraints", model_config, mechanistic_config)
    values = torch.tensor([-1.0, 2.0])
    assert torch.equal(unconstrained.project_state(values), values)
    assert unconstrained.mechanistic_model.parameter_transform == "identity"
    assert torch.equal(positive(values, "identity"), values)
    misspecified = make_model("mechanistic_misspecification", model_config, mechanistic_config)
    assert "gamma_C" not in misspecified.mechanistic_model.raw_parameters
    assert not any(
        misspecified.mechanistic_model.state_names[item.source] == "F"
        and misspecified.mechanistic_model.state_names[item.target] == "C"
        for item in misspecified.mechanistic_model.interactions
    )


def test_no_mechanism_preserves_cardiogb_safeguards() -> None:
    model_config = load_yaml(ROOT / "configs" / "model.yaml")
    mechanistic_config = load_yaml(ROOT / "configs" / "mechanistic_model.yaml")
    model = make_model("no_mechanism", model_config, mechanistic_config)
    assert isinstance(model, CardioGB)
    assert not model.mechanistic_enabled
    values = torch.tensor([[-0.2] * 6, [1.2] * 6], dtype=torch.float32)
    projected = model.project_state(values)
    assert torch.all(projected >= 0.0)
    assert torch.all(projected <= 1.0)
    model.residual_enabled = False
    fields = model.vector_field(0.0, torch.full((2, 6), 0.5))
    assert torch.equal(fields["mechanistic"], torch.zeros(2, 6))
    assert torch.equal(fields["total"], torch.zeros(2, 6))


class ZeroField(nn.Module):
    def forward(self, t, states, graph=None):
        return torch.zeros_like(states)


def test_transition_evaluation_records_constraint_and_stability_diagnostics() -> None:
    source = torch.tensor([[-0.2, 1.2], [0.5, 0.5]], dtype=torch.float32)
    target = torch.tensor([[0.1, 0.9], [0.4, 0.6]], dtype=torch.float32)
    transition = CrossSectionalTransition(source, target, None, 0.0, 1.0, "test")
    row = evaluate_transitions(
        ZeroField(), [transition], device="cpu", step_size=0.25, max_steps_per_transition=8
    )[0]
    assert row["numerically_stable"]
    assert row["prediction_finite_fraction"] == 1.0
    assert np.isclose(row["prediction_out_of_bounds_fraction"], 0.5)
    assert np.isclose(row["prediction_min"], -0.2)
    assert np.isclose(row["prediction_max"], 1.2)
