from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_full_interpretation import local_mechanistic_sensitivities

from cardiogb.models.factory import build_model
from cardiogb.utils.config import load_yaml


def test_local_sensitivity_profiles_cover_all_rates_and_states() -> None:
    model = build_model(
        "cardiogb",
        load_yaml(ROOT / "configs" / "model.yaml"),
        load_yaml(ROOT / "configs" / "mechanistic_model.yaml"),
    )
    states = torch.linspace(0.1, 0.9, 60, dtype=torch.float32).reshape(10, 6)
    rows = local_mechanistic_sensitivities(model, 3.0, states)
    parameters = set(model.mechanistic_model.raw_parameters)
    state_names = set(model.mechanistic_model.state_names)
    assert len(rows) == len(parameters) * len(state_names) == 90
    assert {row["parameter"] for row in rows} == parameters
    assert {row["target_state"] for row in rows} == state_names
    assert np.isfinite([row["signed_local_sensitivity"] for row in rows]).all()
    assert np.isfinite([row["absolute_local_sensitivity"] for row in rows]).all()
    for parameter in parameters:
        profile = [row["absolute_local_sensitivity"] for row in rows if row["parameter"] == parameter]
        assert np.count_nonzero(np.asarray(profile) > 0) == 1
