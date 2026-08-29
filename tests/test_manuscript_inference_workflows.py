from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_uncertainty_inference import transition_level_spearman
from run_full_interpretation import parameter_records

from cardiogb.models.factory import build_model
from cardiogb.utils.config import load_yaml


def test_transition_level_uncertainty_inference_uses_forecast_units() -> None:
    rows = [
        {
            "transition": f"t{transition}",
            "state": state,
            "horizon_days": float(transition + 1),
            "ensemble_std": 0.1 * (transition + 1) + 0.001 * state_index,
            "absolute_error": 0.2 * (transition + 1) + 0.001 * state_index,
        }
        for transition in range(7)
        for state_index, state in enumerate(("I", "A", "F", "C", "V", "M"))
    ]
    result = transition_level_spearman(
        pd.DataFrame(rows), "ensemble_std", "absolute_error", n_resamples=200
    )
    assert result["n_clusters"] == 7
    assert result["cluster_unit"] == "forecast transition"
    assert np.isfinite(result["spearman"])
    assert 0.0 <= result["p_value"] <= 1.0


def test_e7_parameter_records_have_unique_complete_state_scales() -> None:
    model = build_model(
        "cardiogb",
        load_yaml(ROOT / "configs" / "model.yaml"),
        load_yaml(ROOT / "configs" / "mechanistic_model.yaml"),
    )
    rows = parameter_records(model, member=0, seed=17, source="E4", case=2)
    names = [row["parameter"] for row in rows]
    expected_count = len(model.mechanistic_model.raw_parameters) + 2 * len(model.mechanistic_model.state_names)
    assert len(names) == len(set(names)) == expected_count
    assert {f"mechanistic_gate_{state}" for state in ("I", "A", "F", "C", "V", "M")} <= set(names)
    assert {f"residual_scale_{state}" for state in ("I", "A", "F", "C", "V", "M")} <= set(names)
    assert all(row["fit_id"] == "E4:case_2:seed_17" for row in rows)
    assert all(np.isfinite(row["value"]) for row in rows)
