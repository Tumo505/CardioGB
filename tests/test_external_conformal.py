from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_external_prediction import conformal_factor, evaluate


def validation_groups() -> dict[str, dict]:
    rng = np.random.default_rng(17)
    grouped = {}
    for transition in range(4):
        members = np.clip(
            rng.normal(0.4 + 0.02 * transition, 0.03, size=(5, 12, 6)), 0.0, 1.0
        )
        weights = np.repeat(0.2, 5)
        grouped[f"t{transition}"] = {
            "members": members,
            "sources": np.clip(members[0] - 0.02, 0.0, 1.0),
            "target": np.clip(members.mean(axis=0) + np.linspace(0.01, 0.06, 6), 0.0, 1.0),
            "mean": np.tensordot(weights, members, axes=(0, 0)),
            "aggregation_method": "simplex",
            "t0": float(transition),
            "t1": float(transition + 1),
        }
    return grouped


def test_conformal_radii_are_pathway_conditional_and_validation_only() -> None:
    grouped = validation_groups()
    weights = np.repeat(0.2, 5)
    scale, radius, scores, absolute = conformal_factor(grouped, weights, 0.95)
    assert scale.shape == radius.shape == (6,)
    assert scores.shape == absolute.shape == (4, 6)
    assert np.isfinite(scale).all() and np.isfinite(radius).all()
    assert len(np.unique(np.round(radius, 8))) > 1


def test_external_evaluation_uses_state_specific_radii() -> None:
    grouped = validation_groups()
    weights = np.repeat(0.2, 5)
    scale, radius, _, _ = conformal_factor(grouped, weights, 0.95)
    metrics, states = evaluate(grouped, weights, scale, radius, ("I", "A", "F", "C", "V", "M"), "validation")
    assert len(metrics) == 4
    assert len(states) == 24
    first = [row["interval_radius"] for row in states if row["transition"] == "t0"]
    assert np.allclose(first, radius)
