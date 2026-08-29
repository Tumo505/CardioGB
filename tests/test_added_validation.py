from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_added_perturbations import exact_difference_test
from build_added_validation_state_datasets import STATE_NAMES, build_dataset

from cardiogb.data.state_dataset import StateDataset


def test_external_table_conversion_is_edgeless_and_locked(tmp_path: Path) -> None:
    table = tmp_path / "states.csv"
    frame = pd.DataFrame(
        {
            **{state: [0.1, 0.2, 0.3] for state in STATE_NAMES},
            "stage_days": [0.0, 1.0, 1.0],
            "sample": ["s0", "s1", "s2"],
            "observation": ["a", "b", "c"],
        }
    )
    frame.to_csv(table, index=False)
    output = tmp_path / "states.npz"
    report = build_dataset(
        table,
        output,
        time_column="stage_days",
        sample_column="sample",
        observation_column="observation",
    )
    dataset = StateDataset.load(output)
    assert report["training_use"] == "none; locked external validation only"
    assert len(np.unique(dataset.sections)) == len(dataset.states)
    assert dataset.build_graph(k=2).edge_index.shape[1] == 0


def test_exact_biological_unit_permutation_has_valid_resolution() -> None:
    p_value = exact_difference_test(np.asarray([0.0, 0.1]), np.asarray([0.9, 1.0]))
    assert p_value == 2.0 / 6.0
