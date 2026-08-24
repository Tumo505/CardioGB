from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_human_snatac import (
    clustered_group_model,
    clustered_median_difference_ci,
    clustered_pair_contrast,
)


def partially_repeated_patient_table() -> pd.DataFrame:
    rows = []
    for index in range(12):
        patient = f"P{index:02d}"
        if index < 4:
            groups = ("ischemic", "myogenic")
        elif index < 8:
            groups = ("fibrotic",)
        else:
            groups = ("myogenic",)
        for group in groups:
            offset = {"fibrotic": -0.2, "ischemic": 0.3, "myogenic": 0.0}[group]
            rows.append({"patient": patient, "patient_group": group, "pathway": index * 0.01 + offset})
    return pd.DataFrame(rows)


def test_cluster_model_counts_unique_patients_not_patient_group_rows() -> None:
    frame = partially_repeated_patient_table()
    data, groups, model, omnibus = clustered_group_model(frame, "pathway")
    assert len(data) == 16
    assert data["patient"].nunique() == 12
    assert groups == ["fibrotic", "ischemic", "myogenic"]
    assert np.isfinite(float(omnibus.statistic))
    assert 0.0 <= float(omnibus.pvalue) <= 1.0
    estimate, standard_error, statistic, pvalue = clustered_pair_contrast(
        model, groups, "ischemic", "myogenic"
    )
    assert estimate > 0
    assert standard_error > 0
    assert np.isfinite(statistic)
    assert 0.0 <= pvalue <= 1.0


def test_cluster_bootstrap_resamples_whole_patients() -> None:
    frame = partially_repeated_patient_table()
    estimate, lower, upper = clustered_median_difference_ci(
        frame, "pathway", "ischemic", "myogenic", seed=17, n_resamples=500
    )
    assert estimate > 0
    assert np.isfinite([lower, upper]).all()
    assert lower <= estimate <= upper
