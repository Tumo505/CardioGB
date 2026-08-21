from pathlib import Path

import numpy as np

from cardiogb.metrics.uncertainty import calibrated_mean_coverage, conformal_scale
from cardiogb.utils.config import load_yaml


def test_conformal_ensemble_calibration_covers_validation_state_means() -> None:
    predictions = np.asarray(
        [
            [[0.10, 0.20], [0.20, 0.30]],
            [[0.12, 0.18], [0.22, 0.28]],
            [[0.08, 0.22], [0.18, 0.32]],
        ]
    )
    observed = np.asarray([[0.15, 0.22], [0.25, 0.32]])
    calibration = conformal_scale(predictions, observed, confidence=0.8)
    result = calibrated_mean_coverage(predictions, observed, calibration["scale"])
    assert calibration["scale"] >= 0
    assert 0 <= result["coverage"] <= 1
    assert len(result["covered"]) == 2


def test_curated_human_pathways_exclude_implausible_automated_substitutions() -> None:
    config = load_yaml(Path("configs/human_pathways.yaml"))
    genes = {
        gene
        for details in config["pathways"].values()
        for gene in details["genes"]
    }
    assert len(config["pathways"]) == 6
    assert {"ATP2A2", "MPO", "LYZ"} <= genes
    assert not {"ATP1A4", "EPX", "LYZL1"} & genes
