from __future__ import annotations

import json
from pathlib import Path

from cardiogb.experiments import (
    CORE_EXPERIMENTS,
    experiment_readiness,
    run_synthetic_hidden_recovery,
    run_synthetic_parameter_recovery,
)
from cardiogb.utils.config import load_yaml


if __name__ == "__main__":
    config = load_yaml("configs/mechanistic_model.yaml")
    output = Path("results/metrics")
    run_synthetic_parameter_recovery(config, output=output / "e5_parameter_recovery.json")
    run_synthetic_hidden_recovery(config, output=output / "e6_hidden_mechanism.json")
    reports = [
        experiment_readiness(name, "data/processed/zebrafish_states.npz")
        for name in sorted(CORE_EXPERIMENTS)
    ]
    print(json.dumps(reports, indent=2))
