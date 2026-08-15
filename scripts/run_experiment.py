from __future__ import annotations

import argparse
import json
from pathlib import Path

from cardiogb.experiments import (
    experiment_readiness,
    run_synthetic_hidden_recovery,
    run_synthetic_parameter_recovery,
)
from cardiogb.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--processed-data", default="data/processed/zebrafish_regeneration.h5ad")
    parser.add_argument("--output", type=Path, default=Path("results/metrics/experiment.json"))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.experiment == "e5_parameter_recovery":
        result = run_synthetic_parameter_recovery(
            load_yaml("configs/mechanistic_model.yaml"), epochs=args.epochs, output=args.output
        )
    elif args.experiment == "e6_hidden_mechanism":
        result = run_synthetic_hidden_recovery(
            load_yaml("configs/mechanistic_model.yaml"), epochs=args.epochs, output=args.output
        )
    else:
        result = experiment_readiness(args.experiment, args.processed_data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
