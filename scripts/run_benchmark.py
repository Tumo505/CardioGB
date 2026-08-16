from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cardiogb.data.splits import grouped_split
from cardiogb.data.state_dataset import StateDataset
from cardiogb.metrics import distribution_metrics
from cardiogb.utils.io import export_table
from train_cardiogb import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the leakage-safe five-model benchmark")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    dataset = StateDataset.load(args.data)
    metadata = pd.DataFrame({"group": dataset.groups, "stage": dataset.times.astype(str)})
    _, _, test_mask, definition = grouped_split(
        metadata, group_column="group", stage_column="stage", seed=args.seed
    )
    definition.save(args.output_dir / "tables" / "split.json")
    persistence_rows = []
    for transition in dataset.transitions(mask=test_mask):
        persistence_rows.append(
            {
                "model": "persistence",
                "transition": transition.name,
                **distribution_metrics(
                    transition.source_states.numpy(), transition.target_states.numpy()
                ),
            }
        )
    export_table(
        pd.DataFrame(persistence_rows), args.output_dir / "metrics" / "persistence_test.csv"
    )
    reports = {"persistence": {"test_transitions": len(persistence_rows)}}
    for name in ("mechanistic_ode", "neural_ode", "graph_neural_ode", "cardiogb"):
        reports[name] = train_model(
            args.data,
            name,
            args.output_dir,
            epochs_override=args.epochs,
            seed_override=args.seed,
        )
    print(json.dumps({"seed": args.seed, "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
