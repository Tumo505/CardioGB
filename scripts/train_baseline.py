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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["persistence", "mechanistic_ode", "neural_ode", "graph_neural_ode"],
        default="persistence",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.model != "persistence":
        print(json.dumps(train_model(args.data, args.model, args.output_dir)))
        return
    dataset = StateDataset.load(args.data)
    metadata = pd.DataFrame({"group": dataset.groups, "stage": dataset.times.astype(str)})
    _, _, test_mask, definition = grouped_split(
        metadata, group_column="group", stage_column="stage", seed=20260815
    )
    definition.save(args.output_dir / "tables" / "split.json")
    rows = []
    for transition in dataset.transitions(mask=test_mask):
        metrics = distribution_metrics(
            transition.source_states.numpy(), transition.target_states.numpy()
        )
        rows.append({"model": "persistence", "transition": transition.name, **metrics})
    output = args.output_dir / "metrics" / "persistence.csv"
    export_table(pd.DataFrame(rows), output)
    print(json.dumps({"model": "persistence", "transitions": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
