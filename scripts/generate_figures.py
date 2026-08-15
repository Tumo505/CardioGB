from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a benchmark summary plot from saved metrics")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    frame = pd.read_csv(args.metrics)
    required = {"model", "value"}
    if not required.issubset(frame):
        raise ValueError(f"Metrics table must contain {sorted(required)}")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    frame.groupby("model")["value"].mean().sort_values().plot.bar(ax=axis)
    axis.set_ylabel("Metric value")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
