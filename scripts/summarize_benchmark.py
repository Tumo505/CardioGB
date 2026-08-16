from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cardiogb.utils.io import export_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.metrics_dir.glob("*_test.csv"))
    if not files:
        raise FileNotFoundError(f"No *_test.csv files in {args.metrics_dir}")
    frames = [pd.read_csv(path) for path in files]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    required = {"model", "transition", "mmd", "moment_error", "sliced_wasserstein"}
    if not required.issubset(combined.columns):
        raise ValueError(f"Missing benchmark columns: {sorted(required - set(combined.columns))}")
    export_table(combined, args.output)
    metrics = ["mmd", "moment_error", "sliced_wasserstein"]
    summary = combined.groupby("model", observed=True)[metrics].agg(["mean", "std", "median"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    export_table(summary.reset_index(), args.summary)
    print(summary.to_string())


if __name__ == "__main__":
    main()
