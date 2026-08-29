"""Convert processed added-validation tables to model-ready state datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cardiogb.data.state_dataset import StateDataset


STATE_NAMES = ("I", "A", "F", "C", "V", "M")


def build_dataset(
    table: Path,
    output: Path,
    *,
    time_column: str,
    sample_column: str,
    observation_column: str,
) -> dict[str, object]:
    frame = pd.read_csv(table)
    required = {*STATE_NAMES, time_column, sample_column, observation_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{table} is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=[*STATE_NAMES, time_column, sample_column, observation_column])
    # These cohorts are bulk or dissociated-cell assays. Giving every observation
    # its own section intentionally yields an edgeless graph instead of inventing
    # spatial adjacency from row order.
    sections = np.asarray(
        [f"{sample}:{observation}" for sample, observation in zip(frame[sample_column], frame[observation_column])],
        dtype=str,
    )
    dataset = StateDataset(
        states=frame.loc[:, STATE_NAMES].to_numpy(dtype=np.float32),
        coordinates=np.zeros((len(frame), 2), dtype=np.float64),
        sections=sections,
        times=frame[time_column].to_numpy(dtype=np.float64),
        groups=frame[sample_column].astype(str).to_numpy(),
        state_names=STATE_NAMES,
        domains=np.repeat("nonspatial_external", len(frame)),
    )
    dataset.save(output)
    return {
        "input": str(table),
        "output": str(output),
        "observations": len(frame),
        "biological_samples": int(frame[sample_column].nunique()),
        "stages": sorted(frame[time_column].astype(float).unique().tolist()),
        "graph_policy": "edgeless; no spatial coordinates available",
        "training_use": "none; locked external validation only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=Path("results/added_validation"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/zebrafish/validation")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        build_dataset(
            args.tables / "gse106884_sample_states.csv",
            args.output_dir / "gse106884_states.npz",
            time_column="stage_days",
            sample_column="sample",
            observation_column="replicate",
        ),
        build_dataset(
            args.tables / "gse237276_cell_states.csv",
            args.output_dir / "gse237276_states.npz",
            time_column="stage_days",
            sample_column="sample",
            observation_column="cell",
        ),
    ]
    pd.DataFrame(reports).to_json(
        args.tables / "model_ready_validation_datasets.json", orient="records", indent=2
    )
    print(pd.DataFrame(reports).to_string(index=False))


if __name__ == "__main__":
    main()
