from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from cardiogb.data.state_dataset import StateDataset
from cardiogb.utils.io import atomic_json, export_table


def stage_means(dataset: StateDataset, stages: list[float]) -> np.ndarray:
    rows = []
    for stage in stages:
        selected = dataset.times == stage
        if not selected.any():
            raise ValueError(f"stage {stage:g} missing")
        rows.append(dataset.states[selected].mean(axis=0))
    return np.asarray(rows)


def exact_permutation_pvalue(x: np.ndarray, y: np.ndarray, observed: float) -> float:
    values = []
    for order in itertools.permutations(range(len(y))):
        values.append(abs(spearmanr(x, y[list(order)]).statistic))
    return float(np.mean(np.asarray(values) >= abs(observed) - 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pathway-level mouse conservation analysis")
    parser.add_argument("--zebrafish", type=Path, required=True)
    parser.add_argument("--mouse", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    zebrafish = StateDataset.load(args.zebrafish)
    mouse = StateDataset.load(args.mouse)
    if zebrafish.state_names != mouse.state_names:
        raise ValueError("state definitions disagree across species")
    fish_stages = [3.0, 7.0, 14.0, 28.0]
    mouse_stages = [3.0, 7.0, 14.0, 21.0]
    fish = stage_means(zebrafish, fish_stages)
    murine = stage_means(mouse, mouse_stages)
    stage_rows = []
    for species, stages, values in (
        ("zebrafish", fish_stages, fish),
        ("mouse", mouse_stages, murine),
    ):
        for stage, row in zip(stages, values):
            stage_rows.append({"species": species, "stage_days": stage, **dict(zip(zebrafish.state_names, row))})
    export_table(pd.DataFrame(stage_rows), args.output_dir / "matched_stage_scores.csv")
    rows = []
    for index, state in enumerate(zebrafish.state_names):
        rho = float(spearmanr(fish[:, index], murine[:, index]).statistic)
        rows.append(
            {
                "state": state,
                "spearman": rho,
                "pearson": float(pearsonr(fish[:, index], murine[:, index]).statistic),
                "exact_spearman_permutation_p": exact_permutation_pvalue(fish[:, index], murine[:, index], rho),
                "matched_phases": 4,
            }
        )
    export_table(pd.DataFrame(rows), args.output_dir / "pathway_conservation.csv")
    atomic_json(
        {
            "status": "complete",
            "mode": "pathway-level ordinal repair-phase comparison",
            "zebrafish_stages": fish_stages,
            "mouse_stages": mouse_stages,
            "direct_zero_shot_prediction": False,
            "limitations": [
                "four matched phases provide low statistical resolution",
                "mouse spatial data have one sample per stage",
                "correlations are descriptive conservation evidence, not causal or dynamical transfer",
            ],
        },
        args.output_dir / "validation_manifest.json",
    )
    print(json.dumps({"status": "complete", "states": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
