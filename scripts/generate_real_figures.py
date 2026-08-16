from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_LABELS = {
    "I": "Inflammation",
    "A": "Activation",
    "F": "Fibroblast/ECM",
    "C": "CM regeneration",
    "V": "Vascularisation",
    "M": "Mature myocardium",
}


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real-data QC and pilot figures")
    parser.add_argument("--pathway-qc", type=Path, required=True)
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--mi-stage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pathway = pd.read_csv(args.pathway_qc).sort_values("stage_days")
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for state, label in STATE_LABELS.items():
        axis.plot(pathway["stage_days"], pathway[f"{state}_mean"], marker="o", label=label)
    axis.set(xlabel="Days post-amputation", ylabel="Mean programme score", ylim=(0, 1))
    axis.legend(frameon=False, ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, args.output_dir / "pathway_stage_qc.png")

    benchmark = pd.read_csv(args.benchmark_summary).sort_values("mmd_mean")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    lower_error = np.minimum(benchmark["mmd_std"], benchmark["mmd_mean"])
    axis.barh(
        benchmark["model"],
        benchmark["mmd_mean"],
        xerr=np.vstack((lower_error, benchmark["mmd_std"])),
    )
    axis.set_xlim(left=0)
    axis.set(xlabel="MMD (lower is better)", ylabel="")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("10-epoch real-data pilot; not final model comparison")
    save(figure, args.output_dir / "pilot_benchmark_mmd.png")

    mi = pd.read_csv(args.mi_stage).sort_values("stage_days")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(mi["stage_days"], mi["mean"], marker="o", color="#A23B72")
    axis.fill_between(
        mi["stage_days"], mi["mean"] - mi["std"], mi["mean"] + mi["std"],
        color="#A23B72", alpha=0.18,
    )
    axis.set(xlabel="Days post-amputation", ylabel="Mechanistic insufficiency", ylim=(0, 1))
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("Pilot CardioGB observed-state MI")
    save(figure, args.output_dir / "pilot_mi_by_stage.png")


if __name__ == "__main__":
    main()
