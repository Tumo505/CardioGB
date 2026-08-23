from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from cardiogb.data.state_dataset import StateDataset
from cardiogb.utils.config import load_yaml
from cardiogb.utils.io import atomic_json, export_table


def required(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required supplementary input missing: {path}")
    return path


def copy_table(source: Path, output: Path) -> None:
    required(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile final CardioGB supplementary tables")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("manuscript/supplementary_tables"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = StateDataset.load(args.data)
    summary = (
        pd.DataFrame({"stage_days": dataset.times, "section": dataset.sections, "biological_unit": dataset.groups})
        .groupby("stage_days", observed=True)
        .agg(spots=("section", "size"), sections=("section", "nunique"), biological_units=("biological_unit", "nunique"))
        .reset_index()
    )
    export_table(summary, args.output_dir / "Table_S1_dataset_summary.csv")

    pathways = load_yaml("configs/pathways.yaml")
    state_lookup = {value: key for key, value in pathways["states"].items()}
    rows = []
    for pathway, details in pathways["pathways"].items():
        for gene in details["genes"]:
            rows.append({
                "state": state_lookup[pathway],
                "pathway": pathway,
                "gene": gene,
                "description": details["description"],
                "expected_peak": details["expected_peak"],
                "pathway_version": pathways["version"],
            })
    export_table(pd.DataFrame(rows), args.output_dir / "Table_S2_curated_pathway_genes.csv")

    root = args.results_root
    sources = {
        "Table_S3_E1_seed_transition_metrics.csv": root / "final_full_multiseed" / "tables" / "benchmark_metrics.csv",
        "Table_S4_E1_paired_statistics.csv": root / "formal_statistics_revised" / "e1_paired_tests.csv",
        "Table_S5_E2_interpolation_metrics.csv": root / "e2_interpolation_revised" / "tables" / "all_metrics.csv",
        "Table_S6_E3_extrapolation_metrics.csv": root / "e3_extrapolation_revised" / "tables" / "all_metrics.csv",
        "Table_S7_E4_grouped_CV_metrics.csv": root / "e4_group_cv_full" / "tables" / "all_metrics.csv",
        "Table_S8_E5_parameter_recovery.csv": root / "synthetic_recovery_full" / "tables" / "e5_parameter_recovery.csv",
        "Table_S9_E6_hidden_mechanism_recovery.csv": root / "synthetic_recovery_full" / "tables" / "e6_hidden_recovery.csv",
        "Table_S10_E8_ablation_metrics.csv": root / "final_full_ablations" / "tables" / "ablation_metrics.csv",
        "Table_S11_E7_parameter_stability.csv": root / "e7_full_interpretation" / "tables" / "parameter_stability.csv",
        "Table_S12_E7_residual_attribution.csv": root / "e7_full_interpretation" / "tables" / "residual_attribution_all_sections.csv",
        "Table_S13_external_prediction.csv": root / "external_predictive_validation_revised" / "metrics" / "external_prediction.csv",
        "Table_S14_uncertainty_inference.csv": root / "external_predictive_validation_revised" / "tables" / "uncertainty_inferential_tests.csv",
        "Table_S15_human_MI_omnibus.csv": root / "human_snatac_validation_revised" / "tables" / "patient_group_tests.csv",
        "Table_S16_human_MI_posthoc_effects.csv": root / "human_snatac_validation_revised" / "tables" / "patient_group_posthoc_effects.csv",
    }
    for name, source in sources.items():
        copy_table(source, args.output_dir / name)
    atomic_json(
        {
            "status": "complete",
            "tables": ["Table_S1_dataset_summary.csv", "Table_S2_curated_pathway_genes.csv", *sources],
            "source_results_root": str(root),
        },
        args.output_dir / "supplementary_tables_manifest.json",
    )
    print(json.dumps({"status": "complete", "tables": len(sources) + 2}, indent=2))


if __name__ == "__main__":
    main()
