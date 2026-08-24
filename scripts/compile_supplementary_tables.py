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
        "Table_S17_E2_confidence_intervals.csv": root / "formal_statistics_revised" / "e2_confidence_intervals.csv",
        "Table_S18_E3_confidence_intervals.csv": root / "formal_statistics_revised" / "e3_confidence_intervals.csv",
        "Table_S19_E4_biological_fold_confidence_intervals.csv": root / "formal_statistics_revised" / "e4_biological_unit_confidence_intervals.csv",
        "Table_S20_E5_parameter_recovery_confidence_intervals.csv": root / "synthetic_recovery_full" / "tables" / "e5_confidence_intervals.csv",
        "Table_S21_E6_hidden_recovery_confidence_intervals.csv": root / "synthetic_recovery_full" / "tables" / "e6_confidence_intervals.csv",
        "Table_S22_E8_paired_statistics.csv": root / "formal_statistics_revised" / "e8_paired_tests.csv",
        "Table_S23_ensemble_test_metrics.csv": root / "final_full_ensemble" / "metrics" / "ensemble_test.csv",
        "Table_S24_ensemble_weights.csv": root / "final_full_ensemble" / "tables" / "ensemble_weights.csv",
        "Table_S25_ensemble_aggregation_selection.csv": root / "final_full_ensemble" / "tables" / "aggregation_selection.csv",
        "Table_S26_mouse_matched_stage_scores.csv": root / "mouse_validation_revised" / "matched_stage_scores.csv",
        "Table_S27_mouse_pathway_conservation.csv": root / "mouse_validation_revised" / "pathway_conservation.csv",
        "Table_S28_external_descriptive_comparisons.csv": root / "formal_statistics_revised" / "external_descriptive_comparisons.csv",
        "Table_S29_E7_parameter_identifiability_diagnostics.csv": root / "e7_full_interpretation" / "tables" / "parameter_identifiability_diagnostics.csv",
        "Table_S30_E7_parameter_pair_correlations.csv": root / "e7_full_interpretation" / "tables" / "parameter_pair_correlations.csv",
        "Table_S31_E7_mechanistic_insufficiency_by_stage.csv": root / "e7_full_interpretation" / "tables" / "mi_stage_bootstrap.csv",
        "Table_S32_E7_mechanistic_insufficiency_pathway_associations.csv": root / "e7_full_interpretation" / "tables" / "mi_pathway_associations.csv",
        "Table_S33_human_pathway_feature_coverage.csv": root / "human_snatac_validation_revised" / "tables" / "pathway_feature_coverage.csv",
        "Table_S34_human_patient_region_pathways.csv": root / "human_snatac_validation_revised" / "tables" / "patient_region_pathway_accessibility.csv",
        "Table_S35_human_patient_celltype_pathways.csv": root / "human_snatac_validation_revised" / "tables" / "patient_celltype_pathway_accessibility.csv",
        "Table_S36_external_state_mean_predictions.csv": root / "external_predictive_validation_revised" / "tables" / "state_mean_predictions.csv",
        "Table_S37_human_patient_pathway_accessibility.csv": root / "human_snatac_validation_revised" / "tables" / "patient_pathway_accessibility.csv",
        "Table_S38_E7_local_parameter_sensitivity.csv": root / "e7_full_interpretation" / "tables" / "parameter_local_sensitivity.csv",
        "Table_S39_E7_local_sensitivity_profile_correlations.csv": root / "e7_full_interpretation" / "tables" / "parameter_local_sensitivity_correlations.csv",
    }
    calibration_metrics = root / "e3_extrapolation_horizon_calibrated" / "tables" / "all_metrics.csv"
    calibration_tests = root / "formal_statistics_revised" / "e3_horizon_calibration_paired_tests.csv"
    if calibration_metrics.is_file():
        sources["Table_S6b_E3_horizon_calibration_metrics.csv"] = calibration_metrics
    if calibration_tests.is_file():
        sources["Table_S6c_E3_horizon_calibration_statistics.csv"] = calibration_tests
    e7_protocol_stability = root / "e7_full_interpretation" / "tables" / "parameter_stability_by_protocol.csv"
    e7_heldout_attribution = root / "e7_full_interpretation" / "tables" / "residual_attribution_heldout.csv"
    if e7_protocol_stability.is_file():
        sources["Table_S11b_E7_parameter_stability_by_protocol.csv"] = e7_protocol_stability
    if e7_heldout_attribution.is_file():
        sources["Table_S12b_E7_heldout_residual_attribution.csv"] = e7_heldout_attribution
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
