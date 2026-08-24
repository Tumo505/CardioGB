from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cardiogb.data.state_dataset import StateDataset
from cardiogb.utils.io import atomic_json, export_table


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def finite_csv(path: Path, required_columns: tuple[str, ...] = ()) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame]
    if missing:
        return False, f"missing columns {missing}"
    numeric = frame[list(required_columns)].apply(pd.to_numeric, errors="coerce") if required_columns else frame.select_dtypes(include=[np.number])
    if len(numeric.columns) and not np.isfinite(numeric.to_numpy()).all():
        return False, "contains non-finite required numeric values"
    return True, f"{len(frame)} rows"


def main() -> None:
    parser = argparse.ArgumentParser(description="Requirement-by-requirement CardioGB completion audit")
    parser.add_argument("--data", type=Path, default=Path("data/processed/zebrafish_states.npz"))
    parser.add_argument("--mouse", type=Path, default=Path("data/processed/mouse_visium_states.npz"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--manuscript", type=Path, default=Path("manuscript/manuscript_sections.md"))
    parser.add_argument("--figures", type=Path, default=Path("figures/manuscript"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/completion_audit"))
    args = parser.parse_args()
    root = args.results_root
    checks: list[dict] = []

    def add(category: str, requirement: str, passed: bool, evidence: str, detail: str) -> None:
        checks.append({"category": category, "requirement": requirement, "status": "complete" if passed else "incomplete", "evidence": evidence, "detail": detail})

    try:
        fish = StateDataset.load(args.data)
        data_ok = len(fish.state_names) == 6 and len(fish.states) == len(fish.groups) == len(fish.sections) == len(fish.coordinates)
        add("Data", "Parsed zebrafish spatial dataset with validated biological units and six states", data_ok, str(args.data), f"spots={len(fish.states)}, units={pd.Series(fish.groups).nunique()}, sections={pd.Series(fish.sections).nunique()}, states={len(fish.state_names)}")
    except Exception as error:
        add("Data", "Parsed zebrafish spatial dataset with validated biological units and six states", False, str(args.data), repr(error))
    try:
        mouse = StateDataset.load(args.mouse)
        mouse_ok = len(mouse.state_names) == 6 and len(mouse.states) == len(mouse.groups)
        add("Data", "Processed mouse spatial external dataset", mouse_ok, str(args.mouse), f"spots={len(mouse.states)}, stages={pd.Series(mouse.times).nunique()}")
    except Exception as error:
        add("Data", "Processed mouse spatial external dataset", False, str(args.mouse), repr(error))
    graph_path = Path("data/processed/zebrafish_graph_k8.npz")
    add("Data", "Per-section spatial graphs generated", graph_path.is_file() and graph_path.stat().st_size > 0, str(graph_path), f"bytes={graph_path.stat().st_size if graph_path.is_file() else 0}")

    e1_manifest = read_json(root / "final_full_multiseed" / "run_manifest.json")
    e1_models = set(e1_manifest.get("models", []))
    expected_models = {"persistence", "mechanistic_ode", "neural_ode", "graph_neural_ode", "cardiogb"}
    add("Models", "All five prespecified model classes executed", e1_manifest.get("status") == "complete" and e1_models == expected_models and len(e1_manifest.get("completed", [])) == 50, str(root / "final_full_multiseed" / "run_manifest.json"), f"status={e1_manifest.get('status')}, models={sorted(e1_models)}, completed={len(e1_manifest.get('completed', []))}")
    ok, detail = finite_csv(root / "final_full_multiseed" / "tables" / "benchmark_metrics.csv", ("mmd", "moment_error", "sliced_wasserstein"))
    add("Experiments", "E1 main benchmark: ten seeds, five models, finite metrics", ok and len(e1_manifest.get("completed", [])) == 50, str(root / "final_full_multiseed" / "tables" / "benchmark_metrics.csv"), detail)

    protocols = [
        ("E2 temporal interpolation", "e2_interpolation_revised", 15),
        ("E3 temporal extrapolation", "e3_extrapolation_revised", 15),
        ("E4 grouped biological-unit generalization", "e4_group_cv_full", 15),
    ]
    for label, directory, expected in protocols:
        manifest = read_json(root / directory / "run_manifest.json")
        ok, detail = finite_csv(root / directory / "tables" / "all_metrics.csv", ("mmd", "moment_error", "sliced_wasserstein"))
        add("Experiments", label, manifest.get("status") == "complete" and len(manifest.get("completed", [])) == expected and ok, str(root / directory / "run_manifest.json"), f"status={manifest.get('status')}, fits={len(manifest.get('completed', []))}; {detail}")
    calibration = read_json(root / "e3_extrapolation_horizon_calibrated" / "run_manifest.json")
    ok, detail = finite_csv(root / "e3_extrapolation_horizon_calibrated" / "tables" / "all_metrics.csv", ("mmd", "raw_mmd", "persistence_mmd"))
    add("Experiments", "Validation-only E3 horizon calibration", calibration.get("status") == "complete" and calibration.get("test_outcomes_used_for_calibration") is False and ok, str(root / "e3_extrapolation_horizon_calibrated" / "run_manifest.json"), f"status={calibration.get('status')}, no_test_fit={calibration.get('test_outcomes_used_for_calibration')}; {detail}")

    synthetic = read_json(root / "synthetic_recovery_full" / "run_manifest.json")
    e5_ok, e5_detail = finite_csv(root / "synthetic_recovery_full" / "tables" / "e5_parameter_recovery.csv", ("parameter_correlation", "parameter_rmse"))
    e6_ok, e6_detail = finite_csv(root / "synthetic_recovery_full" / "tables" / "e6_hidden_recovery.csv", ("correlation", "rmse"))
    synthetic_matrix = set(float(x) for x in synthetic.get("noise_std", [])) == {0.0, 0.01, 0.05, 0.1} and len(synthetic.get("seeds", [])) == 5
    add("Experiments", "E5 parameter recovery across registered noise levels and seeds", synthetic.get("status") == "complete" and synthetic_matrix and e5_ok, str(root / "synthetic_recovery_full" / "tables" / "e5_parameter_recovery.csv"), e5_detail)
    add("Experiments", "E6 hidden-mechanism recovery across registered noise levels and seeds", synthetic.get("status") == "complete" and synthetic_matrix and e6_ok, str(root / "synthetic_recovery_full" / "tables" / "e6_hidden_recovery.csv"), e6_detail)

    e8 = read_json(root / "final_full_ablations" / "run_manifest.json")
    e8_metrics_path = root / "final_full_ablations" / "tables" / "ablation_metrics.csv"
    e8_ok, e8_detail = finite_csv(e8_metrics_path, ("mmd", "moment_error", "sliced_wasserstein"))
    e8_frame = pd.read_csv(e8_metrics_path) if e8_metrics_path.is_file() else pd.DataFrame()
    diagnostic_rows = e8_frame[e8_frame.get("ablation", pd.Series(dtype=str)) != "full"] if len(e8_frame) else e8_frame
    diagnostics_complete = bool(len(diagnostic_rows)) and diagnostic_rows[["prediction_finite_fraction", "prediction_out_of_bounds_fraction"]].notna().all().all()
    no_mechanism_markers = [read_json(path) for path in sorted((root / "final_full_ablations" / "no_mechanism").glob("seed_*/done.json"))]
    fair_no_mechanism = len(no_mechanism_markers) == 5 and all(item.get("mechanistic_component") is False and item.get("state_projection") is True for item in no_mechanism_markers)
    add("Experiments", "E8 full nine-condition ablation matrix with fair no-mechanism condition", e8.get("status") == "complete" and len(e8.get("completed", [])) == 45 and e8_ok and diagnostics_complete and fair_no_mechanism, str(root / "final_full_ablations" / "run_manifest.json"), f"status={e8.get('status')}, fits={len(e8.get('completed', []))}, diagnostics={diagnostics_complete}, fair_no_mechanism={fair_no_mechanism}; {e8_detail}")

    ensemble = read_json(root / "final_full_ensemble" / "run_manifest.json")
    add("Experiments", "E9 five-member validation-selected ensemble", ensemble.get("status") == "complete" and ensemble.get("members") == 5 and len(ensemble.get("ensemble_weights", [])) == 5, str(root / "final_full_ensemble" / "run_manifest.json"), f"status={ensemble.get('status')}, members={ensemble.get('members')}, aggregation={ensemble.get('selected_aggregation')}")
    uncertainty = read_json(root / "external_predictive_validation_revised" / "tables" / "uncertainty_inference_manifest.json")
    uncertainty_ok, uncertainty_detail = finite_csv(root / "external_predictive_validation_revised" / "tables" / "uncertainty_inferential_tests.csv", ("spearman", "p_value", "p_adjust_bh"))
    add("Experiments", "Uncertainty-versus-error and uncertainty-versus-horizon inference", uncertainty.get("status") == "complete" and uncertainty_ok, str(root / "external_predictive_validation_revised" / "tables" / "uncertainty_inferential_tests.csv"), uncertainty_detail)

    conservation = read_json(root / "mouse_validation_revised" / "validation_manifest.json")
    external = read_json(root / "external_predictive_validation_revised" / "run_manifest.json")
    external_ok, external_detail = finite_csv(root / "external_predictive_validation_revised" / "metrics" / "external_prediction.csv", ("mmd", "moment_error", "sliced_wasserstein", "persistence_sliced_wasserstein"))
    add("External validation", "E10 mouse pathway conservation", conservation.get("status") == "complete", str(root / "mouse_validation_revised" / "validation_manifest.json"), f"status={conservation.get('status')}")
    add("External validation", "E10 frozen zero-shot mouse predictive validation", external.get("status") == "complete" and external.get("members") == 5 and external.get("mouse_retraining") is False and external_ok, str(root / "external_predictive_validation_revised" / "run_manifest.json"), f"status={external.get('status')}, members={external.get('members')}, retraining={external.get('mouse_retraining')}; {external_detail}")

    e7 = read_json(root / "e7_full_interpretation" / "run_manifest.json")
    required_e7 = ["residual_attribution_heldout.csv", "mi_stage_bootstrap.csv", "mi_domain_biological_units.csv", "parameter_stability.csv", "parameter_identifiability_diagnostics.csv", "parameter_local_sensitivity.csv"]
    missing_e7 = [name for name in required_e7 if not (root / "e7_full_interpretation" / "tables" / name).is_file()]
    add("Interpretation", "E7 held-out insufficiency, residual attribution, stability, and identifiability", e7.get("status") == "complete" and e7.get("members") == 10 and e7.get("fold_parameter_fits") == 15 and not missing_e7, str(root / "e7_full_interpretation" / "run_manifest.json"), f"status={e7.get('status')}, members={e7.get('members')}, fold_fits={e7.get('fold_parameter_fits')}, missing={missing_e7}")

    human = read_json(root / "human_snatac_validation_revised" / "validation_manifest.json")
    human_ok, human_detail = finite_csv(root / "human_snatac_validation_revised" / "tables" / "patient_group_posthoc_effects.csv", ("cliffs_delta_group_1_minus_group_2", "p_adjust_bh_global"))
    add("External validation", "E11 patient-level human snATAC-seq translation with effect sizes and multiplicity correction", human.get("status") == "complete" and human_ok, str(root / "human_snatac_validation_revised" / "validation_manifest.json"), f"status={human.get('status')}; {human_detail}")

    statistics = read_json(root / "formal_statistics_revised" / "statistics_manifest.json")
    add("Statistics", "Formal biological-unit confidence intervals and multiplicity-corrected comparisons", statistics.get("status") == "complete", str(root / "formal_statistics_revised" / "statistics_manifest.json"), f"status={statistics.get('status')}, tables={statistics.get('tables')}")

    figure_missing = []
    for index in range(1, 9):
        matches = list(args.figures.glob(f"Figure_{index}_*.png")) + list(args.figures.glob(f"Figure_{index}_*.pdf"))
        if len(matches) != 2 or any(path.stat().st_size == 0 for path in matches):
            figure_missing.append(index)
    add("Outputs", "Registered manuscript Figures 1–8 in PNG and PDF", not figure_missing, str(args.figures), f"missing_or_invalid={figure_missing}")
    main_table_dir = Path("manuscript/main_tables")
    missing_main = [index for index in range(1, 7) if not (main_table_dir / f"Table_{index}.csv").is_file()]
    add("Outputs", "Main manuscript Tables 1–6", not missing_main and (main_table_dir / "main_tables.md").is_file(), str(main_table_dir), f"missing={missing_main}")
    supplement_dir = Path("manuscript/supplementary_tables")
    supplement_files = list(supplement_dir.glob("*.csv")) if supplement_dir.is_dir() else []
    add("Outputs", "Complete supplementary table export", len(supplement_files) >= 39, str(supplement_dir), f"csv_tables={len(supplement_files)}")

    manuscript_text = args.manuscript.read_text(encoding="utf-8") if args.manuscript.is_file() else ""
    headings = ["## Introduction", "## Materials and methods", "## Results", "## Discussion", "### Limitations", "## Software description and reproducibility", "## Data and code availability", "## Figure legends", "## References"]
    missing_headings = [heading for heading in headings if heading not in manuscript_text]
    add("Outputs", "Complete manuscript sections and figure legends", not missing_headings and "AUTO_RESULTS_START" in manuscript_text and "AUTO_RESULTS_END" in manuscript_text, str(args.manuscript), f"missing_headings={missing_headings}, characters={len(manuscript_text)}")

    pipeline = read_json(root / "final_pipeline_manifest.json")
    add("Reproducibility", "Complete registered pipeline manifest", pipeline.get("status") == "complete", str(root / "final_pipeline_manifest.json"), f"status={pipeline.get('status')}, completed={len(pipeline.get('completed', []))}")
    test_manifest = read_json(root / "verification" / "test_manifest.json")
    add("Reproducibility", "Full automated test suite passes after final artifact generation", test_manifest.get("status") == "complete" and int(test_manifest.get("passed", 0)) >= 54 and int(test_manifest.get("failed", 1)) == 0, str(root / "verification" / "test_manifest.json"), f"status={test_manifest.get('status')}, passed={test_manifest.get('passed')}, failed={test_manifest.get('failed')}")

    frame = pd.DataFrame(checks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_table(frame, args.output_dir / "completion_audit.csv")
    complete = bool((frame["status"] == "complete").all())
    summary = {"status": "complete" if complete else "incomplete", "requirements": len(frame), "complete_requirements": int((frame["status"] == "complete").sum()), "incomplete_requirements": int((frame["status"] != "complete").sum())}
    atomic_json(summary, args.output_dir / "completion_audit.json")
    lines = ["# CardioGB completion audit", "", f"Overall status: **{summary['status']}** ({summary['complete_requirements']}/{summary['requirements']} requirements complete).", ""]
    for category, subset in frame.groupby("category", sort=False):
        lines.append(f"## {category}\n")
        for row in subset.itertuples(index=False):
            symbol = "x" if row.status == "complete" else " "
            lines.append(f"- [{symbol}] {row.requirement} — {row.detail} (`{row.evidence}`)")
        lines.append("")
    (args.output_dir / "completion_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
