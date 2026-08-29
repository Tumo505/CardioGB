from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SYNTHETIC_SEEDS_5 = [str(seed) for seed in range(20260815, 20260820)]
FINAL_REAL_SEEDS_5 = [str(seed) for seed in range(20260825, 20260830)]
SEEDS_10 = [str(seed) for seed in range(20260825, 20260835)]


def stages() -> list[tuple[str, list[str]]]:
    python = sys.executable
    zebrafish = "data/processed/zebrafish_states.npz"
    mouse = "data/processed/mouse_visium_states.npz"
    return [
        ("e1_benchmark", [python, "scripts/run_parallel_multiseed_benchmark.py", "--data", zebrafish, "--output-dir", "results/final_full_multiseed", "--epochs", "200", "--seeds", *SEEDS_10, "--workers", "2", "--patch-batch-size", "8", "--memory-fraction-per-worker", "0.48"]),
        ("e5_e6_synthetic", [python, "scripts/run_synthetic_matrix.py", "--output-dir", "results/synthetic_recovery_full", "--epochs", "300", "--seeds", *SYNTHETIC_SEEDS_5, "--noise", "0", "0.01", "0.05", "0.1"]),
        ("e2_interpolation", [python, "scripts/run_manuscript_training.py", "--data", zebrafish, "--protocol", "e2_interpolation", "--output-dir", "results/e2_interpolation_revised", "--epochs", "200", "--seeds", *FINAL_REAL_SEEDS_5]),
        ("e3_extrapolation", [python, "scripts/run_manuscript_training.py", "--data", zebrafish, "--protocol", "e3_extrapolation", "--output-dir", "results/e3_extrapolation_revised", "--epochs", "200", "--seeds", *FINAL_REAL_SEEDS_5]),
        ("e3_horizon_calibration", [python, "scripts/calibrate_extrapolation_horizon.py", "--data", zebrafish, "--checkpoint-root", "results/e3_extrapolation_revised", "--output-dir", "results/e3_extrapolation_horizon_calibrated"]),
        ("e4_group_cv", [python, "scripts/run_manuscript_training.py", "--data", zebrafish, "--protocol", "e4_group_cv", "--output-dir", "results/e4_group_cv_full", "--epochs", "200", "--seeds", *FINAL_REAL_SEEDS_5]),
        ("e8_ablations", [python, "scripts/run_real_ablations.py", "--data", zebrafish, "--rank-data", "data/processed/zebrafish_states_rank_mean.npz", "--output-dir", "results/final_full_ablations", "--reference-dir", "results/final_full_multiseed", "--epochs", "200", "--seeds", *FINAL_REAL_SEEDS_5, "--max-new-models", "0"]),
        ("e9_ensemble", [python, "scripts/train_ensemble.py", "--data", zebrafish, "--output-dir", "results/final_full_ensemble", "--members", "5", "--epochs", "200", "--seed", "20260825", "--max-new-members", "0"]),
        ("e10_species_adapter", [python, "scripts/run_species_adapter_validation.py", "--mouse", mouse, "--checkpoints", "results/final_full_ensemble/checkpoints", "--output-dir", "results/mouse_species_adapter_revised", "--epochs", "100"]),
        ("e10_conservation", [python, "scripts/validate_mouse_conservation.py", "--zebrafish", zebrafish, "--mouse", mouse, "--output-dir", "results/mouse_validation_revised"]),
        ("added_validation_states", [python, "scripts/build_added_validation_state_datasets.py"]),
        ("added_perturbations", [python, "scripts/analyze_added_perturbations.py"]),
        ("e10_external_prediction", [python, "scripts/evaluate_external_prediction.py", "--zebrafish", zebrafish, "--mouse", mouse, "--checkpoints", "results/final_full_ensemble/checkpoints", "--additional", "GSE106884=data/processed/zebrafish/validation/gse106884_states.npz", "GSE237276=data/processed/zebrafish/validation/gse237276_states.npz", "GSE206787=data/processed/mouse/validation/gse206787_states.npz", "--output-dir", "results/external_predictive_validation_revised"]),
        ("e9_uncertainty_inference", [python, "scripts/analyze_uncertainty_inference.py", "--state-predictions", "results/external_predictive_validation_revised/tables/state_mean_predictions.csv", "--output-dir", "results/external_predictive_validation_revised/tables", "--resamples", "10000"]),
        ("e7_interpretation", [python, "scripts/run_full_interpretation.py", "--data", zebrafish, "--checkpoint-root", "results/final_full_multiseed", "--output-dir", "results/e7_full_interpretation", "--ig-steps", "8", "--k", "8"]),
        ("human_snatac", [python, "scripts/validate_human_snatac.py", "--data", "data/external/human_mi/zenodo_6578047/snatac/snATAC-seq-submission.h5ad", "--output-dir", "results/human_snatac_validation_revised", "--batch-size", "4096"]),
        ("formal_statistics", [python, "scripts/analyze_manuscript_statistics.py", "--results-root", "results", "--output-dir", "results/formal_statistics_revised"]),
        ("main_tables", [python, "scripts/compile_main_manuscript_tables.py", "--data", zebrafish, "--mouse", mouse, "--human", "data/external/human_mi/zenodo_6578047/snatac/snATAC-seq-submission.h5ad", "--results-root", "results", "--output-dir", "manuscript/main_tables", "--resamples", "10000"]),
        ("manuscript_results", [python, "scripts/write_manuscript_results.py", "--results-root", "results", "--manuscript", "manuscript/manuscript_sections.md"]),
        ("figures", [python, "scripts/generate_manuscript_figures.py", "--data", zebrafish, "--results-root", "results", "--output-dir", "figures/manuscript"]),
        ("supplementary_tables", [python, "scripts/compile_supplementary_tables.py", "--data", zebrafish, "--results-root", "results", "--output-dir", "manuscript/supplementary_tables"]),
        ("verification_tests", [python, "scripts/run_final_verification_tests.py", "--output-dir", "results/verification"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable full revised CardioGB manuscript pipeline")
    parser.add_argument("--from-stage")
    parser.add_argument("--through-stage")
    parser.add_argument("--manifest", type=Path, default=Path("results/final_pipeline_manifest.json"))
    args = parser.parse_args()
    matrix = stages()
    names = [name for name, _ in matrix]
    start = names.index(args.from_stage) if args.from_stage else 0
    end = names.index(args.through_stage) + 1 if args.through_stage else len(matrix)
    completed = []
    if args.manifest.is_file():
        completed = json.loads(args.manifest.read_text(encoding="utf-8")).get("completed", [])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    for name, command in matrix[start:end]:
        payload = {"status": "running", "completed": completed, "current_stage": name, "stages": names}
        args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        subprocess.run(command, check=True, env=environment)
        if name not in completed:
            completed.append(name)
        args.manifest.write_text(
            json.dumps({"status": "partial", "completed": completed, "current_stage": None, "stages": names}, indent=2),
            encoding="utf-8",
        )
    final_status = "complete" if set(names).issubset(set(completed)) else "partial"
    args.manifest.write_text(
        json.dumps({"status": final_status, "completed": completed, "current_stage": None, "stages": names}, indent=2),
        encoding="utf-8",
    )
    if final_status == "complete":
        subprocess.run(
            [sys.executable, "scripts/audit_manuscript_completion.py"],
            check=True,
            env=environment,
        )
    print(json.dumps({"status": final_status, "completed": completed}, indent=2))


if __name__ == "__main__":
    main()
