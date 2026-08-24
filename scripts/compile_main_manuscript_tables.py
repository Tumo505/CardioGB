from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cardiogb.data.state_dataset import StateDataset
from cardiogb.utils.io import export_table


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required main-table input is missing: {path}")
    return path


def bootstrap_mean(values: np.ndarray, seed: int, resamples: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def dataset_row(name: str, species: str, modality: str, dataset: StateDataset, role: str) -> dict:
    return {
        "dataset": name,
        "species": species,
        "modality": modality,
        "observations": int(len(dataset.states)),
        "biological_units": int(pd.Series(dataset.groups).nunique()),
        "sections_or_samples": int(pd.Series(dataset.sections).nunique()),
        "timepoints_or_groups": int(pd.Series(dataset.times).nunique()),
        "model_states": int(len(dataset.state_names)),
        "analysis_role": role,
    }


def table_1(data: Path, mouse: Path, human: Path | None) -> pd.DataFrame:
    zebrafish = StateDataset.load(data)
    murine = StateDataset.load(mouse)
    rows = [
        dataset_row("Zebrafish regeneration atlas", "Danio rerio", "Stereo-seq", zebrafish, "training and internal validation"),
        dataset_row("Neonatal mouse repair series", "Mus musculus", "spatial transcriptomics", murine, "frozen external validation"),
    ]
    if human is not None and human.is_file():
        import anndata as ad

        backed = ad.read_h5ad(human, backed="r")
        try:
            obs = backed.obs
            rows.append(
                {
                    "dataset": "Human myocardial infarction",
                    "species": "Homo sapiens",
                    "modality": "snATAC-seq",
                    "observations": int(backed.n_obs),
                    "biological_units": int(obs["patient"].nunique()),
                    "sections_or_samples": int(obs["sample"].nunique()),
                    "timepoints_or_groups": int(obs["patient_group"].nunique()),
                    "model_states": 6,
                    "analysis_role": "patient-level translational validation",
                }
            )
        finally:
            backed.file.close()
    return pd.DataFrame(rows)


def table_2(root: Path, resamples: int) -> pd.DataFrame:
    data = pd.read_csv(require(root / "final_full_multiseed" / "tables" / "benchmark_metrics.csv"))
    seed = data.groupby(["model", "seed"], observed=True)[["mmd", "sliced_wasserstein", "moment_error"]].mean().reset_index()
    rows = []
    for model, subset in seed.groupby("model", observed=True):
        row = {"model": model, "n_seeds": int(subset["seed"].nunique())}
        for index, metric in enumerate(("mmd", "sliced_wasserstein", "moment_error")):
            mean, lower, upper = bootstrap_mean(subset[metric].to_numpy(), 20260825 + index, resamples)
            row.update({f"{metric}_mean": mean, f"{metric}_ci_lower": lower, f"{metric}_ci_upper": upper})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sliced_wasserstein_mean")


def table_3(root: Path, resamples: int) -> pd.DataFrame:
    e2 = pd.read_csv(require(root / "e2_interpolation_revised" / "tables" / "all_metrics.csv"))
    e3 = pd.read_csv(require(root / "e3_extrapolation_horizon_calibrated" / "tables" / "all_metrics.csv"))
    rows = []
    for case, subset in e2.groupby("case", observed=True):
        for index, metric in enumerate(("mmd", "sliced_wasserstein", "moment_error")):
            unit = subset.groupby("seed", observed=True)[metric].mean().to_numpy()
            mean, lower, upper = bootstrap_mean(unit, 20260900 + len(rows) + index, resamples)
            rows.append({"experiment": "E2 interpolation", "case": str(case), "method": "CardioGB", "metric": metric, "mean": mean, "ci_lower": lower, "ci_upper": upper, "n_seeds": len(unit)})
    method_columns = {
        "calibrated CardioGB": {"mmd": "mmd", "sliced_wasserstein": "sliced_wasserstein", "moment_error": "moment_error"},
        "raw CardioGB": {"mmd": "raw_mmd", "sliced_wasserstein": "raw_sliced_wasserstein", "moment_error": "raw_moment_error"},
        "persistence": {"mmd": "persistence_mmd", "sliced_wasserstein": "persistence_sliced_wasserstein", "moment_error": "persistence_moment_error"},
    }
    for horizon, subset in e3.groupby("horizon_days", observed=True):
        for method, columns in method_columns.items():
            for metric, column in columns.items():
                unit = subset.groupby("seed", observed=True)[column].mean().to_numpy()
                mean, lower, upper = bootstrap_mean(unit, 20261000 + len(rows), resamples)
                rows.append({"experiment": "E3 extrapolation", "case": f"horizon_{horizon:g}_days", "method": method, "metric": metric, "mean": mean, "ci_lower": lower, "ci_upper": upper, "n_seeds": len(unit)})
    return pd.DataFrame(rows)


def table_4(root: Path) -> pd.DataFrame:
    stability = pd.read_csv(require(root / "e7_full_interpretation" / "tables" / "parameter_stability.csv"))
    columns = ["parameter", "mean", "ci_lower", "ci_upper", "std", "coefficient_of_variation", "sign_consistency", "n_e1_fits", "n_e4_fits"]
    return stability[columns].sort_values("coefficient_of_variation")


def table_5(root: Path, resamples: int) -> pd.DataFrame:
    data = pd.read_csv(require(root / "final_full_ablations" / "tables" / "ablation_metrics.csv"))
    seed = data.groupby(["ablation", "seed"], observed=True)[["mmd", "sliced_wasserstein", "moment_error", "prediction_out_of_bounds_fraction"]].mean().reset_index()
    rows = []
    for ablation, subset in seed.groupby("ablation", observed=True):
        row = {"ablation": ablation, "n_seeds": int(subset["seed"].nunique())}
        for metric in ("mmd", "sliced_wasserstein", "moment_error", "prediction_out_of_bounds_fraction"):
            mean, lower, upper = bootstrap_mean(subset[metric].to_numpy(), 20261100 + len(rows), resamples)
            row.update({f"{metric}_mean": mean, f"{metric}_ci_lower": lower, f"{metric}_ci_upper": upper})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sliced_wasserstein_mean")


def table_6(root: Path, top_n: int) -> pd.DataFrame:
    data = pd.read_csv(require(root / "e7_full_interpretation" / "tables" / "residual_attribution_heldout.csv"))
    unit = data.groupby(["biological_unit", "target_state", "input_state"], observed=True)["mean_absolute_integrated_gradient"].mean().reset_index()
    rows = []
    for (target, source), subset in unit.groupby(["target_state", "input_state"], observed=True):
        values = subset["mean_absolute_integrated_gradient"].to_numpy(float)
        mean, lower, upper = bootstrap_mean(values, 20261200 + len(rows), 10000)
        rows.append({"input_pathway": source, "residual_target": target, "mean_absolute_attribution": mean, "ci_lower": lower, "ci_upper": upper, "n_biological_units": int(subset["biological_unit"].nunique()), "claim_scope": "model-derived association; hypothesis generating"})
    return pd.DataFrame(rows).sort_values("mean_absolute_attribution", ascending=False).head(top_n)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile CardioGB main manuscript Tables 1–6")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mouse", type=Path, required=True)
    parser.add_argument("--human", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("manuscript/main_tables"))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--top-residuals", type=int, default=20)
    args = parser.parse_args()
    tables = [
        ("Table 1 | Dataset summary", table_1(args.data, args.mouse, args.human)),
        ("Table 2 | Main predictive benchmark", table_2(args.results_root, args.resamples)),
        ("Table 3 | Interpolation and extrapolation performance", table_3(args.results_root, args.resamples)),
        ("Table 4 | Mechanistic parameter estimates and confidence intervals", table_4(args.results_root)),
        ("Table 5 | Ablation results", table_5(args.results_root, args.resamples)),
        ("Table 6 | Top inferred residual pathway relationships", table_6(args.results_root, args.top_residuals)),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sections = ["# CardioGB main manuscript tables"]
    for index, (title, frame) in enumerate(tables, start=1):
        export_table(frame, args.output_dir / f"Table_{index}.csv")
        sections.extend([f"\n## {title}\n", markdown_table(frame)])
    (args.output_dir / "main_tables.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"compiled {len(tables)} main tables in {args.output_dir}")


if __name__ == "__main__":
    main()
