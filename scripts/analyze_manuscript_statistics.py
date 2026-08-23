from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cardiogb.statistics import benjamini_hochberg, paired_group_permutation_test
from cardiogb.utils.io import atomic_json, export_table

METRICS = ("mmd", "moment_error", "sliced_wasserstein")


def bootstrap_difference(values, seed=20260815, n=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_model_tests(frame, label_column, reference, unit_column, family):
    reduced = frame.groupby([unit_column, label_column], observed=True)[list(METRICS)].mean().reset_index()
    rows = []
    labels = sorted(set(reduced[label_column]) - {reference})
    for metric in METRICS:
        pivot = reduced.pivot(index=unit_column, columns=label_column, values=metric)
        for label in labels:
            if reference not in pivot or label not in pivot:
                continue
            paired = pivot[[reference, label]].dropna()
            if len(paired) < 2:
                continue
            difference = paired[reference].to_numpy() - paired[label].to_numpy()
            estimate, lower, upper = bootstrap_difference(difference)
            test = paired_group_permutation_test(paired[reference].to_numpy(), paired[label].to_numpy(), seed=20260815)
            standard_deviation = difference.std(ddof=1)
            rows.append({
                "family": family, "metric": metric, "reference": reference, "comparator": label,
                "mean_difference_reference_minus_comparator": estimate,
                "median_difference_reference_minus_comparator": float(np.median(difference)),
                "paired_standardized_effect_dz": float(estimate / standard_deviation) if standard_deviation > 0 else np.nan,
                "reference_win_fraction_lower_error": float(np.mean(difference < 0)),
                "ci_lower": lower, "ci_upper": upper, "p_value": test["p_value"],
                "n_units": len(paired), "unit": unit_column,
            })
    result = pd.DataFrame(rows)
    if len(result):
        result["p_adjust_bh_within_family"] = benjamini_hochberg(result["p_value"].to_numpy())
    return result


def protocol_ci(path: Path, protocol: str):
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    rows = []
    group_columns = ["case", "transition"]
    for keys, subset in frame.groupby(group_columns, observed=True):
        for metric in METRICS:
            seed_values = subset.groupby("seed", observed=True)[metric].mean().to_numpy()
            estimate, lower, upper = bootstrap_difference(seed_values)
            rows.append({
                "protocol": protocol, "case": keys[0], "transition": keys[1], "metric": metric,
                "estimate": estimate, "ci_lower": lower, "ci_upper": upper,
                "n_seeds": len(seed_values), "resampling_unit": "initialization seed",
            })
    return pd.DataFrame(rows)


def e4_biological_ci(path: Path):
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    rows = []
    for metric in METRICS:
        fold = frame.groupby(["case", "seed"], observed=True)[metric].mean().groupby("case").mean().to_numpy()
        estimate, lower, upper = bootstrap_difference(fold)
        rows.append({
            "protocol": "E4", "metric": metric, "estimate": estimate, "ci_lower": lower,
            "ci_upper": upper, "n_biological_folds": len(fold),
            "resampling_unit": "held-out biological replicate; seeds averaged within fold",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Formal manuscript statistics and multiple-testing correction")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root
    outputs, missing = {}, []
    benchmark_path = root / "final_full_multiseed" / "tables" / "benchmark_metrics.csv"
    if benchmark_path.is_file():
        result = paired_model_tests(pd.read_csv(benchmark_path), "model", "cardiogb", "seed", "E1 model comparisons")
        export_table(result, args.output_dir / "e1_paired_tests.csv")
        outputs["e1_tests"] = len(result)
    else:
        missing.append(str(benchmark_path))
    ablation_path = root / "final_full_ablations" / "tables" / "ablation_metrics.csv"
    if ablation_path.is_file():
        result = paired_model_tests(pd.read_csv(ablation_path), "ablation", "full", "seed", "E8 ablations")
        export_table(result, args.output_dir / "e8_paired_tests.csv")
        outputs["e8_tests"] = len(result)
    else:
        missing.append(str(ablation_path))
    for protocol, directory in (("E2", "e2_interpolation_revised"), ("E3", "e3_extrapolation_revised")):
        path = root / directory / "tables" / "all_metrics.csv"
        result = protocol_ci(path, protocol)
        if len(result):
            export_table(result, args.output_dir / f"{protocol.lower()}_confidence_intervals.csv")
            outputs[f"{protocol.lower()}_intervals"] = len(result)
        else:
            missing.append(str(path))
    e4_path = root / "e4_group_cv_full" / "tables" / "all_metrics.csv"
    result = e4_biological_ci(e4_path)
    if len(result):
        export_table(result, args.output_dir / "e4_biological_unit_confidence_intervals.csv")
        outputs["e4_intervals"] = len(result)
    else:
        missing.append(str(e4_path))
    external_path = root / "external_predictive_validation_revised" / "metrics" / "external_prediction.csv"
    if external_path.is_file():
        external = pd.read_csv(external_path)
        mouse = external[external["protocol"].str.startswith("mouse")].copy()
        for metric in METRICS:
            mouse[f"delta_vs_persistence_{metric}"] = mouse[metric] - mouse[f"persistence_{metric}"]
            mouse[f"delta_vs_equal_{metric}"] = mouse[metric] - mouse[f"equal_{metric}"]
        export_table(mouse, args.output_dir / "external_descriptive_comparisons.csv")
        outputs["external_transitions"] = len(mouse)
    else:
        missing.append(str(external_path))
    atomic_json(
        {
            "status": "complete" if not missing else "partial",
            "outputs": outputs, "missing": missing,
            "confidence_intervals": "percentile bootstrap, 10000 resamples",
            "multiple_testing": "Benjamini-Hochberg within prespecified E1 and E8 test families",
            "external_inference": "descriptive only because mouse has one biological sample per stage",
        },
        args.output_dir / "statistics_manifest.json",
    )
    print(json.dumps({"outputs": outputs, "missing": missing}, indent=2))


if __name__ == "__main__":
    main()