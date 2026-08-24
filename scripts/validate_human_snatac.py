from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kruskal, mannwhitneyu, norm
import statsmodels.api as sm

from cardiogb.utils.config import load_yaml
from cardiogb.utils.io import atomic_json, export_table


def bh_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def cliffs_delta(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    differences = first[:, None] - second[None, :]
    return float((np.count_nonzero(differences > 0) - np.count_nonzero(differences < 0)) / differences.size)


def median_difference_ci(first, second, *, seed, n_resamples=10000):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    estimate = float(np.median(first) - np.median(second))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for iteration in range(n_resamples):
        a = rng.choice(first, size=len(first), replace=True)
        b = rng.choice(second, size=len(second), replace=True)
        draws[iteration] = np.median(a) - np.median(b)
    return estimate, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def clustered_group_model(patient: pd.DataFrame, pathway: str):
    data = patient[["patient", "patient_group", pathway]].dropna().copy()
    data["patient"] = data["patient"].astype(str)
    data["patient_group"] = data["patient_group"].astype(str)
    groups = sorted(data["patient_group"].unique())
    if len(groups) < 2 or data["patient"].nunique() < 3:
        raise ValueError("cluster-robust group inference requires at least two groups and three patients")
    design = pd.get_dummies(data["patient_group"], dtype=float).reindex(columns=groups, fill_value=0.0)
    model = sm.OLS(data[pathway].to_numpy(float), design.to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": data["patient"].to_numpy(), "use_correction": True}
    )
    contrast = np.zeros((len(groups) - 1, len(groups)), dtype=float)
    for row, column in enumerate(range(1, len(groups))):
        contrast[row, 0] = -1.0
        contrast[row, column] = 1.0
    omnibus = model.wald_test(contrast, use_f=False, scalar=True)
    return data, groups, model, omnibus


def clustered_pair_contrast(model, groups, first_name: str, second_name: str):
    contrast = np.zeros(len(groups), dtype=float)
    contrast[groups.index(first_name)] = 1.0
    contrast[groups.index(second_name)] = -1.0
    estimate = float(contrast @ model.params)
    variance = float(contrast @ np.asarray(model.cov_params()) @ contrast)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    statistic = estimate / standard_error if standard_error > 0 else np.nan
    pvalue = float(2.0 * norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan
    return estimate, standard_error, float(statistic), pvalue


def clustered_median_difference_ci(
    patient: pd.DataFrame,
    pathway: str,
    first_name: str,
    second_name: str,
    *,
    seed: int,
    n_resamples: int = 10000,
):
    data = patient[["patient", "patient_group", pathway]].dropna().copy()
    data["patient"] = data["patient"].astype(str)
    data["patient_group"] = data["patient_group"].astype(str)
    first = data.loc[data["patient_group"] == first_name, pathway].to_numpy(float)
    second = data.loc[data["patient_group"] == second_name, pathway].to_numpy(float)
    estimate = float(np.median(first) - np.median(second))
    patient_ids = data["patient"].unique()
    grouped = {name: part for name, part in data.groupby("patient", observed=True)}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        pieces = [grouped[name] for name in sampled]
        bootstrap = pd.concat(pieces, ignore_index=True)
        a = bootstrap.loc[bootstrap["patient_group"] == first_name, pathway].to_numpy(float)
        b = bootstrap.loc[bootstrap["patient_group"] == second_name, pathway].to_numpy(float)
        if len(a) and len(b):
            draws.append(float(np.median(a) - np.median(b)))
    if not draws:
        return estimate, np.nan, np.nan
    return estimate, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def robust_unit_scale(values: np.ndarray, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("pathway values must be a finite one-dimensional array")
    lower, upper = np.quantile(values, [lower_quantile, upper_quantile])
    if upper <= lower:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Backed validation and pathway accessibility analysis of human-MI snATAC H5AD")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pathways", type=Path, default=Path("configs/human_pathways.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    dataset = ad.read_h5ad(args.data, backed="r")
    required = {"patient", "patient_group", "region", "sample", "cell_type_original"}
    missing = required - set(dataset.obs.columns)
    if missing:
        raise ValueError(f"missing required snATAC metadata: {sorted(missing)}")
    pathway_config = load_yaml(args.pathways)
    feature_lookup = {str(name).upper(): index for index, name in enumerate(dataset.var_names)}
    pathway_genes = {
        pathway: sorted(set(details["genes"]))
        for pathway, details in pathway_config["pathways"].items()
    }
    coverage_rows, selected = [], {}
    for pathway, genes in sorted(pathway_genes.items()):
        matched = [gene for gene in genes if gene.upper() in feature_lookup]
        selected[pathway] = [feature_lookup[gene.upper()] for gene in matched]
        coverage_rows.append({
            "pathway": pathway, "ortholog_genes": len(genes), "matched_features": len(matched),
            "coverage_fraction": len(matched) / len(genes) if genes else 0.0,
            "matched_genes": ";".join(matched),
            "unmatched_genes": ";".join(sorted(set(genes) - set(matched))),
        })
    if any(len(indices) == 0 for indices in selected.values()):
        absent = [name for name, indices in selected.items() if not indices]
        raise ValueError(f"no snATAC features for pathways: {absent}")
    scores = {name: np.empty(dataset.n_obs, dtype=np.float32) for name in selected}
    union = sorted(set(index for indices in selected.values() for index in indices))
    union_position = {index: position for position, index in enumerate(union)}
    local_selected = {
        pathway: [union_position[index] for index in indices]
        for pathway, indices in selected.items()
    }
    feature_sum = np.zeros(len(union), dtype=np.float64)
    feature_square_sum = np.zeros(len(union), dtype=np.float64)
    for start in range(0, dataset.n_obs, args.batch_size):
        stop = min(start + args.batch_size, dataset.n_obs)
        block = dataset.X[start:stop, union]
        dense = block.toarray() if sparse.issparse(block) else np.asarray(block)
        feature_sum += dense.sum(axis=0, dtype=np.float64)
        feature_square_sum += np.square(dense, dtype=np.float64).sum(axis=0)
    feature_mean = feature_sum / dataset.n_obs
    feature_variance = np.maximum(feature_square_sum / dataset.n_obs - np.square(feature_mean), 0.0)
    feature_std = np.sqrt(feature_variance)
    feature_std[feature_std <= np.finfo(np.float64).eps] = 1.0
    for start in range(0, dataset.n_obs, args.batch_size):
        stop = min(start + args.batch_size, dataset.n_obs)
        block = dataset.X[start:stop, union]
        dense = block.toarray() if sparse.issparse(block) else np.asarray(block)
        standardized = (dense - feature_mean) / feature_std
        for pathway, indices in local_selected.items():
            scores[pathway][start:stop] = standardized[:, indices].mean(axis=1)
    for pathway in scores:
        scores[pathway] = robust_unit_scale(scores[pathway])
    frame = dataset.obs[["patient", "patient_group", "region", "sample", "cell_type_original"]].reset_index(drop=False)
    for pathway, values in scores.items():
        frame[pathway] = values
    export_table(pd.DataFrame(coverage_rows), args.output_dir / "tables" / "pathway_feature_coverage.csv")
    export_table(frame, args.output_dir / "tables" / "cell_pathway_accessibility.parquet")
    patient = frame.groupby(["patient", "patient_group"], observed=True)[list(scores)].mean().reset_index()
    export_table(patient, args.output_dir / "tables" / "patient_pathway_accessibility.csv")
    region = frame.groupby(["patient", "patient_group", "region"], observed=True)[list(scores)].mean().reset_index()
    export_table(region, args.output_dir / "tables" / "patient_region_pathway_accessibility.csv")
    cell_type = frame.groupby(["patient", "patient_group", "cell_type_original"], observed=True)[list(scores)].mean().reset_index()
    export_table(cell_type, args.output_dir / "tables" / "patient_celltype_pathway_accessibility.csv")
    tests, posthoc = [], []
    grouped_patients = {str(name): part for name, part in patient.groupby("patient_group", observed=True)}
    for pathway_index, pathway in enumerate(scores):
        groups = [part[pathway].to_numpy() for part in grouped_patients.values() if len(part)]
        statistic, descriptive_pvalue = kruskal(*groups)
        group_count = len(groups)
        epsilon_squared = max(0.0, float((statistic - group_count + 1) / max(len(patient) - group_count, 1)))
        model_data, model_groups, cluster_model, omnibus = clustered_group_model(patient, pathway)
        tests.append({
            "pathway": pathway,
            "test": "patient-cluster-robust OLS omnibus Wald test",
            "statistic": float(omnibus.statistic),
            "degrees_of_freedom": group_count - 1,
            "p_value": float(omnibus.pvalue),
            "descriptive_kruskal_statistic": float(statistic),
            "descriptive_kruskal_p_value": float(descriptive_pvalue),
            "descriptive_epsilon_squared": epsilon_squared,
            "biological_unit": "patient cluster",
            "n_patients": int(model_data["patient"].nunique()),
            "n_patient_group_rows": len(model_data),
        })
        for contrast_index, (first_name, second_name) in enumerate(combinations(sorted(grouped_patients), 2)):
            first = grouped_patients[first_name][pathway].to_numpy()
            second = grouped_patients[second_name][pathway].to_numpy()
            descriptive_test = mannwhitneyu(first, second, alternative="two-sided", method="auto")
            adjusted_difference, cluster_se, cluster_z, cluster_p = clustered_pair_contrast(
                cluster_model, model_groups, first_name, second_name
            )
            estimate, lower, upper = clustered_median_difference_ci(
                patient,
                pathway,
                first_name,
                second_name,
                seed=20260815 + pathway_index * 100 + contrast_index,
            )
            posthoc.append({
                "pathway": pathway,
                "group_1": first_name,
                "group_2": second_name,
                "contrast": f"{first_name} vs {second_name}",
                "n_group_1": len(first),
                "n_group_2": len(second),
                "median_group_1": float(np.median(first)),
                "median_group_2": float(np.median(second)),
                "median_difference_group_1_minus_group_2": estimate,
                "median_difference_ci_lower": lower,
                "median_difference_ci_upper": upper,
                "cliffs_delta_group_1_minus_group_2": cliffs_delta(first, second),
                "cluster_robust_mean_difference_group_1_minus_group_2": adjusted_difference,
                "cluster_robust_standard_error": cluster_se,
                "cluster_robust_z": cluster_z,
                "p_value": cluster_p,
                "mann_whitney_u_descriptive": float(descriptive_test.statistic),
                "mann_whitney_p_value_descriptive": float(descriptive_test.pvalue),
                "biological_unit": "patient cluster",
            })
    test_frame = pd.DataFrame(tests)
    test_frame["p_adjust_bh"] = bh_adjust(test_frame["p_value"])
    export_table(test_frame, args.output_dir / "tables" / "patient_group_tests.csv")
    posthoc_frame = pd.DataFrame(posthoc)
    posthoc_frame["p_adjust_bh_global"] = bh_adjust(posthoc_frame["p_value"])
    posthoc_frame["p_adjust_bh_within_pathway"] = posthoc_frame.groupby(
        "pathway", observed=True
    )["p_value"].transform(lambda values: bh_adjust(values.to_numpy()))
    export_table(posthoc_frame, args.output_dir / "tables" / "patient_group_posthoc_effects.csv")
    categories = {}
    for column in sorted(required):
        categories[column] = {str(k): int(v) for k, v in dataset.obs[column].value_counts(dropna=False).items()}
    manifest = {
        "status": "complete", "path": str(args.data), "shape": [int(dataset.n_obs), int(dataset.n_vars)],
        "obs_names_unique": bool(dataset.obs_names.is_unique), "var_names_unique": bool(dataset.var_names.is_unique),
        "matrix_mode": "backed two-pass scoring; only union-pathway columns materialized per cell batch",
        "input_normalization": str(dataset.uns.get("X_normalization", "unspecified")),
        "pathway_scoring": "submitted normalized gene activity; per-feature standardization; within-pathway mean; 1st/99th percentile clipping and [0,1] mapping",
        "pathway_source": str(args.pathways), "pathway_version": pathway_config["version"],
        "categories": categories, "inference_unit": "patient cluster; repeated patient-group regions retained within cluster",
        "omnibus": "OLS group means with patient-cluster-robust covariance and Wald test; Kruskal-Wallis retained descriptively",
        "posthoc": "patient-cluster-robust group contrasts with Cliff delta, whole-patient bootstrap median-difference intervals, and BH correction; Mann-Whitney retained descriptively",
        "limitations": [
            "X is interpreted as the submitted feature matrix; no peak-to-gene causality is inferred",
            "pathway means are regulatory-feature summaries and are not interchangeable with RNA expression scores",
            "human-MI analysis is translational and does not validate zebrafish dynamical prediction",
        ],
    }
    atomic_json(manifest, args.output_dir / "validation_manifest.json")
    dataset.file.close()
    print(json.dumps({"status": "complete", "shape": manifest["shape"], "pathways": len(scores)}, indent=2))


if __name__ == "__main__":
    main()