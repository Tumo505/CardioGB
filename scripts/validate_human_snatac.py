from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kruskal, mannwhitneyu

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
    for start in range(0, dataset.n_obs, args.batch_size):
        stop = min(start + args.batch_size, dataset.n_obs)
        block = dataset.X[start:stop, union]
        for pathway, indices in local_selected.items():
            pathway_block = block[:, indices]
            if sparse.issparse(pathway_block):
                values = np.asarray(pathway_block.mean(axis=1)).ravel()
            else:
                values = np.asarray(pathway_block).mean(axis=1)
            scores[pathway][start:stop] = values
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
        statistic, pvalue = kruskal(*groups)
        group_count = len(groups)
        epsilon_squared = max(0.0, float((statistic - group_count + 1) / max(len(patient) - group_count, 1)))
        tests.append({
            "pathway": pathway,
            "test": "Kruskal-Wallis across patient groups",
            "statistic": float(statistic),
            "epsilon_squared": epsilon_squared,
            "p_value": float(pvalue),
            "biological_unit": "patient",
            "n_patients": len(patient),
        })
        for contrast_index, (first_name, second_name) in enumerate(combinations(sorted(grouped_patients), 2)):
            first = grouped_patients[first_name][pathway].to_numpy()
            second = grouped_patients[second_name][pathway].to_numpy()
            test = mannwhitneyu(first, second, alternative="two-sided", method="auto")
            estimate, lower, upper = median_difference_ci(
                first,
                second,
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
                "mann_whitney_u": float(test.statistic),
                "p_value": float(test.pvalue),
                "biological_unit": "patient",
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
        "matrix_mode": "backed; one union-pathway column read per cell batch",
        "pathway_source": str(args.pathways), "pathway_version": pathway_config["version"],
        "categories": categories, "inference_unit": "patient",
        "posthoc": "pairwise Mann-Whitney tests with Cliff delta, median-difference bootstrap intervals, and BH correction",
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