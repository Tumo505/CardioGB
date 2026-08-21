from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kruskal

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
    tests = []
    for pathway in scores:
        groups = [part[pathway].to_numpy() for _, part in patient.groupby("patient_group", observed=True) if len(part)]
        statistic, pvalue = kruskal(*groups)
        tests.append({"pathway": pathway, "test": "Kruskal-Wallis across patient groups", "statistic": float(statistic), "p_value": float(pvalue), "biological_unit": "patient", "n_patients": len(patient)})
    test_frame = pd.DataFrame(tests)
    test_frame["p_adjust_bh"] = bh_adjust(test_frame["p_value"])
    export_table(test_frame, args.output_dir / "tables" / "patient_group_tests.csv")
    categories = {}
    for column in sorted(required):
        categories[column] = {str(k): int(v) for k, v in dataset.obs[column].value_counts(dropna=False).items()}
    manifest = {
        "status": "complete", "path": str(args.data), "shape": [int(dataset.n_obs), int(dataset.n_vars)],
        "obs_names_unique": bool(dataset.obs_names.is_unique), "var_names_unique": bool(dataset.var_names.is_unique),
        "matrix_mode": "backed; one union-pathway column read per cell batch",
        "pathway_source": str(args.pathways), "pathway_version": pathway_config["version"],
        "categories": categories, "inference_unit": "patient",
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