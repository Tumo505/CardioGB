from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from cardiogb.data.pathway_scoring import score_pathways
from cardiogb.data.preprocessing import library_size_log1p
from cardiogb.data.state_dataset import StateDataset
from cardiogb.utils.config import load_yaml
from cardiogb.utils.io import atomic_json, export_table


STATE_KEYS = ("I", "A", "F", "C", "V", "M")
STAGES = ((3.0, "3d"), (7.0, "7d"), (14.0, "14d"), (21.0, "21d"))


def read_10x_h5(path: Path) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    with h5py.File(path) as handle:
        matrix = handle["matrix"]
        shape = tuple(map(int, matrix["shape"][:]))
        gene_by_spot = sparse.csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
            shape=shape,
        )
        genes = np.asarray([item.decode() for item in matrix["features"]["name"][:]])
        barcodes = np.asarray([item.decode() for item in matrix["barcodes"][:]])
    return gene_by_spot.T.tocsr(), genes, barcodes


def select_orthologs(path: Path, pathway_config: dict) -> tuple[dict[str, list[str]], dict]:
    table = pd.read_csv(path)
    table = table[table["target_species"].eq("mus_musculus") & table["target_gene"].notna()].copy()
    table["priority"] = table["homology_type"].map({"ortholog_one2one": 0, "ortholog_one2many": 1}).fillna(2)
    table["identity"] = pd.to_numeric(table["target_identity"], errors="coerce").fillna(-1)
    table = table.sort_values(["source_gene", "priority", "identity"], ascending=[True, True, False])
    best = table.drop_duplicates("source_gene").set_index("source_gene")
    pathways: dict[str, list[str]] = {}
    coverage = {}
    for state, long_name in pathway_config["states"].items():
        sources = pathway_config["pathways"][long_name]["genes"]
        mapped = {gene: str(best.loc[gene, "target_gene"]) for gene in sources if gene in best.index}
        pathways[long_name] = list(dict.fromkeys(mapped.values()))
        coverage[state] = {
            "source_genes": len(sources),
            "mapped_source_genes": len(mapped),
            "mapping_coverage": len(mapped) / len(sources),
            "selected_targets": pathways[long_name],
        }
    return pathways, coverage


def curated_mouse_pathways(mouse_config: dict, fish_config: dict) -> tuple[dict[str, list[str]], dict]:
    pathways = {
        long_name: list(mouse_config["pathways"][long_name]["genes"])
        for long_name in mouse_config["pathways"]
    }
    coverage = {}
    for state, long_name in fish_config["states"].items():
        source_count = len(fish_config["pathways"][long_name]["genes"])
        targets = pathways[long_name]
        coverage[state] = {
            "source_genes": source_count,
            "curated_target_genes": len(targets),
            "selected_targets": targets,
            "mapping_version": mouse_config["version"],
        }
    return pathways, coverage


def positions_for_barcodes(path: Path, barcodes: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(
        path,
        header=None,
        names=("barcode", "in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"),
    ).set_index("barcode")
    missing = set(barcodes) - set(frame.index)
    if missing:
        raise ValueError(f"{len(missing)} matrix barcodes lack spatial coordinates")
    return frame.loc[barcodes, ["pixel_col", "pixel_row"]].to_numpy(dtype=float)


def score_combined(expression, genes, pathways, config):
    normalized = library_size_log1p(expression)
    result = score_pathways(
        normalized,
        genes,
        pathways,
        method="mean_scaled",
        output_scaling="robust_minmax",
        min_genes=3,
        random_state=20260815,
    )
    order = [result.pathway_names.index(config["states"][state]) for state in STATE_KEYS]
    return result, result.values[:, order]


def process_spatial(root: Path, pathways, pathway_config, output: Path, results: Path, coverage: dict):
    matrices, coordinates, sections, times, genes = [], [], [], [], None
    for stage, folder in STAGES:
        matrix, local_genes, barcodes = read_10x_h5(root / folder / "filtered_feature_bc_matrix.h5")
        if genes is None:
            genes = local_genes
        elif not np.array_equal(genes, local_genes):
            raise ValueError("mouse Visium samples do not share an identical feature order")
        matrices.append(matrix)
        coordinates.append(positions_for_barcodes(root / folder / "spatial" / "tissue_positions_list.csv", barcodes))
        sections.extend([folder] * len(barcodes))
        times.extend([stage] * len(barcodes))
    expression = sparse.vstack(matrices, format="csr")
    result, states = score_combined(expression, genes, pathways, pathway_config)
    dataset = StateDataset(
        states=states,
        coordinates=np.vstack(coordinates),
        sections=np.asarray(sections),
        times=np.asarray(times),
        groups=np.asarray(sections),
        state_names=STATE_KEYS,
    )
    dataset.save(output)
    frame = pd.DataFrame(states, columns=STATE_KEYS)
    frame["stage_days"] = times
    summary = frame.groupby("stage_days", observed=True)[list(STATE_KEYS)].agg(["mean", "std", "median"])
    summary.columns = [f"{state}_{stat}" for state, stat in summary.columns]
    export_table(summary.reset_index(), results / "mouse_spatial_stage_scores.csv")
    gene_lookup = set(map(str.casefold, genes))
    for state, long_name in pathway_config["states"].items():
        targets = pathways[long_name]
        observed = [gene for gene in targets if gene.casefold() in gene_lookup]
        coverage[state]["expressed_targets"] = observed
        coverage[state]["expressed_target_count"] = len(observed)
    atomic_json(
        {
            "spots": len(states),
            "genes": len(genes),
            "stages": [stage for stage, _ in STAGES],
            "pathway_coverage": coverage,
            "matched_genes": {name: list(value) for name, value in result.matched_genes.items()},
            "missing_genes": {name: list(value) for name, value in result.missing_genes.items()},
            "replication_warning": "one spatial sample per stage; stage comparisons are descriptive",
        },
        results / "mouse_spatial_qc.json",
    )


def read_scrna_sample(matrix_path: Path):
    prefix = matrix_path.name.removesuffix("_matrix.mtx.gz")
    genes_path = matrix_path.with_name(prefix + "_genes.tsv.gz")
    with gzip.open(genes_path, "rt") as handle:
        genes = pd.read_csv(handle, sep="\t", header=None).iloc[:, 1].astype(str).to_numpy()
    with gzip.open(matrix_path, "rb") as handle:
        matrix = mmread(handle).tocsr().T
    return prefix, matrix, genes


def process_scrna(root: Path, pathways, pathway_config, results: Path):
    samples = []
    for matrix_path in sorted(root.glob("*_matrix.mtx.gz")):
        sample, matrix, local_genes = read_scrna_sample(matrix_path)
        samples.append((sample, matrix, local_genes))
    if not samples:
        raise ValueError("no mouse scRNA matrix files found")
    common = set(map(str, samples[0][2]))
    for _, _, local_genes in samples[1:]:
        common.intersection_update(map(str, local_genes))
    genes = np.asarray(list(dict.fromkeys(gene for gene in map(str, samples[0][2]) if gene in common)))
    matrices, sample_names = [], []
    for sample, matrix, local_genes in samples:
        lookup = {}
        for index, gene in enumerate(map(str, local_genes)):
            lookup.setdefault(gene, index)
        matrices.append(matrix[:, [lookup[gene] for gene in genes]])
        sample_names.extend([sample] * matrix.shape[0])
    expression = sparse.vstack(matrices, format="csr")
    _, states = score_combined(expression, genes, pathways, pathway_config)
    frame = pd.DataFrame(states, columns=STATE_KEYS)
    frame["sample"] = sample_names
    summary = frame.groupby("sample", observed=True)[list(STATE_KEYS)].mean().reset_index()
    parsed = summary["sample"].str.extract(r"P(?P<age>\d+)_(?P<day>\d+)(?P<condition>MI|Sham)$")
    summary = pd.concat((summary, parsed), axis=1)
    export_table(summary, results / "mouse_scrna_sample_scores.csv")
    contrasts = []
    for (age, day), group in summary.groupby(["age", "day"], observed=True):
        if set(group["condition"]) != {"MI", "Sham"}:
            continue
        mi = group.set_index("condition").loc["MI", list(STATE_KEYS)]
        sham = group.set_index("condition").loc["Sham", list(STATE_KEYS)]
        contrasts.extend(
            {"age": age, "day": day, "state": state, "mi_minus_sham": float(mi[state] - sham[state])}
            for state in STATE_KEYS
        )
    export_table(pd.DataFrame(contrasts), results / "mouse_scrna_mi_sham_contrasts.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process neonatal mouse external-validation data")
    parser.add_argument("--spatial-root", type=Path, required=True)
    parser.add_argument("--scrna-root", type=Path, required=True)
    parser.add_argument("--orthology", type=Path, required=True)
    parser.add_argument("--mouse-pathways", type=Path, default=Path("configs/mouse_pathways.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml("configs/pathways.yaml")
    mouse_config = load_yaml(args.mouse_pathways)
    pathways, coverage = curated_mouse_pathways(mouse_config, config)
    process_spatial(args.spatial_root, pathways, config, args.output, args.results_dir, coverage)
    process_scrna(args.scrna_root, pathways, config, args.results_dir)
    print(json.dumps({"status": "complete", "spatial_output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
