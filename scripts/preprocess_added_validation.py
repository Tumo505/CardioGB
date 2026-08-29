from __future__ import annotations

import argparse
import csv
import gzip
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from cardiogb.data.pathway_scoring import score_pathways
from cardiogb.data.preprocessing import library_size_log1p
from cardiogb.data.state_dataset import StateDataset
from cardiogb.utils.config import load_yaml
from cardiogb.utils.io import atomic_json, export_table
from preprocess_mouse_validation import read_10x_h5


STATE_KEYS = ("I", "A", "F", "C", "V", "M")
GSE206787_VISIUM = {
    0.0: "GSM7898096_Visium_heart_undamaged",
    3.0: "GSM7898097_Visium_heart_3dPI",
    7.0: "GSM7898098_Visium_heart_7dPI",
    14.0: "GSM7898099_Visium_heart_14dPI",
}


def mouse_pathways() -> tuple[dict[str, list[str]], dict]:
    fish = load_yaml("configs/pathways.yaml")
    mouse = load_yaml("configs/mouse_pathways.yaml")
    pathways = {
        name: list(mouse["pathways"][name]["genes"])
        for name in mouse["pathways"]
    }
    return pathways, fish


def score_mouse(expression, genes, pathways, fish_config):
    result = score_pathways(
        library_size_log1p(expression),
        genes,
        pathways,
        method="mean_scaled",
        output_scaling="robust_minmax",
        min_genes=3,
        random_state=20260815,
    )
    order = [result.pathway_names.index(fish_config["states"][key]) for key in STATE_KEYS]
    return result, result.values[:, order]


def _safe_destination(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"unsafe archive path: {relative}")
    return target


def extract_gse206787_inputs(archive: Path, interim: Path) -> dict[float, tuple[Path, Path]]:
    interim.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with tarfile.open(archive) as outer:
        members = {member.name: member for member in outer.getmembers()}
        for stage, prefix in GSE206787_VISIUM.items():
            matrix_name = prefix + "_filtered_feature_bc_matrix.h5"
            image_name = prefix + "_images.tar.gz"
            matrix_path = _safe_destination(interim, matrix_name)
            if not matrix_path.is_file():
                source = outer.extractfile(members[matrix_name])
                if source is None:
                    raise ValueError(f"cannot read {matrix_name}")
                matrix_path.write_bytes(source.read())
            positions_path = _safe_destination(interim, prefix + "_tissue_positions_list.csv")
            if not positions_path.is_file():
                nested_source = outer.extractfile(members[image_name])
                if nested_source is None:
                    raise ValueError(f"cannot read {image_name}")
                with tarfile.open(fileobj=nested_source, mode="r:gz") as nested:
                    position_member = next(
                        member for member in nested.getmembers()
                        if member.name.endswith("tissue_positions_list.csv")
                    )
                    source = nested.extractfile(position_member)
                    if source is None:
                        raise ValueError(f"cannot read positions in {image_name}")
                    positions_path.write_bytes(source.read())
            outputs[stage] = matrix_path, positions_path
    return outputs


def process_gse206787(raw: Path, interim: Path, output: Path, results: Path) -> None:
    pathways, fish_config = mouse_pathways()
    files = extract_gse206787_inputs(raw, interim)
    matrices, genes, coordinates, sections, times = [], None, [], [], []
    for stage, prefix in GSE206787_VISIUM.items():
        matrix, local_genes, barcodes = read_10x_h5(files[stage][0])
        if genes is None:
            genes = local_genes
        elif not np.array_equal(genes, local_genes):
            raise ValueError("GSE206787 Visium feature orders differ")
        positions = pd.read_csv(
            files[stage][1], header=None,
            names=("barcode", "in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"),
        ).set_index("barcode")
        missing = set(barcodes) - set(positions.index)
        if missing:
            raise ValueError(f"{len(missing)} GSE206787 barcodes lack positions")
        coordinates.append(positions.loc[barcodes, ["pixel_col", "pixel_row"]].to_numpy(float))
        matrices.append(matrix)
        sections.extend([prefix] * len(barcodes))
        times.extend([stage] * len(barcodes))
    expression = sparse.vstack(matrices, format="csr")
    result, states = score_mouse(expression, genes, pathways, fish_config)
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
    frame["sample"] = sections
    export_table(frame.groupby(["sample", "stage_days"], observed=True)[list(STATE_KEYS)].mean().reset_index(), results / "gse206787_sample_states.csv")
    atomic_json(
        {
            "study": "GSE206787",
            "spots": len(states),
            "stages": sorted(files),
            "matched_genes": {key: list(value) for key, value in result.matched_genes.items()},
            "role": "independent adult mouse MI external prediction",
            "replication_warning": "one Visium sample per stage",
        },
        results / "gse206787_qc.json",
    )


def normalize_cell_id(value: str) -> str:
    value = str(value)
    return value[:-2] + "-1" if value.endswith(".1") else value


def selected_gene_rows(matrix_path: Path, pathways: dict[str, list[str]]):
    wanted = {gene.casefold() for genes in pathways.values() for gene in genes}
    selected: dict[str, np.ndarray] = {}
    with gzip.open(matrix_path, "rt", newline="") as handle:
        header = next(csv.reader([handle.readline()]))
        has_gene_header = header[0].casefold() in {"gene", "gene_symbol", "symbol"}
        cells = header[1:] if has_gene_header else header
        for line in handle:
            gene, values = line.rstrip("\r\n").split(",", 1)
            if gene.casefold() in wanted and gene.casefold() not in selected:
                selected[gene.casefold()] = np.fromstring(values, sep=",", dtype=np.float32)
    return np.asarray([normalize_cell_id(cell) for cell in cells]), selected


def score_selected_cells(cells, rows, metadata, id_column, pathways):
    local = metadata.copy()
    local["_cell_key"] = local[id_column].map(normalize_cell_id)
    local = local.drop_duplicates("_cell_key").set_index("_cell_key")
    missing = [cell for cell in cells if cell not in local.index]
    if missing:
        raise ValueError(f"{len(missing)} matrix cells lack metadata")
    local = local.loc[cells].reset_index()
    library = pd.to_numeric(local["nCount_RNA"], errors="coerce").to_numpy(float)
    library = np.maximum(library, 1.0)
    scores = []
    coverage = {}
    for name, genes in pathways.items():
        matched = [gene for gene in genes if gene.casefold() in rows]
        if len(matched) < 3:
            raise ValueError(f"{name} has only {len(matched)} matched genes")
        block = np.column_stack([np.log1p(rows[gene.casefold()] * 1e4 / library) for gene in matched])
        scale = block.std(axis=0)
        scale[scale == 0] = 1.0
        score = ((block - block.mean(axis=0)) / scale).mean(axis=1)
        low, high = np.quantile(score, (0.01, 0.99))
        score = np.zeros_like(score) if high == low else np.clip((score - low) / (high - low), 0, 1)
        scores.append(score.astype(np.float32))
        coverage[name] = {"matched": matched, "missing": [gene for gene in genes if gene not in matched]}
    return local, np.column_stack(scores), coverage


def process_gse217828(root: Path, results: Path) -> None:
    pathways, fish_config = mouse_pathways()
    order = [fish_config["states"][key] for key in STATE_KEYS]
    pathways = {name: pathways[name] for name in order}
    datasets = {
        "scrna": (
            root / "GSE217828_scRNA_integrated.raw_count.csv.gz",
            root / "GSE217828_scRNA_integrated.meta_data.csv.gz",
            "Cell_ID",
            "sample",
        ),
        "snrna": (
            root / "GSE217828_snRNA_raw_umi.csv.gz",
            root / "GSE217828_snRNA_metadata.csv.gz",
            "Cell_ID",
            "group1",
        ),
    }
    qc = {"study": "GSE217828", "role": "held-out mammalian intervention validation"}
    for modality, (matrix_path, metadata_path, id_column, unit_column) in datasets.items():
        if modality == "scrna":
            with gzip.open(metadata_path, "rt", newline="") as handle:
                declared = next(csv.reader([handle.readline()]))
            metadata = pd.read_csv(
                metadata_path,
                header=None,
                skiprows=1,
                names=["Cell_ID", *declared],
            )
        else:
            metadata = pd.read_csv(metadata_path)
        cells, rows = selected_gene_rows(matrix_path, pathways)
        metadata, states, coverage = score_selected_cells(cells, rows, metadata, id_column, pathways)
        frame = pd.DataFrame(states, columns=STATE_KEYS)
        keep = [column for column in (unit_column, "genotype", "condition", "group2", "Cell_type", "Cell_state") if column in metadata]
        frame = pd.concat((metadata[keep].reset_index(drop=True), frame), axis=1)
        summary = frame.groupby(keep[:1], observed=True)[list(STATE_KEYS)].agg(["mean", "std", "count"]).reset_index()
        summary.columns = [
            str(first) if not str(second) or str(second).startswith("Unnamed") else f"{first}_{second}"
            for first, second in summary.columns
        ]
        export_table(summary, results / f"gse217828_{modality}_biological_unit_states.csv")
        qc[modality] = {"cells": len(cells), "biological_unit_column": unit_column, "coverage": coverage}
    atomic_json(qc, results / "gse217828_qc.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess newly added independent validation cohorts")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/added_validation"))
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    process_gse206787(
        args.raw_root / "mouse/validation/GSE206787/GSE206787_RAW.tar",
        args.interim_root / "mouse/validation/GSE206787",
        args.processed_root / "mouse/validation/gse206787_states.npz",
        args.results_dir,
    )
    process_gse217828(args.raw_root / "mouse/validation/GSE217828", args.results_dir)
    readiness = {
        "ready": ["GSE206787", "GSE217828"],
        "requires_ensembl_to_symbol_mapping": ["GSE106884", "GSE153170", "GSE234990", "GSE237276"],
        "reason": "downloaded zebrafish validation matrices use Ensembl IDs while registered pathways use curated gene symbols",
    }
    atomic_json(readiness, args.results_dir / "validation_readiness.json")
    print(json.dumps(readiness, indent=2))


if __name__ == "__main__":
    main()
