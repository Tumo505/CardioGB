from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from cardiogb.data.pathway_scoring import score_pathways
from cardiogb.utils.config import load_yaml
from cardiogb.utils.io import atomic_json, export_table


STATE_KEYS = ("I", "A", "F", "C", "V", "M")
GSE153170_GENOTYPE = {
    "GSM4635175": "WT",
    "GSM4635176": "WT",
    "GSM4635177": "prrx1b-/-",
    "GSM4635178": "prrx1b-/-",
}


def registered_pathways():
    config = load_yaml("configs/pathways.yaml")
    names = [config["states"][key] for key in STATE_KEYS]
    return config, {name: list(config["pathways"][name]["genes"]) for name in names}


def load_mapping(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    table = pd.read_csv(path)
    table = table[table["gene_symbol"].notna()].copy()
    id_to_symbol = table.drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")["gene_symbol"].astype(str).to_dict()
    symbol_to_id = {}
    for identifier, symbol in id_to_symbol.items():
        symbol_to_id.setdefault(symbol.casefold(), identifier)
    return id_to_symbol, symbol_to_id


def score_registered(expression: np.ndarray, genes: list[str], pathways, min_genes: int = 3):
    result = score_pathways(
        expression,
        genes,
        pathways,
        method="mean_scaled",
        output_scaling="robust_minmax",
        min_genes=min_genes,
        random_state=20260815,
    )
    return result, result.values


def normalize_counts(counts: np.ndarray, library: np.ndarray) -> np.ndarray:
    library = np.maximum(np.asarray(library, dtype=np.float64), 1.0)
    return np.log1p(np.asarray(counts, dtype=np.float64) * (1e4 / library[:, None]))


def process_gse106884(root: Path, mapping: dict[str, str], pathways, results: Path):
    table = pd.read_csv(root / "GSE106884_mRNA_gene_TPM.txt.gz", sep="\t", index_col=0)
    symbols = pd.Series(table.index.map(mapping), index=table.index)
    keep = symbols.notna()
    collapsed = table.loc[keep].assign(_symbol=symbols.loc[keep].to_numpy()).groupby("_symbol", observed=True).sum()
    expression = np.log1p(collapsed.to_numpy(float).T)
    result, states = score_registered(expression, collapsed.index.astype(str).tolist(), pathways)
    frame = pd.DataFrame(states, columns=STATE_KEYS)
    frame["sample"] = table.columns
    parsed = frame["sample"].str.extract(r"(?P<stage_days>\d+)dpa mRNA, rep(?P<replicate>\d+)")
    frame = pd.concat((frame, parsed), axis=1)
    frame["stage_days"] = pd.to_numeric(frame["stage_days"])
    export_table(frame, results / "gse106884_sample_states.csv")
    return {"samples": len(frame), "matched_genes": {key: list(value) for key, value in result.matched_genes.items()}}


def stream_tar_counts(archive: Path, mapping: dict[str, str], pathway_genes: set[str]):
    samples = []
    with tarfile.open(archive) as outer:
        for member in outer.getmembers():
            source = outer.extractfile(member)
            if source is None:
                continue
            with gzip.open(source, "rt") as handle:
                header = handle.readline().rstrip("\r\n").split("\t")[1:]
                library = np.zeros(len(header), dtype=np.float64)
                selected: dict[str, np.ndarray] = {}
                for line in handle:
                    identifier, values_text = line.rstrip("\r\n").split("\t", 1)
                    values = np.fromstring(values_text, sep="\t", dtype=np.float32)
                    library += values
                    symbol = mapping.get(identifier)
                    if symbol is not None and symbol.casefold() in pathway_genes:
                        if symbol in selected:
                            selected[symbol] += values
                        else:
                            selected[symbol] = values
            accession = member.name.split("_", 1)[0]
            samples.append((accession, member.name, header, library, selected))
    return samples


def process_cell_study(accession: str, archive: Path, mapping, pathways, results: Path):
    wanted = {gene.casefold() for genes in pathways.values() for gene in genes}
    samples = stream_tar_counts(archive, mapping, wanted)
    frames, coverage = [], None
    for sample, filename, cells, library, selected in samples:
        genes = list(selected)
        expression = normalize_counts(np.column_stack([selected[gene] for gene in genes]), library)
        minimum = 1 if accession == "GSE237276" else 3
        result, states = score_registered(expression, genes, pathways, min_genes=minimum)
        coverage = {key: list(value) for key, value in result.matched_genes.items()}
        frame = pd.DataFrame(states, columns=STATE_KEYS)
        frame["cell"] = [f"{sample}:{cell}" for cell in cells]
        frame["sample"] = sample
        if accession == "GSE153170":
            frame["genotype"] = GSE153170_GENOTYPE[sample]
            frame["stage_days"] = 7.0
        else:
            match = re.search(r"_(1|3|7)d(?:p|p)c?i", filename, flags=re.IGNORECASE)
            if match is None:
                match = re.search(r"_(1|3|7)dpi", filename, flags=re.IGNORECASE)
            frame["stage_days"] = float(match.group(1)) if match else np.nan
        frames.append(frame)
    cells = pd.concat(frames, ignore_index=True)
    export_table(cells, results / f"{accession.lower()}_cell_states.csv")
    group_columns = ["sample", "stage_days"] + (["genotype"] if "genotype" in cells else [])
    summary = cells.groupby(group_columns, observed=True)[list(STATE_KEYS)].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        str(first) if not str(second) or str(second).startswith("Unnamed") else f"{first}_{second}"
        for first, second in summary.columns
    ]
    export_table(summary, results / f"{accession.lower()}_biological_unit_states.csv")
    return {"cells": len(cells), "samples": len(samples), "matched_genes": coverage}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess downloaded zebrafish validation cohorts")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/zebrafish/validation"))
    parser.add_argument("--mapping", type=Path, default=Path("data/processed/zebrafish_ensembl_symbols.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/added_validation"))
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    _, pathways = registered_pathways()
    mapping, _ = load_mapping(args.mapping)
    qc = {
        "GSE106884": process_gse106884(args.raw_root / "GSE106884", mapping, pathways, args.results_dir),
        "GSE153170": process_cell_study("GSE153170", args.raw_root / "GSE153170/GSE153170_RAW.tar", mapping, pathways, args.results_dir),
        "GSE237276": process_cell_study("GSE237276", args.raw_root / "GSE237276/GSE237276_RAW.tar", mapping, pathways, args.results_dir),
    }
    first = args.raw_root / "GSE234990/GSE234990_RAW.tar"
    second = args.raw_root / "GSE237276/GSE237276_RAW.tar"
    qc["GSE234990"] = {
        "sha256": sha256(first),
        "duplicates_GSE237276_download": sha256(first) == sha256(second),
        "analysis_policy": "deduplicated by GSM; GSE237276 processed once",
    }
    qc["GSE153170"]["experimental_unit_note"] = "four CEL-seq2 libraries; cells are not independent biological replicates"
    atomic_json(
        {
            "ready": ["GSE106884", "GSE153170", "GSE206787", "GSE217828", "GSE237276"],
            "deduplicated": {"GSE234990": "same downloaded GSM archive as GSE237276"},
            "mapping": str(args.mapping),
        },
        args.results_dir / "validation_readiness.json",
    )
    atomic_json(qc, args.results_dir / "zebrafish_validation_qc.json")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "matched_genes"} for key, value in qc.items()}, indent=2))


if __name__ == "__main__":
    main()
