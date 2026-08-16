from __future__ import annotations

import argparse
import json
import time
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from cardiogb.utils.config import load_yaml
from cardiogb.utils.io import atomic_json, export_table


BASE_URL = "https://rest.ensembl.org"
TARGETS = ("mus_musculus", "homo_sapiens")


def query_biomart_batch(genes: list[str]) -> pd.DataFrame:
    attributes = [
        "ensembl_gene_id",
        "external_gene_name",
        "mmusculus_homolog_ensembl_gene",
        "mmusculus_homolog_associated_gene_name",
        "mmusculus_homolog_orthology_type",
        "mmusculus_homolog_perc_id",
        "hsapiens_homolog_ensembl_gene",
        "hsapiens_homolog_associated_gene_name",
        "hsapiens_homolog_orthology_type",
        "hsapiens_homolog_perc_id",
    ]
    attribute_xml = "".join(f'<Attribute name="{name}" />' for name in attributes)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" '
        'count="" datasetConfigVersion="0.6">'
        '<Dataset name="drerio_gene_ensembl" interface="default">'
        f'<Filter name="external_gene_name" value="{",".join(genes)}" />'
        f"{attribute_xml}</Dataset></Query>"
    )
    last_error = None
    for host in ("https://www.ensembl.org", "https://useast.ensembl.org"):
        try:
            response = requests.post(
                f"{host}/biomart/martservice", data={"query": xml}, timeout=120
            )
            response.raise_for_status()
            if response.text.startswith("Query ERROR"):
                raise RuntimeError(response.text.strip())
            return pd.read_csv(StringIO(response.text), sep="\t", header=None, names=attributes)
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
    raise RuntimeError(f"Ensembl BioMart batch failed: {last_error}")


def query_biomart(genes: list[str], batch_size: int = 15) -> pd.DataFrame:
    frames = [
        query_biomart_batch(genes[start : start + batch_size])
        for start in range(0, len(genes), batch_size)
    ]
    return pd.concat(frames, ignore_index=True)


def fetch_homologies(gene: str, retries: int = 3) -> list[dict[str, object]]:
    url = f"{BASE_URL}/homology/symbol/danio_rerio/{gene}"
    collected = []
    for species in TARGETS:
        for attempt in range(retries):
            response = requests.get(
                url,
                params={"target_species": species, "type": "orthologues", "format": "full"},
                headers={"Accept": "application/json"},
                timeout=60,
            )
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2**attempt)
                continue
            if response.status_code in {400, 404}:
                break
            response.raise_for_status()
            data = response.json().get("data", [])
            if data:
                collected.extend(data[0].get("homologies", []))
            break
        else:
            # Preserve an explicit unmapped entry rather than aborting the
            # complete versioned table on one unstable/ambiguous symbol.
            continue
    return collected


def lookup_symbols(ids: list[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        response = requests.post(
            f"{BASE_URL}/lookup/id",
            json={"ids": batch},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
        for identifier, details in response.json().items():
            result[identifier] = None if details is None else details.get("display_name")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Ensembl zebrafish orthology tables")
    parser.add_argument("--pathways", type=Path, default=Path("configs/pathways.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/orthology.csv"))
    parser.add_argument(
        "--report", type=Path, default=Path("results/tables/orthology_coverage.json")
    )
    args = parser.parse_args()
    config = load_yaml(args.pathways)
    pathway_membership: dict[str, list[str]] = {}
    for pathway, details in config["pathways"].items():
        for gene in details["genes"]:
            pathway_membership.setdefault(str(gene), []).append(str(pathway))

    biomart = query_biomart(sorted(pathway_membership))
    rows = []
    mapped: dict[str, set[str]] = {target: set() for target in TARGETS}
    prefixes = {"mus_musculus": "mmusculus", "homo_sapiens": "hsapiens"}
    for _, record in biomart.iterrows():
        source_gene = str(record["external_gene_name"])
        for species, prefix in prefixes.items():
            target_id = record[f"{prefix}_homolog_ensembl_gene"]
            if pd.isna(target_id) or not str(target_id):
                continue
            mapped[species].add(source_gene)
            rows.append(
                {
                    "source_gene": source_gene,
                    "pathways": "|".join(pathway_membership[source_gene]),
                    "target_species": species,
                    "target_gene": record[f"{prefix}_homolog_associated_gene_name"],
                    "target_ensembl_id": target_id,
                    "homology_type": record[f"{prefix}_homolog_orthology_type"],
                    "source_identity": None,
                    "target_identity": record[f"{prefix}_homolog_perc_id"],
                    "source": "Ensembl BioMart drerio_gene_ensembl",
                }
            )
    export_table(pd.DataFrame(rows), args.output)
    all_genes = set(pathway_membership)
    report = {
        "pathway_version": config["version"],
        "source_species": "danio_rerio",
        "source_url": "https://www.ensembl.org/biomart/martview",
        "total_source_genes": len(all_genes),
        "targets": {
            target: {
                "mapped_source_genes": len(mapped[target]),
                "coverage": len(mapped[target]) / len(all_genes),
                "unmapped": sorted(all_genes - mapped[target]),
            }
            for target in TARGETS
        },
    }
    atomic_json(report, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
