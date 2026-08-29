from __future__ import annotations

import argparse
import gzip
import json
import tarfile
import time
from pathlib import Path

import pandas as pd
import requests

from cardiogb.utils.io import export_table


def collect_ids(raw_root: Path) -> list[str]:
    identifiers = set()
    bulk = raw_root / "zebrafish/validation/GSE106884/GSE106884_mRNA_gene_TPM.txt.gz"
    with gzip.open(bulk, "rt") as handle:
        next(handle)
        for line in handle:
            identifiers.add(line.split("\t", 1)[0].strip('"'))
    for accession in ("GSE153170", "GSE234990", "GSE237276"):
        archive = raw_root / f"zebrafish/validation/{accession}/{accession}_RAW.tar"
        with tarfile.open(archive) as outer:
            for member in outer.getmembers():
                source = outer.extractfile(member)
                if source is None:
                    continue
                with gzip.open(source, "rt") as handle:
                    next(handle)
                    for line in handle:
                        identifiers.add(line.split("\t", 1)[0])
    return sorted(value for value in identifiers if value.startswith("ENSDARG"))


def mapping_from_release_gtf(
    identifiers: list[str],
    cache_path: Path,
    url: str = "https://ftp.ensembl.org/pub/release-116/gtf/danio_rerio/Danio_rerio.GRCz11.116.gtf.gz",
) -> list[dict[str, str]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file():
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with cache_path.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    wanted = set(identifiers)
    mapped = {}
    with gzip.open(cache_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attributes = {}
            for item in fields[8].split(";"):
                item = item.strip()
                if " " in item:
                    key, value = item.split(" ", 1)
                    attributes[key] = value.strip().strip('"')
            identifier = attributes.get("gene_id", "").split(".")[0]
            if identifier in wanted:
                mapped[identifier] = {
                    "ensembl_gene_id": identifier,
                    "gene_symbol": attributes.get("gene_name"),
                    "biotype": attributes.get("gene_biotype"),
                    "species": "danio_rerio",
                    "assembly_name": "GRCz11",
                    "ensembl_release": 116,
                }
    return [
        mapped.get(
            identifier,
            {
                "ensembl_gene_id": identifier,
                "gene_symbol": None,
                "biotype": None,
                "species": "danio_rerio",
                "assembly_name": "GRCz11",
                "ensembl_release": 116,
            },
        )
        for identifier in identifiers
    ]

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a versioned zebrafish Ensembl-to-symbol table")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/zebrafish_ensembl_symbols.csv"))
    parser.add_argument(
        "--gtf-cache",
        type=Path,
        default=Path("data/raw/reference/ensembl/release_116/Danio_rerio.GRCz11.116.gtf.gz"),
    )
    args = parser.parse_args()
    identifiers = collect_ids(args.raw_root)
    rows = mapping_from_release_gtf(identifiers, args.gtf_cache)
    export_table(pd.DataFrame(rows), args.output)
    mapped = sum(bool(row["gene_symbol"]) for row in rows)
    print(json.dumps({"identifiers": len(rows), "mapped": mapped, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
