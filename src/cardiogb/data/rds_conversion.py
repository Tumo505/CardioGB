"""Conservative RDS inspection and conversion readiness checks."""

from __future__ import annotations

import gzip
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.io import mmread


@dataclass(frozen=True)
class RDSInspection:
    path: str
    size_bytes: int
    compression: str
    rscript: str | None
    status: str
    r_classes: tuple[str, ...] = ()
    top_level_names: tuple[str, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_rds(path: str | Path, rscript: str | None = None) -> RDSInspection:
    """Inspect an RDS object when R is available; otherwise return a clear report."""
    rds_path = Path(path).expanduser().resolve()
    if not rds_path.is_file():
        raise FileNotFoundError(rds_path)
    with rds_path.open("rb") as handle:
        magic = handle.read(2)
    compression = "gzip" if magic == b"\x1f\x8b" else "unknown_or_uncompressed"
    executable = rscript or shutil.which("Rscript")
    if executable is None:
        return RDSInspection(
            path=str(rds_path),
            size_bytes=rds_path.stat().st_size,
            compression=compression,
            rscript=None,
            status="dependency_missing",
            message="Rscript is unavailable; object structure was not guessed.",
        )

    code = (
        "x <- readRDS(commandArgs(trailingOnly=TRUE)[1]); "
        "cat('CLASS\\t', paste(class(x), collapse='|'), '\\n', sep=''); "
        "n <- names(x); if (is.null(n)) n <- character(); "
        "cat('NAMES\\t', paste(n, collapse='|'), '\\n', sep='')"
    )
    completed = subprocess.run(
        [executable, "--vanilla", "-e", code, str(rds_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return RDSInspection(
            str(rds_path), rds_path.stat().st_size, compression, executable,
            "inspection_failed", message=completed.stderr.strip(),
        )
    values: dict[str, tuple[str, ...]] = {}
    for line in completed.stdout.splitlines():
        if "\t" in line:
            key, raw = line.split("\t", 1)
            values[key] = tuple(value for value in raw.split("|") if value)
    return RDSInspection(
        str(rds_path), rds_path.stat().st_size, compression, executable,
        "inspected", values.get("CLASS", ()), values.get("NAMES", ()),
    )


def can_stream_gzip(path: str | Path, bytes_to_read: int = 64) -> bool:
    """Check whether a gzip-compressed RDS stream can be opened."""
    try:
        with gzip.open(path, "rb") as handle:
            handle.read(bytes_to_read)
        return True
    except (OSError, EOFError):
        return False


def run_seurat_export(
    input_rds: str | Path,
    output_dir: str | Path,
    *,
    assay: str,
    rscript: str | None = None,
    exporter_script: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the bundled Seurat exporter without guessing when R is unavailable."""
    executable = rscript or shutil.which("Rscript")
    if executable is None:
        raise RuntimeError("Rscript is required for Seurat RDS conversion")
    input_path = Path(input_rds).resolve()
    output_path = Path(output_dir).resolve()
    script = Path(exporter_script or Path(__file__).parents[3] / "scripts" / "export_seurat.R")
    output_path.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [executable, "--vanilla", str(script), str(input_path), str(output_path), assay],
        check=True,
        capture_output=True,
        text=True,
    )


def mtx_bundle_to_h5ad(
    bundle_dir: str | Path,
    output_path: str | Path,
    *,
    coordinate_columns: tuple[str, str] = ("x", "y"),
) -> tuple[int, int]:
    """Convert the deterministic Seurat MatrixMarket export into AnnData."""
    try:
        import anndata as ad
    except ImportError as error:
        raise RuntimeError("anndata is required for H5AD conversion") from error
    bundle = Path(bundle_dir).resolve()
    matrix = mmread(bundle / "matrix.mtx").tocsr().T
    genes = pd.read_csv(bundle / "genes.tsv", sep="\t", header=None, names=["gene"])
    barcodes = pd.read_csv(bundle / "barcodes.tsv", sep="\t", header=None, names=["barcode"])
    metadata = read_exported_metadata(bundle / "metadata.tsv")
    if matrix.shape != (len(barcodes), len(genes)):
        raise ValueError(
            f"Matrix shape {matrix.shape} disagrees with barcodes/genes "
            f"({len(barcodes)}, {len(genes)})"
        )
    metadata = metadata.reindex(barcodes["barcode"].astype(str))
    if metadata.isna().all(axis=1).any():
        raise ValueError("Exported metadata is missing one or more expression barcodes")
    adata = ad.AnnData(X=matrix, obs=metadata, var=pd.DataFrame(index=genes["gene"].astype(str)))
    if all(column in metadata for column in coordinate_columns):
        adata.obsm["spatial"] = metadata[list(coordinate_columns)].to_numpy(dtype=float)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output, compression="gzip")
    return adata.shape


def read_exported_metadata(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    frame.index = frame.index.astype(str)
    if not frame.index.is_unique:
        raise ValueError("Exported metadata identifiers are not unique")
    return frame
