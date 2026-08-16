from __future__ import annotations

import argparse
from pathlib import Path

from cardiogb.data.rds_conversion import mtx_bundle_to_h5ad, run_seurat_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Export observed Seurat counts and convert to H5AD")
    parser.add_argument("--rds", type=Path, required=True)
    parser.add_argument("--assay", default="Spatial")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rscript", type=Path)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    if not args.skip_export:
        completed = run_seurat_export(
            args.rds,
            args.bundle,
            assay=args.assay,
            rscript=None if args.rscript is None else str(args.rscript),
        )
        print(completed.stdout)
    print("H5AD shape:", mtx_bundle_to_h5ad(args.bundle, args.output))


if __name__ == "__main__":
    main()
