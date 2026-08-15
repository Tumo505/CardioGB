from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd

from cardiogb.data.graphs import build_spatial_knn_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--section-column", default="orig.ident")
    parser.add_argument("--coordinate-key", default="spatial")
    parser.add_argument("--k", type=int, nargs="+", default=[8])
    parser.add_argument("--qc-output", type=Path)
    args = parser.parse_args()
    adata = ad.read_h5ad(args.input, backed="r")
    rows = []
    for k in args.k:
        graph = build_spatial_knn_graph(
            adata.obsm[args.coordinate_key], adata.obs[args.section_column].astype(str), k=k
        )
        target = args.output if len(args.k) == 1 else args.output.with_name(
            f"{args.output.stem}_k{k}{args.output.suffix}"
        )
        graph.save(target)
        rows.append({"k": k, "output": str(target), **graph.statistics()})
    if args.qc_output:
        args.qc_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.qc_output, index=False)
    print(json.dumps(rows, indent=2))
    adata.file.close()


if __name__ == "__main__":
    main()
