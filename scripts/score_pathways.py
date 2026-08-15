from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np

from cardiogb.data.pathway_scoring import score_pathways
from cardiogb.data.preprocessing import library_size_log1p
from cardiogb.data.state_dataset import StateDataset
from cardiogb.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pathways", type=Path, default=Path("configs/pathways.yaml"))
    parser.add_argument("--section-column", default="orig.ident")
    parser.add_argument("--group-column", default="cid")
    parser.add_argument("--stage-column", default="time_points")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--coordinate-key", default="spatial")
    args = parser.parse_args()
    config = load_yaml(args.pathways)
    data_config = load_yaml(args.data_config)
    adata = ad.read_h5ad(args.input)
    normalized = library_size_log1p(adata.X)
    pathways = {name: details["genes"] for name, details in config["pathways"].items()}
    scoring = config["scoring"]
    result = score_pathways(
        normalized,
        adata.var_names,
        pathways,
        method=scoring["method"],
        output_scaling=scoring["output_scaling"],
        min_genes=int(scoring["min_genes"]),
    )
    state_order = [config["states"][name] for name in ("I", "A", "F", "C", "V", "M")]
    column_order = [result.pathway_names.index(name) for name in state_order]
    stage_values = adata.obs[args.stage_column]
    if not np.issubdtype(stage_values.dtype, np.number):
        stage_map = data_config["datasets"]["zebrafish"]["stages"]
        mapped = stage_values.astype(str).map(stage_map)
        if mapped.isna().any():
            unknown = sorted(stage_values[mapped.isna()].astype(str).unique())
            raise ValueError(f"Unknown stage labels: {unknown}")
        times = mapped.to_numpy(dtype=float)
    else:
        times = stage_values.to_numpy(dtype=float)
    dataset = StateDataset(
        states=result.values[:, column_order],
        coordinates=np.asarray(adata.obsm[args.coordinate_key]),
        sections=adata.obs[args.section_column].astype(str).to_numpy(),
        times=times,
        groups=adata.obs[args.group_column].astype(str).to_numpy(),
        state_names=("I", "A", "F", "C", "V", "M"),
    )
    dataset.save(args.output)


if __name__ == "__main__":
    main()
