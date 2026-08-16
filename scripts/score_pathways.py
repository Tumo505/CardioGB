from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cardiogb.data.pathway_scoring import score_pathways
from cardiogb.data.preprocessing import library_size_log1p
from cardiogb.data.state_dataset import StateDataset
from cardiogb.utils.config import load_yaml
from cardiogb.utils.io import atomic_json, export_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pathways", type=Path, default=Path("configs/pathways.yaml"))
    parser.add_argument("--section-column", default="orig.ident")
    parser.add_argument("--group-column", default="cid")
    parser.add_argument("--stage-column", default="time_points")
    parser.add_argument("--domain-column", default="annotation")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--coordinate-key", default="spatial")
    parser.add_argument("--qc-output", type=Path, default=Path("results/tables/pathway_qc.csv"))
    parser.add_argument(
        "--diagnostics-output", type=Path, default=Path("results/tables/pathway_genes.json")
    )
    parser.add_argument(
        "--domain-qc-output", type=Path, default=Path("results/tables/pathway_domain_qc.csv")
    )
    parser.add_argument("--method", choices=("mean_scaled", "rank_mean", "module_score"))
    parser.add_argument("--output-scaling", choices=("minmax", "robust_minmax"))
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
        method=args.method or scoring["method"],
        output_scaling=args.output_scaling or scoring["output_scaling"],
        min_genes=int(scoring["min_genes"]),
    )
    state_order = [config["states"][name] for name in ("I", "A", "F", "C", "V", "M")]
    column_order = [result.pathway_names.index(name) for name in state_order]
    stage_values = adata.obs[args.stage_column]
    if not pd.api.types.is_numeric_dtype(stage_values.dtype):
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
        domains=adata.obs[args.domain_column].astype(str).to_numpy(),
    )
    dataset.save(args.output)
    qc = pd.DataFrame(result.values[:, column_order], columns=("I", "A", "F", "C", "V", "M"))
    qc["stage_days"] = times
    summary = qc.groupby("stage_days", observed=True).agg(["count", "mean", "std", "median"])
    summary.columns = [f"{state}_{metric}" for state, metric in summary.columns]
    export_table(summary.reset_index(), args.qc_output)
    qc["domain"] = adata.obs[args.domain_column].astype(str).to_numpy()
    domain_summary = qc.groupby("domain", observed=True)[list("IAFCVM")].mean()
    domain_summary["spots"] = qc.groupby("domain", observed=True).size()
    export_table(domain_summary.reset_index(), args.domain_qc_output)
    atomic_json(
        {
            "pathway_version": config["version"],
            "matched_genes": {name: list(genes) for name, genes in result.matched_genes.items()},
            "missing_genes": {name: list(genes) for name, genes in result.missing_genes.items()},
            "state_correlation": qc[list("IAFCVM")].corr().to_dict(),
        },
        args.diagnostics_output,
    )


if __name__ == "__main__":
    main()
