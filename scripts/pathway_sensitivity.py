from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cardiogb.data.pathway_scoring import score_pathways
from cardiogb.data.preprocessing import library_size_log1p
from cardiogb.utils.config import load_yaml
from cardiogb.utils.io import export_table


METHODS = ("mean_scaled", "rank_mean", "module_score")
STATE_KEYS = ("I", "A", "F", "C", "V", "M")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pathway scoring definitions")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pathways", type=Path, default=Path("configs/pathways.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/pathway_sensitivity.csv"))
    args = parser.parse_args()
    config = load_yaml(args.pathways)
    adata = ad.read_h5ad(args.input)
    expression = library_size_log1p(adata.X)
    pathways = {name: details["genes"] for name, details in config["pathways"].items()}
    pathway_names = [config["states"][key] for key in STATE_KEYS]
    outputs = {}
    for method in METHODS:
        result = score_pathways(
            expression,
            adata.var_names,
            pathways,
            method=method,
            output_scaling="robust_minmax",
            min_genes=int(config["scoring"]["min_genes"]),
            random_state=20260815,
        )
        outputs[method] = result.values[:, [result.pathway_names.index(name) for name in pathway_names]]

    rows = []
    primary = outputs["mean_scaled"]
    stages = adata.obs["time_points"].astype(str).to_numpy()
    domains = adata.obs["annotation"].astype(str).to_numpy()
    for method, values in outputs.items():
        for index, state in enumerate(STATE_KEYS):
            stage_primary = pd.DataFrame({"group": stages, "value": primary[:, index]}).groupby(
                "group", observed=True
            )["value"].mean()
            stage_variant = pd.DataFrame({"group": stages, "value": values[:, index]}).groupby(
                "group", observed=True
            )["value"].mean().reindex(stage_primary.index)
            domain_primary = pd.DataFrame({"group": domains, "value": primary[:, index]}).groupby(
                "group", observed=True
            )["value"].mean()
            domain_variant = pd.DataFrame({"group": domains, "value": values[:, index]}).groupby(
                "group", observed=True
            )["value"].mean().reindex(domain_primary.index)
            rows.append(
                {
                    "pathway_version": config["version"],
                    "method": method,
                    "state": state,
                    "spot_spearman_vs_mean_scaled": spearmanr(
                        primary[:, index], values[:, index]
                    ).statistic,
                    "stage_mean_spearman_vs_mean_scaled": spearmanr(
                        stage_primary, stage_variant
                    ).statistic,
                    "domain_mean_spearman_vs_mean_scaled": spearmanr(
                        domain_primary, domain_variant
                    ).statistic,
                }
            )
    export_table(pd.DataFrame(rows), args.output)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
