from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cardiogb.statistics import benjamini_hochberg
from cardiogb.utils.io import atomic_json, export_table


def clustered_spearman(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    cluster: str = "state",
    seed: int = 20260815,
    n_resamples: int = 10000,
) -> dict[str, float | int | str]:
    data = frame[[cluster, x, y]].dropna().copy()
    clusters = data[cluster].astype(str).unique()
    if len(clusters) < 2 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return {
            "spearman": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "n_rows": len(data),
            "n_clusters": len(clusters),
            "cluster_unit": cluster,
        }
    observed = float(spearmanr(data[x], data[y]).statistic)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_resamples, dtype=float)
    null = np.empty(n_resamples, dtype=float)
    grouped = {name: part.copy() for name, part in data.groupby(cluster, observed=True)}
    for iteration in range(n_resamples):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = []
        for draw, name in enumerate(sampled):
            piece = grouped[name].copy()
            piece[cluster] = f"{name}__draw_{draw}"
            pieces.append(piece)
        boot = pd.concat(pieces, ignore_index=True)
        bootstrap[iteration] = spearmanr(boot[x], boot[y]).statistic
        permuted = data.copy()
        for _, indices in data.groupby(cluster, sort=False).groups.items():
            positions = data.index.get_indexer(indices)
            permuted.iloc[positions, permuted.columns.get_loc(y)] = rng.permutation(
                data.loc[indices, y].to_numpy()
            )
        null[iteration] = spearmanr(permuted[x], permuted[y]).statistic
    finite_bootstrap = bootstrap[np.isfinite(bootstrap)]
    finite_null = null[np.isfinite(null)]
    return {
        "spearman": observed,
        "ci_lower": float(np.quantile(finite_bootstrap, 0.025)),
        "ci_upper": float(np.quantile(finite_bootstrap, 0.975)),
        "p_value": float((np.count_nonzero(np.abs(finite_null) >= abs(observed)) + 1) / (len(finite_null) + 1)),
        "n_rows": len(data),
        "n_clusters": len(clusters),
        "cluster_unit": cluster,
    }


def transition_level_spearman(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    seed: int = 20260815,
    n_resamples: int = 10000,
) -> dict[str, float | int | str]:
    data = frame[["transition", x, y]].dropna().groupby("transition", observed=True)[[x, y]].mean().reset_index()
    if len(data) < 3 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return {
            "spearman": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
            "p_value": np.nan, "n_rows": len(data), "n_clusters": len(data),
            "cluster_unit": "forecast transition",
        }
    observed = float(spearmanr(data[x], data[y]).statistic)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_resamples, dtype=float)
    null = np.empty(n_resamples, dtype=float)
    x_values = data[x].to_numpy()
    y_values = data[y].to_numpy()
    for iteration in range(n_resamples):
        sampled = rng.integers(0, len(data), size=len(data))
        bootstrap[iteration] = spearmanr(x_values[sampled], y_values[sampled]).statistic
        null[iteration] = spearmanr(x_values, rng.permutation(y_values)).statistic
    finite_bootstrap = bootstrap[np.isfinite(bootstrap)]
    finite_null = null[np.isfinite(null)]
    return {
        "spearman": observed,
        "ci_lower": float(np.quantile(finite_bootstrap, 0.025)) if len(finite_bootstrap) else np.nan,
        "ci_upper": float(np.quantile(finite_bootstrap, 0.975)) if len(finite_bootstrap) else np.nan,
        "p_value": float((np.count_nonzero(np.abs(finite_null) >= abs(observed)) + 1) / (len(finite_null) + 1)),
        "n_rows": len(data), "n_clusters": len(data), "cluster_unit": "forecast transition",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-aware uncertainty–error and horizon inference")
    parser.add_argument("--state-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    frame = pd.read_csv(args.state_predictions)
    required = {"protocol", "transition", "state", "horizon_days", "ensemble_std", "absolute_error"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing uncertainty columns: {sorted(missing)}")
    rows = []
    for protocol, subset in frame.groupby("protocol", observed=True):
        for label, x, y in (
            ("uncertainty_vs_error", "ensemble_std", "absolute_error"),
            ("uncertainty_vs_horizon", "horizon_days", "ensemble_std"),
            ("error_vs_horizon", "horizon_days", "absolute_error"),
        ):
            result = clustered_spearman(
                subset, x, y, seed=20260815 + len(rows), n_resamples=args.resamples
            )
            rows.append({"protocol": protocol, "analysis_level": "state_clustered", "test": label, "x": x, "y": y, **result})
            transition_result = transition_level_spearman(
                subset, x, y, seed=20260815 + len(rows), n_resamples=args.resamples
            )
            rows.append({"protocol": protocol, "analysis_level": "transition_aggregated", "test": label, "x": x, "y": y, **transition_result})
    result = pd.DataFrame(rows)
    result["p_adjust_bh"] = np.nan
    valid = result["p_value"].notna()
    result.loc[valid, "p_adjust_bh"] = benjamini_hochberg(
        result.loc[valid, "p_value"].to_numpy()
    )
    export_table(result, args.output_dir / "uncertainty_inferential_tests.csv")
    atomic_json(
        {
            "status": "complete",
            "tests": len(result),
            "bootstrap_resamples": args.resamples,
            "permutation_resamples": args.resamples,
            "multiplicity": "Benjamini-Hochberg across the prespecified uncertainty test family",
            "resampling_unit": "primary transition-aggregated bootstrap/permutation with pathway-state-clustered sensitivity analysis",
            "scope": "forecast-transition and transition-state inference; mouse analyses are exploratory because biological replication remains unavailable",
        },
        args.output_dir / "uncertainty_inference_manifest.json",
    )
    print(json.dumps({"status": "complete", "tests": len(result)}, indent=2))


if __name__ == "__main__":
    main()
