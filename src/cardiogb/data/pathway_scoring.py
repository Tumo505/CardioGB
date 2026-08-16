"""Configurable pathway-to-state scoring for dense and sparse expression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.stats import rankdata


@dataclass(frozen=True)
class PathwayScoreResult:
    values: np.ndarray
    pathway_names: tuple[str, ...]
    matched_genes: dict[str, tuple[str, ...]]
    missing_genes: dict[str, tuple[str, ...]]


def score_pathways(
    expression: np.ndarray | sparse.spmatrix,
    gene_names: Sequence[str],
    pathways: Mapping[str, Sequence[str]],
    *,
    method: str = "mean_scaled",
    output_scaling: str | None = "minmax",
    min_genes: int = 1,
    module_bins: int = 24,
    module_control_size: int = 25,
    random_state: int = 0,
) -> PathwayScoreResult:
    """Score rows (spots/cells) against pathway gene sets.

    ``mean_scaled`` z-scores each gene before averaging, ``rank_mean`` averages
    per-gene percentile ranks, and ``module_score`` subtracts expression-bin-
    matched control genes in the spirit of Seurat AddModuleScore.
    """
    if expression.ndim != 2:
        raise ValueError("expression must be a two-dimensional observations-by-genes matrix")
    if expression.shape[1] != len(gene_names):
        raise ValueError("gene_names length must match expression columns")
    if method not in {"mean_scaled", "mean_expression", "rank_mean", "module_score"}:
        raise ValueError(f"Unsupported scoring method: {method}")
    if output_scaling not in {None, "minmax", "robust_minmax"}:
        raise ValueError(f"Unsupported output scaling: {output_scaling}")

    lookup: dict[str, int] = {}
    original: dict[str, str] = {}
    for index, gene in enumerate(gene_names):
        key = str(gene).casefold()
        if key not in lookup:
            lookup[key] = index
            original[key] = str(gene)

    columns: list[np.ndarray] = []
    names: list[str] = []
    matched: dict[str, tuple[str, ...]] = {}
    missing: dict[str, tuple[str, ...]] = {}
    gene_average = None
    gene_bins = None
    rng = np.random.default_rng(random_state)
    if method == "module_score":
        gene_average = np.asarray(expression.mean(axis=0)).reshape(-1)
        order = np.argsort(np.argsort(gene_average, kind="stable"), kind="stable")
        gene_bins = np.minimum(module_bins - 1, order * module_bins // len(order))
    for pathway, requested_genes in pathways.items():
        keys = [str(gene).casefold() for gene in requested_genes]
        present_keys = list(dict.fromkeys(key for key in keys if key in lookup))
        absent = tuple(str(gene) for gene, key in zip(requested_genes, keys) if key not in lookup)
        if len(present_keys) < min_genes:
            raise ValueError(
                f"Pathway {pathway!r} matched {len(present_keys)} genes; minimum is {min_genes}"
            )
        indices = [lookup[key] for key in present_keys]
        block = expression[:, indices]
        if method == "module_score":
            assert gene_bins is not None
            pathway_indices = set(indices)
            controls = []
            for index in indices:
                candidates = np.flatnonzero(gene_bins == gene_bins[index])
                candidates = np.asarray([item for item in candidates if item not in pathway_indices])
                if len(candidates):
                    controls.extend(
                        rng.choice(
                            candidates,
                            size=min(module_control_size, len(candidates)),
                            replace=False,
                        ).tolist()
                    )
            controls = sorted(set(controls))
            if not controls:
                raise ValueError(f"No matched control genes for pathway {pathway!r}")
            score = np.asarray(block.mean(axis=1)).reshape(-1) - np.asarray(
                expression[:, controls].mean(axis=1)
            ).reshape(-1)
        else:
            if sparse.issparse(block):
                block = block.toarray()
            block = np.asarray(block, dtype=np.float64)
        if method == "mean_scaled":
            mean = block.mean(axis=0, keepdims=True)
            std = block.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            block = (block - mean) / std
            score = block.mean(axis=1)
        elif method == "rank_mean":
            score = (rankdata(block, axis=0, method="average") / len(block)).mean(axis=1)
        elif method == "mean_expression":
            score = block.mean(axis=1)
        if output_scaling == "minmax":
            low, high = float(score.min()), float(score.max())
            score = np.zeros_like(score) if high == low else (score - low) / (high - low)
        elif output_scaling == "robust_minmax":
            low, high = np.quantile(score, (0.01, 0.99))
            if high == low:
                score = np.zeros_like(score)
            else:
                score = np.clip((score - low) / (high - low), 0.0, 1.0)
        columns.append(score.astype(np.float32, copy=False))
        names.append(str(pathway))
        matched[str(pathway)] = tuple(original[key] for key in present_keys)
        missing[str(pathway)] = absent

    values = np.column_stack(columns) if columns else np.empty((expression.shape[0], 0), np.float32)
    return PathwayScoreResult(values, tuple(names), matched, missing)
