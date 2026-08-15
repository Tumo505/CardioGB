"""Configurable pathway-to-state scoring for dense and sparse expression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse


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
) -> PathwayScoreResult:
    """Score rows (spots/cells) against pathway gene sets.

    ``mean_scaled`` z-scores each matched gene across observations before
    averaging. ``mean_expression`` averages expression directly. Gene matching
    is case-insensitive and diagnostics are returned for every pathway.
    """
    if expression.ndim != 2:
        raise ValueError("expression must be a two-dimensional observations-by-genes matrix")
    if expression.shape[1] != len(gene_names):
        raise ValueError("gene_names length must match expression columns")
    if method not in {"mean_scaled", "mean_expression"}:
        raise ValueError(f"Unsupported scoring method: {method}")
    if output_scaling not in {None, "minmax"}:
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
        if sparse.issparse(block):
            block = block.toarray()
        block = np.asarray(block, dtype=np.float64)
        if method == "mean_scaled":
            mean = block.mean(axis=0, keepdims=True)
            std = block.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            block = (block - mean) / std
        score = block.mean(axis=1)
        if output_scaling == "minmax":
            low, high = float(score.min()), float(score.max())
            score = np.zeros_like(score) if high == low else (score - low) / (high - low)
        columns.append(score.astype(np.float32, copy=False))
        names.append(str(pathway))
        matched[str(pathway)] = tuple(original[key] for key in present_keys)
        missing[str(pathway)] = absent

    values = np.column_stack(columns) if columns else np.empty((expression.shape[0], 0), np.float32)
    return PathwayScoreResult(values, tuple(names), matched, missing)

