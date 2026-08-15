"""Conservative pathway-level cross-species validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def matched_pathway_correlations(
    reference: np.ndarray,
    external: np.ndarray,
    state_names: Sequence[str],
) -> dict[str, float]:
    """Correlate matched stage/domain summaries, never individual spots."""
    reference, external = np.asarray(reference), np.asarray(external)
    if reference.shape != external.shape or reference.ndim != 2:
        raise ValueError("matched summary matrices must have equal [groups, states] shape")
    if reference.shape[1] != len(state_names):
        raise ValueError("state_names length disagrees with matrices")
    result = {}
    for index, name in enumerate(state_names):
        x, y = reference[:, index], external[:, index]
        result[str(name)] = 0.0 if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])
    return result


def orthology_coverage(
    pathway_genes: Mapping[str, Sequence[str]], orthology: Mapping[str, str]
) -> dict[str, dict[str, float]]:
    """Report mapping coverage before any cross-species score comparison."""
    report = {}
    for pathway, genes in pathway_genes.items():
        unique = set(map(str, genes))
        mapped = sum(gene in orthology for gene in unique)
        report[pathway] = {
            "total": len(unique),
            "mapped": mapped,
            "coverage": mapped / len(unique) if unique else 0.0,
        }
    return report
