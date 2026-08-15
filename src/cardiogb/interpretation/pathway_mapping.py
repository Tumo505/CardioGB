"""Cross-species pathway mapping helpers."""

from __future__ import annotations

from typing import Mapping, Sequence


def map_gene_set(
    genes: Sequence[str],
    orthology: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mapped: list[str] = []
    missing: list[str] = []
    for gene in genes:
        targets = orthology.get(gene, ())
        if targets:
            mapped.extend(targets)
        else:
            missing.append(gene)
    return tuple(dict.fromkeys(mapped)), tuple(missing)
