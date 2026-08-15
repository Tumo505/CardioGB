"""Leakage-safe per-section spatial graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SpatialGraph:
    edge_index: np.ndarray
    edge_attr: np.ndarray
    section_ids: np.ndarray
    k: int
    symmetric: bool

    @property
    def num_nodes(self) -> int:
        return int(len(self.section_ids))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def statistics(self) -> dict[str, float | int]:
        adjacency = sparse.coo_matrix(
            (np.ones(self.num_edges), (self.edge_index[0], self.edge_index[1])),
            shape=(self.num_nodes, self.num_nodes),
        ).tocsr()
        components, _ = connected_components(adjacency, directed=False)
        distances = self.edge_attr[:, 0]
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_components": int(components),
            "mean_distance": float(distances.mean()) if len(distances) else 0.0,
            "max_distance": float(distances.max()) if len(distances) else 0.0,
        }

    def to_torch(self, device: str = "cpu") -> tuple[object, object]:
        import torch

        return (
            torch.as_tensor(self.edge_index, dtype=torch.long, device=device),
            torch.as_tensor(self.edge_attr, dtype=torch.float32, device=device),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            section_ids=self.section_ids,
            k=np.asarray(self.k),
            symmetric=np.asarray(self.symmetric),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SpatialGraph":
        with np.load(path, allow_pickle=False) as values:
            return cls(
                values["edge_index"], values["edge_attr"], values["section_ids"],
                int(values["k"]), bool(values["symmetric"]),
            )


def build_spatial_knn_graph(
    coordinates: np.ndarray,
    section_ids: Sequence[Hashable],
    *,
    k: int = 8,
    symmetric: bool = True,
) -> SpatialGraph:
    """Build kNN edges independently within each section.

    Edge attributes are ``[distance, delta_x, delta_y]``. No edge can connect
    two sections, which prevents a common spatial preprocessing error.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    sections = np.asarray(section_ids)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [num_nodes, 2]")
    if len(sections) != len(coordinates):
        raise ValueError("section_ids length must match coordinates")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates contain non-finite values")
    if k < 1:
        raise ValueError("k must be positive")

    edges: set[tuple[int, int]] = set()
    for section in dict.fromkeys(sections.tolist()):
        indices = np.flatnonzero(sections == section)
        if len(indices) <= 1:
            continue
        local_k = min(k, len(indices) - 1)
        tree = cKDTree(coordinates[indices])
        _, neighbours = tree.query(coordinates[indices], k=local_k + 1)
        neighbours = np.atleast_2d(neighbours)
        for local_source, local_targets in enumerate(neighbours):
            source = int(indices[local_source])
            for local_target in np.atleast_1d(local_targets):
                target = int(indices[int(local_target)])
                if source == target:
                    continue
                edges.add((source, target))
                if symmetric:
                    edges.add((target, source))

    ordered = sorted(edges)
    if ordered:
        edge_index = np.asarray(ordered, dtype=np.int64).T
        delta = coordinates[edge_index[1]] - coordinates[edge_index[0]]
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        edge_attr = np.concatenate((distance, delta), axis=1).astype(np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 3), dtype=np.float32)
    return SpatialGraph(edge_index, edge_attr, sections, k, symmetric)
