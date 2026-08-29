"""Model-ready pathway-state datasets and cross-sectional transitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from cardiogb.data.graphs import build_spatial_knn_graph
from cardiogb.models.message_passing import TorchGraph
from cardiogb.training.trainer import CrossSectionalTransition


@dataclass(frozen=True)
class StateDataset:
    states: np.ndarray
    coordinates: np.ndarray
    sections: np.ndarray
    times: np.ndarray
    groups: np.ndarray
    state_names: tuple[str, ...]
    domains: np.ndarray | None = None

    def validate(self) -> None:
        n = len(self.states)
        if self.states.shape != (n, len(self.state_names)):
            raise ValueError("states shape disagrees with state names")
        for name, values in {
            "coordinates": self.coordinates,
            "sections": self.sections,
            "times": self.times,
            "groups": self.groups,
        }.items():
            if len(values) != n:
                raise ValueError(f"{name} length disagrees with states")
        if self.domains is not None and len(self.domains) != n:
            raise ValueError("domains length disagrees with states")
        if self.coordinates.shape != (n, 2):
            raise ValueError("coordinates must have shape [observations, 2]")
        if not np.isfinite(self.states).all() or not np.isfinite(self.coordinates).all():
            raise ValueError("states and coordinates must be finite")

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            states=np.asarray(self.states, dtype=np.float32),
            coordinates=np.asarray(self.coordinates, dtype=np.float64),
            sections=np.asarray(self.sections, dtype=str),
            times=np.asarray(self.times, dtype=np.float64),
            groups=np.asarray(self.groups, dtype=str),
            state_names=np.asarray(self.state_names, dtype=str),
        )
        if self.domains is not None:
            payload["domains"] = np.asarray(self.domains, dtype=str)
        np.savez_compressed(target, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "StateDataset":
        with np.load(path, allow_pickle=False) as values:
            result = cls(
                values["states"], values["coordinates"], values["sections"],
                values["times"], values["groups"], tuple(map(str, values["state_names"])),
                values["domains"] if "domains" in values.files else None,
            )
        result.validate()
        return result

    def build_graph(self, k: int = 8) -> TorchGraph:
        graph = build_spatial_knn_graph(self.coordinates, self.sections, k=k)
        edge_index, edge_attr = graph.to_torch()
        return TorchGraph(edge_index, edge_attr)

    def transitions(
        self,
        *,
        mask: np.ndarray | None = None,
        k: int = 8,
        adjacent_only: bool = True,
        max_nodes: int | None = None,
    ) -> list[CrossSectionalTransition]:
        self.validate()
        selected = np.ones(len(self.states), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if len(selected) != len(self.states):
            raise ValueError("mask length disagrees with dataset")
        global_graph = self.build_graph(k)
        times = sorted(np.unique(self.times[selected]).tolist())
        pairs = list(zip(times[:-1], times[1:])) if adjacent_only else [
            (start, end) for i, start in enumerate(times[:-1]) for end in times[i + 1 :]
        ]
        transitions = []
        for start, end in pairs:
            source_index = np.flatnonzero(selected & (self.times == start))
            target_index = np.flatnonzero(selected & (self.times == end))
            if len(source_index) == 0 or len(target_index) == 0:
                continue
            base_name = f"{start:g}_to_{end:g}"
            intermediate_times = tuple(float(value) for value in times if start < value < end)
            chunks = self._source_chunks(source_index, max_nodes)
            for chunk_number, (section, chunk) in enumerate(chunks):
                source_nodes = torch.as_tensor(chunk, dtype=torch.long)
                patch_name = base_name
                if len(chunks) > 1:
                    patch_name = f"{base_name}__{section}__patch_{chunk_number:03d}"
                transitions.append(
                    CrossSectionalTransition(
                        source_states=torch.as_tensor(self.states[chunk], dtype=torch.float32),
                        target_states=torch.as_tensor(self.states[target_index], dtype=torch.float32),
                        graph=global_graph.induced_subgraph(source_nodes),
                        t0=float(start),
                        t1=float(end),
                        name=patch_name,
                        evaluation_group=base_name if len(chunks) > 1 else None,
                        intermediate_times=intermediate_times,
                    )
                )
        return transitions

    def _source_chunks(
        self, source_index: np.ndarray, max_nodes: int | None
    ) -> list[tuple[str, np.ndarray]]:
        if max_nodes is None:
            return [("all", source_index)]
        if max_nodes < 2:
            raise ValueError("max_nodes must be at least two")
        chunks: list[tuple[str, np.ndarray]] = []
        source_sections = self.sections[source_index].astype(str)
        for section in sorted(np.unique(source_sections)):
            local = source_index[source_sections == section]
            order = np.lexsort((self.coordinates[local, 1], self.coordinates[local, 0]))
            ordered = local[order]
            for offset in range(0, len(ordered), max_nodes):
                chunks.append((section, ordered[offset : offset + max_nodes]))
        return chunks

    def transition_patches_between(
        self,
        t0: float,
        t1: float,
        *,
        source_mask: np.ndarray | None = None,
        target_mask: np.ndarray | None = None,
        k: int = 8,
        max_nodes: int = 4000,
        name: str | None = None,
    ) -> list[CrossSectionalTransition]:
        """Construct section-restricted patches for one unmatched transition."""
        source_selected = np.ones(len(self.states), dtype=bool) if source_mask is None else np.asarray(source_mask, dtype=bool)
        target_selected = np.ones(len(self.states), dtype=bool) if target_mask is None else np.asarray(target_mask, dtype=bool)
        source_index = np.flatnonzero(source_selected & (self.times == t0))
        target_index = np.flatnonzero(target_selected & (self.times == t1))
        if not len(source_index) or not len(target_index):
            raise ValueError(f"transition {t0:g}->{t1:g} has an empty stage distribution")
        graph = self.build_graph(k)
        base_name = name or f"{t0:g}_to_{t1:g}"
        chunks = self._source_chunks(source_index, max_nodes)
        return [
            CrossSectionalTransition(
                torch.as_tensor(self.states[chunk], dtype=torch.float32),
                torch.as_tensor(self.states[target_index], dtype=torch.float32),
                graph.induced_subgraph(torch.as_tensor(chunk, dtype=torch.long)),
                float(t0),
                float(t1),
                f"{base_name}__{section}__patch_{index:03d}",
                base_name,
            )
            for index, (section, chunk) in enumerate(chunks)
        ]

    def transition_between(
        self,
        t0: float,
        t1: float,
        *,
        source_mask: np.ndarray | None = None,
        target_mask: np.ndarray | None = None,
        k: int = 8,
        name: str | None = None,
    ) -> CrossSectionalTransition:
        """Construct one explicitly unmatched source/target stage transition."""
        source_selected = np.ones(len(self.states), dtype=bool) if source_mask is None else np.asarray(source_mask, dtype=bool)
        target_selected = np.ones(len(self.states), dtype=bool) if target_mask is None else np.asarray(target_mask, dtype=bool)
        source_index = np.flatnonzero(source_selected & (self.times == t0))
        target_index = np.flatnonzero(target_selected & (self.times == t1))
        if not len(source_index) or not len(target_index):
            raise ValueError(f"transition {t0:g}->{t1:g} has an empty stage distribution")
        global_graph = self.build_graph(k)
        return CrossSectionalTransition(
            torch.as_tensor(self.states[source_index], dtype=torch.float32),
            torch.as_tensor(self.states[target_index], dtype=torch.float32),
            global_graph.induced_subgraph(torch.as_tensor(source_index, dtype=torch.long)),
            float(t0),
            float(t1),
            name or f"{t0:g}_to_{t1:g}",
        )
