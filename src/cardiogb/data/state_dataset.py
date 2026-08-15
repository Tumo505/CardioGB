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
        if self.coordinates.shape != (n, 2):
            raise ValueError("coordinates must have shape [observations, 2]")
        if not np.isfinite(self.states).all() or not np.isfinite(self.coordinates).all():
            raise ValueError("states and coordinates must be finite")

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            states=self.states,
            coordinates=self.coordinates,
            sections=self.sections,
            times=self.times,
            groups=self.groups,
            state_names=np.asarray(self.state_names),
        )

    @classmethod
    def load(cls, path: str | Path) -> "StateDataset":
        with np.load(path, allow_pickle=False) as values:
            result = cls(
                values["states"], values["coordinates"], values["sections"],
                values["times"], values["groups"], tuple(map(str, values["state_names"])),
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
            source_nodes = torch.as_tensor(source_index, dtype=torch.long)
            transitions.append(
                CrossSectionalTransition(
                    source_states=torch.as_tensor(self.states[source_index], dtype=torch.float32),
                    target_states=torch.as_tensor(self.states[target_index], dtype=torch.float32),
                    graph=global_graph.induced_subgraph(source_nodes),
                    t0=float(start),
                    t1=float(end),
                    name=f"{start:g}_to_{end:g}",
                )
            )
        return transitions
