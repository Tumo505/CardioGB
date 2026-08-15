"""Explicit grouped split definitions for biological-unit generalisation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitDefinition:
    group_column: str
    train_samples: tuple[str, ...]
    validation_samples: tuple[str, ...]
    test_samples: tuple[str, ...]
    train_stages: tuple[str, ...]
    validation_stages: tuple[str, ...]
    test_stages: tuple[str, ...]
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def grouped_split(
    metadata: pd.DataFrame,
    *,
    group_column: str,
    stage_column: str,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
    stratify_by_stage: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SplitDefinition]:
    """Split whole groups, never individual spots, into train/validation/test."""
    if group_column not in metadata or stage_column not in metadata:
        raise ValueError("group and stage columns must exist in metadata")
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be positive and sum to less than one")
    groups = np.asarray(sorted(map(str, metadata[group_column].dropna().unique())))
    if len(groups) < 3:
        raise ValueError("at least three biological groups are required")
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, ...], list[str]] = {}
    if stratify_by_stage:
        stage_sets = metadata.assign(_group=metadata[group_column].astype(str)).groupby("_group")[
            stage_column
        ].agg(lambda values: tuple(sorted(map(str, set(values)))))
        for group, stratum in stage_sets.items():
            strata.setdefault(stratum, []).append(group)
    else:
        strata[()] = groups.tolist()
    train_parts: list[str] = []
    validation_parts: list[str] = []
    test_parts: list[str] = []
    for stratum, members in strata.items():
        shuffled = np.asarray(sorted(members))
        rng.shuffle(shuffled)
        if len(shuffled) < 3:
            raise ValueError(f"stratum {stratum} has fewer than three biological groups")
        n_test = max(1, round(len(shuffled) * test_fraction))
        n_validation = max(1, round(len(shuffled) * validation_fraction))
        if n_test + n_validation >= len(shuffled):
            n_validation = 1
            n_test = 1
        test_parts.extend(shuffled[:n_test])
        validation_parts.extend(shuffled[n_test : n_test + n_validation])
        train_parts.extend(shuffled[n_test + n_validation :])
    train_groups = np.asarray(train_parts)
    validation_groups = np.asarray(validation_parts)
    test_groups = np.asarray(test_parts)
    group_values = metadata[group_column].astype(str)
    train = group_values.isin(train_groups).to_numpy()
    validation = group_values.isin(validation_groups).to_numpy()
    test = group_values.isin(test_groups).to_numpy()
    if np.any(train & validation) or np.any(train & test) or np.any(validation & test):
        raise AssertionError("grouped split masks overlap")

    def stages(mask: np.ndarray) -> tuple[str, ...]:
        return tuple(sorted(map(str, metadata.loc[mask, stage_column].unique())))

    definition = SplitDefinition(
        group_column=group_column,
        train_samples=tuple(sorted(train_groups)),
        validation_samples=tuple(sorted(validation_groups)),
        test_samples=tuple(sorted(test_groups)),
        train_stages=stages(train),
        validation_stages=stages(validation),
        test_stages=stages(test),
        seed=seed,
    )
    return train, validation, test, definition


def temporal_stage_split(
    metadata: pd.DataFrame,
    *,
    stage_column: str,
    train_stages: Sequence[str],
    validation_stages: Sequence[str],
    test_stages: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sets = [set(map(str, values)) for values in (train_stages, validation_stages, test_stages)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("temporal stage sets overlap")
    values = metadata[stage_column].astype(str)
    return tuple(values.isin(items).to_numpy() for items in sets)  # type: ignore[return-value]
