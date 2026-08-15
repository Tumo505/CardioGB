"""Manuscript-ready interpretation tables for fitted CardioGB models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from cardiogb.interpretation.attribution import integrated_gradients
from cardiogb.interpretation.mechanistic_insufficiency import mechanistic_insufficiency


@torch.no_grad()
def insufficiency_table(
    model: torch.nn.Module,
    states: torch.Tensor,
    graph: Any,
    *,
    time: float,
    stages: Sequence[object] | None = None,
    domains: Sequence[object] | None = None,
    sections: Sequence[object] | None = None,
) -> pd.DataFrame:
    terms = model.vector_field(time, states, graph)
    values = mechanistic_insufficiency(terms["mechanistic"], terms["residual"])
    frame = pd.DataFrame({"node": np.arange(len(states)), "mi": values.cpu().numpy()})
    for name, labels in (("stage", stages), ("domain", domains), ("section", sections)):
        if labels is not None:
            if len(labels) != len(frame):
                raise ValueError(f"{name} labels disagree with states")
            frame[name] = np.asarray(labels)
    return frame


def attribution_table(
    residual_model: torch.nn.Module,
    states: torch.Tensor,
    graph: Any,
    state_names: Sequence[str],
    *,
    time: float,
    target_state: int,
    steps: int = 32,
) -> pd.DataFrame:
    attribution = integrated_gradients(
        residual_model, time, states, graph, target_state=target_state, steps=steps
    )
    mean_absolute = attribution.abs().mean(dim=0).cpu().numpy()
    return pd.DataFrame(
        {
            "target_state": state_names[target_state],
            "input_state": list(state_names),
            "mean_absolute_attribution": mean_absolute,
            "claim_scope": "model-derived association; hypothesis generating",
        }
    )
