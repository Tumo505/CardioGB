"""Mechanistic insufficiency scores at node and grouped levels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor


def mechanistic_insufficiency(
    mechanistic_term: Tensor,
    residual_term: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    if mechanistic_term.shape != residual_term.shape:
        raise ValueError("mechanistic and residual terms must have equal shapes")
    mech_norm = mechanistic_term.norm(dim=-1)
    residual_norm = residual_term.norm(dim=-1)
    return residual_norm / (mech_norm + residual_norm + eps)


def aggregate_insufficiency(values: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"mechanistic_insufficiency": values, "group": groups})
    return frame.groupby("group", observed=True)["mechanistic_insufficiency"].agg(
        ["count", "mean", "std", "median"]
    ).reset_index()
