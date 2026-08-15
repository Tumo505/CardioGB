"""Evaluation metrics for unmatched biological distributions."""

from __future__ import annotations

import numpy as np
import torch

from cardiogb.losses.distribution import moment_matching, rbf_mmd, sliced_wasserstein


def distribution_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    pred = torch.as_tensor(predicted, dtype=torch.float32)
    obs = torch.as_tensor(observed, dtype=torch.float32)
    return {
        "mmd": float(rbf_mmd(pred, obs)),
        "moment_error": float(moment_matching(pred, obs)),
        "sliced_wasserstein": float(sliced_wasserstein(pred, obs)),
    }
