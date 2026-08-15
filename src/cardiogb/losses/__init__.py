"""Distributional, spatial, biological, and residual objectives."""

from cardiogb.losses.distribution import moment_matching, rbf_mmd, sliced_wasserstein
from cardiogb.losses.objective import LossWeights, cardiogb_objective

__all__ = [
    "LossWeights",
    "cardiogb_objective",
    "moment_matching",
    "rbf_mmd",
    "sliced_wasserstein",
]
