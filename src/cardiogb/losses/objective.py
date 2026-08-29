"""Composable CardioGB training objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from cardiogb.losses.distribution import moment_matching, rbf_mmd, sliced_wasserstein
from cardiogb.losses.residual import residual_penalty
from cardiogb.losses.spatial import graph_smoothness
from cardiogb.models.constraints import bounds_penalty


@dataclass(frozen=True)
class LossWeights:
    distribution: float = 1.0
    spatial: float = 0.0
    biology: float = 0.0
    residual: float = 0.0
    persistence_regret: float = 0.0
    stability: float = 0.0
    semigroup: float = 0.0
    moments: float = 0.0
    wasserstein: float = 0.0


def cardiogb_objective(
    predicted: Tensor,
    observed: Tensor,
    *,
    graph: Any = None,
    residual: Tensor | None = None,
    residual_energy: Tensor | None = None,
    distribution_predicted: Tensor | None = None,
    distribution_observed: Tensor | None = None,
    persistence_reference: Tensor | None = None,
    stability_penalty: Tensor | None = None,
    semigroup_penalty: Tensor | None = None,
    regret_margin: float = 0.0,
    weights: LossWeights = LossWeights(),
) -> tuple[Tensor, dict[str, Tensor]]:
    distribution_predicted = predicted if distribution_predicted is None else distribution_predicted
    distribution_observed = observed if distribution_observed is None else distribution_observed
    components = {
        "distribution": rbf_mmd(distribution_predicted, distribution_observed),
        "moments": moment_matching(distribution_predicted, distribution_observed),
        "wasserstein": sliced_wasserstein(
            distribution_predicted, distribution_observed, num_projections=32, num_quantiles=64
        ),
        "biology": bounds_penalty(predicted),
        "spatial": predicted.new_zeros(()) if graph is None else graph_smoothness(predicted, graph),
        "residual": (
            residual_energy
            if residual_energy is not None
            else predicted.new_zeros(()) if residual is None else residual_penalty(residual)
        ),
    }
    predictive = (
        weights.distribution * components["distribution"]
        + weights.moments * components["moments"]
        + weights.wasserstein * components["wasserstein"]
    )
    if persistence_reference is None:
        baseline = predicted.new_zeros(())
        regret = predicted.new_zeros(())
    else:
        baseline = (
            weights.distribution * rbf_mmd(persistence_reference, distribution_observed)
            + weights.moments * moment_matching(persistence_reference, distribution_observed)
            + weights.wasserstein * sliced_wasserstein(
                persistence_reference,
                distribution_observed,
                num_projections=32,
                num_quantiles=64,
            )
        )
        regret = torch.relu(predictive - baseline + regret_margin)
    stability = predicted.new_zeros(()) if stability_penalty is None else stability_penalty
    semigroup = predicted.new_zeros(()) if semigroup_penalty is None else semigroup_penalty
    if persistence_reference is not None:
        components["persistence_baseline"] = baseline
        components["persistence_regret"] = regret
    if stability_penalty is not None:
        components["stability"] = stability
    if semigroup_penalty is not None:
        components["semigroup"] = semigroup
    total = (
        weights.distribution * components["distribution"]
        + weights.moments * components["moments"]
        + weights.wasserstein * components["wasserstein"]
        + weights.spatial * components["spatial"]
        + weights.biology * components["biology"]
        + weights.residual * components["residual"]
        + weights.persistence_regret * regret
        + weights.stability * stability
        + weights.semigroup * semigroup
    )
    return total, components
