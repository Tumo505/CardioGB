"""Deep-ensemble prediction utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from cardiogb.ode.integration import integrate_model
from cardiogb.training.trainer import (
    CrossSectionalTrainer,
    CrossSectionalTransition,
    TrainerConfig,
)
from cardiogb.losses.objective import LossWeights
from cardiogb.utils.seed import seed_everything


def train_deep_ensemble(
    model_factory: Callable[[], nn.Module],
    train: Sequence[CrossSectionalTransition],
    validation: Sequence[CrossSectionalTransition],
    *,
    members: int,
    device: str,
    trainer_config: TrainerConfig = TrainerConfig(),
    loss_weights: LossWeights = LossWeights(),
    seed: int = 0,
    checkpoint_directory: str | Path | None = None,
) -> tuple[list[nn.Module], list[list[dict[str, float]]]]:
    """Train independently initialized members with recorded member seeds."""
    if members < 2:
        raise ValueError("a deep ensemble requires at least two members")
    models, histories = [], []
    for member in range(members):
        seed_everything(seed + member)
        model = model_factory()
        trainer = CrossSectionalTrainer(
            model, device=device, config=trainer_config, loss_weights=loss_weights
        )
        checkpoint = None
        if checkpoint_directory is not None:
            checkpoint = Path(checkpoint_directory) / f"member_{member:02d}.pt"
        histories.append(trainer.fit(train, validation, checkpoint_path=checkpoint))
        models.append(trainer.model)
    return models, histories


@torch.no_grad()
def predict_ensemble(
    models: Sequence[nn.Module],
    states: Tensor,
    graph: Any,
    t0: float,
    t1: float,
    *,
    step_size: float,
    method: str = "rk4",
) -> tuple[Tensor, Tensor, Tensor]:
    if not models:
        raise ValueError("ensemble must contain at least one model")
    outputs = []
    for model in models:
        model.eval()
        device = next(model.parameters()).device
        local_graph = graph.to(device) if hasattr(graph, "to") else graph
        outputs.append(
            integrate_model(
                model, states.to(device), local_graph, t0, t1, step_size=step_size, method=method
            ).cpu()
        )
    stacked = torch.stack(outputs)
    return stacked.mean(0), stacked.std(0, unbiased=False), stacked
