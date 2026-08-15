"""Cross-sectional distribution trainer with checkpointing and early stopping."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from cardiogb.losses.objective import LossWeights, cardiogb_objective
from cardiogb.models.message_passing import TorchGraph
from cardiogb.ode.integration import integrate_model
from cardiogb.training.callbacks import EarlyStopping


@dataclass(frozen=True)
class CrossSectionalTransition:
    source_states: Tensor
    target_states: Tensor
    graph: TorchGraph | Any
    t0: float
    t1: float
    name: str = "transition"


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    step_size: float = 0.05
    solver: str = "rk4"
    gradient_clip: float | None = 5.0
    early_stopping_patience: int = 25
    max_distribution_samples: int = 2048
    mixed_precision: bool = False


class CrossSectionalTrainer:
    """Train on distributions; no target spot is paired to a source spot."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str,
        config: TrainerConfig = TrainerConfig(),
        loss_weights: LossWeights = LossWeights(),
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = config
        self.loss_weights = loss_weights
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.amp_enabled = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.history: list[dict[str, float]] = []

    def _move_transition(self, transition: CrossSectionalTransition) -> CrossSectionalTransition:
        graph = transition.graph.to(self.device) if hasattr(transition.graph, "to") else transition.graph
        return CrossSectionalTransition(
            transition.source_states.to(self.device),
            transition.target_states.to(self.device),
            graph,
            transition.t0,
            transition.t1,
            transition.name,
        )

    def _subsample(self, values: Tensor) -> Tensor:
        limit = self.config.max_distribution_samples
        if len(values) <= limit:
            return values
        index = torch.randperm(len(values), device=values.device)[:limit]
        return values[index]

    def loss_for_transition(
        self, transition: CrossSectionalTransition
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch = self._move_transition(transition)
        prediction = integrate_model(
            self.model,
            batch.source_states,
            batch.graph,
            batch.t0,
            batch.t1,
            step_size=self.config.step_size,
            method=self.config.solver,
        )
        residual = None
        if hasattr(self.model, "vector_field"):
            residual = self.model.vector_field(batch.t1, prediction, batch.graph).get("residual")
        return cardiogb_objective(
            prediction,
            batch.target_states,
            graph=batch.graph,
            residual=residual,
            distribution_predicted=self._subsample(prediction),
            distribution_observed=self._subsample(batch.target_states),
            weights=self.loss_weights,
        )

    def _epoch(self, transitions: Iterable[CrossSectionalTransition], training: bool) -> float:
        self.model.train(training)
        losses: list[float] = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for transition in transitions:
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    loss, _ = self.loss_for_transition(transition)
                if training:
                    self.scaler.scale(loss).backward()
                    if self.config.gradient_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                losses.append(float(loss.detach()))
        if not losses:
            raise ValueError("at least one transition is required")
        return sum(losses) / len(losses)

    def fit(
        self,
        train: Iterable[CrossSectionalTransition],
        validation: Iterable[CrossSectionalTransition],
        *,
        checkpoint_path: str | Path | None = None,
    ) -> list[dict[str, float]]:
        train, validation = list(train), list(validation)
        stopper = EarlyStopping(self.config.early_stopping_patience)
        best_state = deepcopy(self.model.state_dict())
        for epoch in range(self.config.epochs):
            train_loss = self._epoch(train, True)
            validation_loss = self._epoch(validation, False)
            self.history.append(
                {"epoch": float(epoch), "train_loss": train_loss, "validation_loss": validation_loss}
            )
            if validation_loss < stopper.best:
                best_state = deepcopy(self.model.state_dict())
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path, epoch, validation_loss)
            if stopper.update(validation_loss):
                break
        self.model.load_state_dict(best_state)
        return self.history

    def save_checkpoint(self, path: str | Path, epoch: int, validation_loss: float) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "validation_loss": validation_loss,
                "trainer_config": asdict(self.config),
                "loss_weights": asdict(self.loss_weights),
            },
            target,
        )
