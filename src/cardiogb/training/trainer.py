"""Cross-sectional distribution trainer with checkpointing and early stopping."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from cardiogb.losses.objective import LossWeights, cardiogb_objective
from cardiogb.models.message_passing import TorchGraph
from cardiogb.ode.integration import integrate_model, integrate_model_with_residual_energy
from cardiogb.training.callbacks import EarlyStopping


@dataclass(frozen=True)
class CrossSectionalTransition:
    source_states: Tensor
    target_states: Tensor
    graph: TorchGraph | Any
    t0: float
    t1: float
    name: str = "transition"
    evaluation_group: str | None = None


    intermediate_times: tuple[float, ...] = ()
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
    max_ode_steps_per_transition: int | None = 16
    mixed_precision: bool = False
    amp_dtype: str = "bfloat16"
    patches_per_transition_per_epoch: int | None = 2
    thermal_cooldown_every_epochs: int | None = None
    thermal_cooldown_seconds: float = 0.0
    patch_batch_size: int = 1
    force_float32_integration: bool = True
    mechanistic_learning_rate_scale: float = 0.25
    gradient_checkpointing: bool = True
    gradient_checkpoint_interval: int = 2
    cache_transitions_on_device: bool = True
    max_device_transition_cache_bytes: int = 268_435_456
    warm_start_epochs: int = 0


    multi_horizon_curriculum_epochs: int = 0
    stability_velocity_target: float = 0.4
    regret_margin: float = 0.0
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
        if hasattr(model, "mechanistic_model") and hasattr(model, "residual_model"):
            mechanistic_parameters = list(model.mechanistic_model.parameters())
            mechanistic_ids = {id(parameter) for parameter in mechanistic_parameters}
            residual_parameters = [
                parameter for parameter in model.parameters() if id(parameter) not in mechanistic_ids
            ]
            parameter_groups = [
                {
                    "params": mechanistic_parameters,
                    "lr": config.learning_rate * config.mechanistic_learning_rate_scale,
                },
                {"params": residual_parameters, "lr": config.learning_rate},
            ]
        else:
            parameter_groups = model.parameters()
        self.optimizer = torch.optim.AdamW(
            parameter_groups, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.amp_enabled = config.mixed_precision and self.device.type == "cuda"
        amp_dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        if config.amp_dtype not in amp_dtypes:
            raise ValueError(f"unsupported AMP dtype: {config.amp_dtype}")
        self.amp_dtype = amp_dtypes[config.amp_dtype]
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )
        self.history: list[dict[str, float]] = []
        self.current_epoch = 0

    @staticmethod
    def _transition_nbytes(transition: CrossSectionalTransition) -> int:
        tensors = [transition.source_states, transition.target_states]
        if transition.graph is not None:
            tensors.extend([transition.graph.edge_index, transition.graph.edge_attr])
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors if tensor is not None)

    def _move_transition(self, transition: CrossSectionalTransition) -> CrossSectionalTransition:
        graph = transition.graph.to(self.device) if hasattr(transition.graph, "to") else transition.graph
        return CrossSectionalTransition(
            source_states=transition.source_states.to(self.device),
            target_states=transition.target_states.to(self.device),
            graph=graph,
            t0=transition.t0,
            t1=transition.t1,
            name=transition.name,
            evaluation_group=transition.evaluation_group,
            intermediate_times=transition.intermediate_times,
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
        step_size = self.config.step_size
        if self.config.max_ode_steps_per_transition is not None:
            interval = abs(batch.t1 - batch.t0)
            step_size = max(step_size, interval / self.config.max_ode_steps_per_transition)
        stable_float32 = (
            self.config.force_float32_integration
            and self.device.type == "cuda"
            and hasattr(self.model, "project_state")
        )
        context = (
            torch.autocast(device_type="cuda", enabled=False)
            if stable_float32
            else nullcontext()
        )
        with context:
            source_states = batch.source_states.float() if stable_float32 else batch.source_states
            prediction, trajectory_residual_energy = integrate_model_with_residual_energy(
                self.model,
                source_states,
                batch.graph,
                batch.t0,
                batch.t1,
                step_size=step_size,
                method=self.config.solver,
                checkpoint_steps=(
                    self.config.gradient_checkpoint_interval
                    if self.config.gradient_checkpointing and self.model.training
                    else False
                ),
            )
            residual = None
            stability_penalty = prediction.new_zeros(())
            semigroup_penalty = prediction.new_zeros(())
            if hasattr(self.model, "vector_field"):
                duration = abs(batch.t1 - batch.t0)
                start_components = self.model.vector_field(
                    batch.t0,
                    source_states,
                    batch.graph,
                    forecast_horizon=duration,
                )
                end_components = self.model.vector_field(
                    batch.t1,
                    prediction,
                    batch.graph,
                    forecast_horizon=duration,
                )
                residual = end_components.get("residual")
                target = self.config.stability_velocity_target
                velocity = (
                    torch.relu(start_components["total"].abs() - target).square().mean()
                    + torch.relu(end_components["total"].abs() - target).square().mean()
                )
                acceleration = (
                    (end_components["total"] - start_components["total"])
                    / max(duration, torch.finfo(prediction.dtype).eps)
                ).square().mean()
                stability_penalty = velocity + 0.1 * acceleration
            if (
                batch.intermediate_times
                and self.loss_weights.semigroup > 0
                and hasattr(self.model, "vector_field")
            ):
                midpoint = min(
                    batch.intermediate_times,
                    key=lambda value: abs(value - 0.5 * (batch.t0 + batch.t1)),
                )
                composed = integrate_model(
                    self.model, source_states, batch.graph, batch.t0, midpoint,
                    step_size=step_size, method=self.config.solver,
                    checkpoint_steps=(
                        self.config.gradient_checkpoint_interval
                        if self.config.gradient_checkpointing and self.model.training
                        else False
                    ),
                )
                composed = integrate_model(
                    self.model, composed, batch.graph, midpoint, batch.t1,
                    step_size=step_size, method=self.config.solver,
                    checkpoint_steps=(
                        self.config.gradient_checkpoint_interval
                        if self.config.gradient_checkpointing and self.model.training
                        else False
                    ),
                )
                semigroup_penalty = (prediction - composed).square().mean()
        return cardiogb_objective(
            prediction,
            batch.target_states,
            graph=batch.graph,
            residual=residual,
            residual_energy=trajectory_residual_energy,
            distribution_predicted=self._subsample(prediction),
            distribution_observed=self._subsample(batch.target_states),
            persistence_reference=self._subsample(source_states),
            stability_penalty=stability_penalty,
            semigroup_penalty=semigroup_penalty,
            regret_margin=self.config.regret_margin,
            weights=self.loss_weights,
        )

    def _select_transition_patches(
        self, transitions: Iterable[CrossSectionalTransition], training: bool
    ) -> list[CrossSectionalTransition]:
        items = list(transitions)
        curriculum = self.config.multi_horizon_curriculum_epochs
        if training and curriculum > 0 and items:
            durations = [abs(item.t1 - item.t0) for item in items]
            shortest, longest = min(durations), max(durations)
            fraction = min(1.0, (self.current_epoch + 1) / curriculum)
            cutoff = shortest + fraction * (longest - shortest)
            eligible = [
                item for item in items
                if abs(item.t1 - item.t0) <= cutoff + 1e-12
            ]
            items = eligible or [
                item for item in items if abs(item.t1 - item.t0) == shortest
            ]
        limit = self.config.patches_per_transition_per_epoch
        if limit is None:
            return items
        grouped: dict[str, list[CrossSectionalTransition]] = {}
        for item in items:
            grouped.setdefault(item.evaluation_group or item.name, []).append(item)
        selected = []
        for patches in grouped.values():
            if len(patches) <= limit:
                selected.extend(patches)
            elif training:
                order = torch.randperm(len(patches))[:limit].tolist()
                selected.extend(patches[index] for index in order)
            else:
                order = torch.linspace(0, len(patches) - 1, steps=limit).round().long().tolist()
                selected.extend(patches[index] for index in order)
        return selected

    @staticmethod
    def _combine_transition_patches(
        patches: list[CrossSectionalTransition],
    ) -> CrossSectionalTransition:
        if len(patches) == 1:
            return patches[0]
        first = patches[0]
        if any((item.t0, item.t1) != (first.t0, first.t1) for item in patches):
            raise ValueError("batched transition patches must share t0 and t1")
        source_states = torch.cat([item.source_states for item in patches], dim=0)
        edge_indices = []
        edge_attributes = []
        offset = 0
        for item in patches:
            if item.graph is None:
                offset += len(item.source_states)
                continue
            edge_indices.append(item.graph.edge_index + offset)
            edge_attributes.append(item.graph.edge_attr)
            offset += len(item.source_states)
        graph = None
        if edge_indices:
            graph = TorchGraph(torch.cat(edge_indices, dim=1), torch.cat(edge_attributes, dim=0))
        return CrossSectionalTransition(
            source_states=source_states,
            target_states=first.target_states,
            graph=graph,
            t0=first.t0,
            t1=first.t1,
            name=f"{first.evaluation_group or first.name}__batch_{len(patches)}",
            evaluation_group=first.evaluation_group,
            intermediate_times=first.intermediate_times,
        )

    def _batch_transition_patches(
        self, transitions: list[CrossSectionalTransition]
    ) -> list[CrossSectionalTransition]:
        batch_size = self.config.patch_batch_size
        if batch_size <= 1:
            return transitions
        grouped: dict[tuple[str, float, float], list[CrossSectionalTransition]] = {}
        for item in transitions:
            key = (item.evaluation_group or item.name, item.t0, item.t1)
            grouped.setdefault(key, []).append(item)
        batches = []
        for patches in grouped.values():
            for offset in range(0, len(patches), batch_size):
                batches.append(self._combine_transition_patches(patches[offset : offset + batch_size]))
        return batches

    def _epoch(self, transitions: Iterable[CrossSectionalTransition], training: bool) -> float:
        self.model.train(training)
        transitions = self._select_transition_patches(transitions, training)
        transitions = self._batch_transition_patches(transitions)
        losses: list[float] = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for transition in transitions:
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
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
        start_epoch: int = 0,
        initial_best: float = float("inf"),
    ) -> list[dict[str, float]]:
        train, validation = list(train), list(validation)
        transition_cache_bytes = sum(self._transition_nbytes(item) for item in [*train, *validation])
        if (
            self.device.type == "cuda"
            and self.config.cache_transitions_on_device
            and transition_cache_bytes <= self.config.max_device_transition_cache_bytes
        ):
            train = [self._move_transition(item) for item in train]
            validation = [self._move_transition(item) for item in validation]
        supports_warm_start = (
            hasattr(self.model, "mechanistic_model")
            and hasattr(self.model, "residual_model")
            and hasattr(self.model, "residual_enabled")
        )
        if start_epoch == 0 and supports_warm_start and self.config.warm_start_epochs > 0:
            self.model.residual_enabled = False
            for parameter in self.model.residual_model.parameters():
                parameter.requires_grad_(False)
            self.model.raw_residual_scale.requires_grad_(False)
            if hasattr(self.model, "raw_mechanistic_gate"):
                self.model.raw_mechanistic_gate.requires_grad_(False)
            for epoch in range(self.config.warm_start_epochs):
                self.current_epoch = 0
                train_loss = self._epoch(train, True)
                validation_loss = self._epoch(validation, False)
                self.history.append(
                    {
                        "epoch": float(epoch - self.config.warm_start_epochs),
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "warm_start": 1.0,
                    }
                )
            self.model.residual_enabled = True
            for parameter in self.model.residual_model.parameters():
                parameter.requires_grad_(True)
            self.model.raw_residual_scale.requires_grad_(True)
            if hasattr(self.model, "raw_mechanistic_gate"):
                self.model.raw_mechanistic_gate.requires_grad_(
                    bool(getattr(self.model, "learn_mechanistic_gate", True))
                )

        stopper = EarlyStopping(self.config.early_stopping_patience, best=initial_best)
        best_state = deepcopy(self.model.state_dict())
        for epoch in range(start_epoch, self.config.epochs):
            self.current_epoch = epoch
            train_loss = self._epoch(train, True)
            validation_loss = self._epoch(validation, False)
            self.history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "warm_start": 0.0,
                }
            )
            if validation_loss < stopper.best:
                best_state = deepcopy(self.model.state_dict())
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path, epoch, validation_loss)
            if stopper.update(validation_loss):
                break
            cooldown_interval = self.config.thermal_cooldown_every_epochs
            if (
                self.device.type == "cuda"
                and cooldown_interval is not None
                and cooldown_interval > 0
                and self.config.thermal_cooldown_seconds > 0
                and (epoch + 1) % cooldown_interval == 0
            ):
                torch.cuda.empty_cache()
                time.sleep(self.config.thermal_cooldown_seconds)
        self.model.load_state_dict(best_state)
        return self.history

    def resume_from_checkpoint(self, path: str | Path) -> tuple[int, float]:
        """Restore the best model/optimizer state and return the next epoch and best loss."""
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        return int(payload["epoch"]) + 1, float(payload["validation_loss"])

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
