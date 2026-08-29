from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from cardiogb.data.splits import grouped_split
from cardiogb.data.state_dataset import StateDataset
from cardiogb.losses import LossWeights
from cardiogb.models.factory import build_model
from cardiogb.protocols import evaluate_transitions
from cardiogb.training import CrossSectionalTrainer, TrainerConfig
from cardiogb.utils.config import load_yaml
from cardiogb.utils.device import resolve_device
from cardiogb.utils.io import export_table
from cardiogb.utils.seed import seed_everything


def train_model(
    data_path: Path,
    model_name: str,
    output_dir: Path,
    *,
    epochs_override: int | None = None,
    seed_override: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    model_config = load_yaml("configs/model.yaml")
    mech_config = load_yaml("configs/mechanistic_model.yaml")
    train_config = load_yaml("configs/train.yaml")
    seed = int(train_config["seed"] if seed_override is None else seed_override)
    seed_everything(seed)
    dataset = StateDataset.load(data_path)
    metadata = pd.DataFrame({"group": dataset.groups, "stage": dataset.times.astype(str)})
    train_mask, validation_mask, test_mask, definition = grouped_split(
        metadata,
        group_column="group",
        stage_column="stage",
        seed=seed,
    )
    definition.save(output_dir / "tables" / "split.json")
    model = build_model(model_name, model_config, mech_config)
    device = resolve_device(train_config.get("device", "auto")).selected
    cuda_memory_fraction = float(
        os.environ.get(
            "CARDIOGB_CUDA_MEMORY_FRACTION",
            train_config.get("cuda_memory_fraction", 0.65),
        )
    )
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            cuda_memory_fraction
        )
    ode = model_config["ode"]
    max_nodes = int(train_config["batching"]["max_nodes"])
    multi_horizon = train_config.get("multi_horizon", {})
    trainer = CrossSectionalTrainer(
        model,
        device=device,
        config=TrainerConfig(
            epochs=int(epochs_override or train_config["epochs"]),
            learning_rate=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            step_size=float(ode["step_size"]),
            solver=ode["solver"],
            early_stopping_patience=int(train_config["early_stopping_patience"]),
            max_distribution_samples=int(train_config["max_distribution_samples"]),
            max_ode_steps_per_transition=int(train_config["max_ode_steps_per_transition"]),
            mixed_precision=bool(train_config["mixed_precision"]),
            amp_dtype=str(train_config.get("amp_dtype", "bfloat16")),
            patches_per_transition_per_epoch=int(
                train_config["batching"]["patches_per_transition_per_epoch"]
            ),
            thermal_cooldown_every_epochs=int(
                train_config["thermal_cooldown_every_epochs"]
            ),
            thermal_cooldown_seconds=float(train_config["thermal_cooldown_seconds"]),
            patch_batch_size=int(
                os.environ.get(
                    "CARDIOGB_PATCH_BATCH_SIZE",
                    train_config["batching"]["patch_batch_size"],
                )
            ),
            force_float32_integration=bool(train_config["force_float32_integration"]),
            mechanistic_learning_rate_scale=float(
                train_config["mechanistic_learning_rate_scale"]
            ),
            warm_start_epochs=int(train_config.get("warm_start_epochs", 0)),
            gradient_checkpointing=bool(train_config.get("gradient_checkpointing", True)),
            gradient_checkpoint_interval=int(train_config.get("gradient_checkpoint_interval", 1)),
            multi_horizon_curriculum_epochs=int(multi_horizon.get("curriculum_epochs", 0)),
            stability_velocity_target=float(
                multi_horizon.get("stability_velocity_target", 0.4)
            ),
            regret_margin=float(multi_horizon.get("regret_margin", 0.0)),
        ),
        loss_weights=LossWeights(
            distribution=float(train_config["loss"]["lambda_distribution"]),
            moments=float(train_config["loss"]["lambda_moments"]),
            wasserstein=float(train_config["loss"]["lambda_wasserstein"]),
            spatial=float(train_config["loss"]["lambda_spatial"]),
            biology=float(train_config["loss"]["lambda_biology"]),
            residual=float(train_config["loss"]["lambda_residual"]),
            persistence_regret=float(
                train_config["loss"].get("lambda_persistence_regret", 0.0)
            ),
            stability=float(train_config["loss"].get("lambda_stability", 0.0)),
            semigroup=float(train_config["loss"].get("lambda_semigroup", 0.0)),
        ),
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    checkpoint_path = output_dir / "checkpoints" / f"{model_name}.pt"
    start_epoch = 0
    initial_best = float("inf")
    if resume and checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        trainer.model.load_state_dict(payload["model"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        initial_best = float(payload["validation_loss"])
    history = trainer.fit(
        dataset.transitions(
            mask=train_mask,
            k=int(model_config["graph"]["k"]),
            max_nodes=max_nodes,
            adjacent_only=not bool(multi_horizon.get("enabled", False)),
        ),
        dataset.transitions(
            mask=validation_mask,
            k=int(model_config["graph"]["k"]),
            max_nodes=max_nodes,
        ),
        checkpoint_path=checkpoint_path,
        start_epoch=start_epoch,
        initial_best=initial_best,
    )
    export_table(pd.DataFrame(history), output_dir / "metrics" / f"{model_name}_history.csv")
    evaluation = evaluate_transitions(
        trainer.model,
        dataset.transitions(mask=test_mask, k=int(model_config["graph"]["k"]), max_nodes=max_nodes),
        device=device,
        step_size=float(ode["step_size"]),
        solver=ode["solver"],
        max_steps_per_transition=int(train_config["max_ode_steps_per_transition"]),
    )
    for row in evaluation:
        row["model"] = model_name
    metrics_path = output_dir / "metrics" / f"{model_name}_test.csv"
    export_table(pd.DataFrame(evaluation), metrics_path)
    if hasattr(trainer.model, "mechanistic_model"):
        parameters = trainer.model.mechanistic_model.constrained_parameters()
    elif hasattr(trainer.model, "constrained_parameters"):
        parameters = trainer.model.constrained_parameters()
    else:
        parameters = {}
    if parameters:
        parameter_rows = [
            {"model": model_name, "parameter": name, "value": float(value.detach().cpu())}
            for name, value in parameters.items()
        ]
        if hasattr(trainer.model, "mechanistic_gate"):
            parameter_rows.extend(
                {"model": model_name, "parameter": f"mechanistic_gate_{state}", "value": float(value)}
                for state, value in zip(dataset.state_names, trainer.model.mechanistic_gate().detach().cpu())
            )
            parameter_rows.extend(
                {"model": model_name, "parameter": f"residual_scale_{state}", "value": float(value)}
                for state, value in zip(dataset.state_names, trainer.model.residual_scale().detach().cpu())
            )
        export_table(
            pd.DataFrame(parameter_rows), output_dir / "tables" / f"{model_name}_parameters.csv"
        )
    return {
        "device": device,
        "seed": seed,
        "epochs": len(history),
        "test_nodes": int(test_mask.sum()),
        "test_transitions": len(evaluation),
        "metrics": str(metrics_path),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
        ),
        "batch_max_nodes": max_nodes,
        "patch_batch_size": trainer.config.patch_batch_size,
        "cuda_memory_fraction": cuda_memory_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="cardiogb")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.model == "persistence":
        raise ValueError("Use train_baseline.py for the parameter-free persistence baseline")
    print(json.dumps(train_model(
        args.data,
        args.model,
        args.output_dir,
        epochs_override=args.epochs,
        seed_override=args.seed,
        resume=args.resume,
    )))


if __name__ == "__main__":
    main()
