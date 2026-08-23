from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from cardiogb.data.splits import grouped_split
from cardiogb.data.state_dataset import StateDataset
from cardiogb.losses import LossWeights
from cardiogb.models.factory import build_model
from cardiogb.training import CrossSectionalTrainer, TrainerConfig
from cardiogb.utils.config import load_yaml
from cardiogb.utils.device import resolve_device
from cardiogb.utils.io import atomic_json, export_table
from cardiogb.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only CardioGB model selection; never evaluates test units")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    dataset = StateDataset.load(args.data)
    model_config = load_yaml("configs/model.yaml")
    mech_config = load_yaml("configs/mechanistic_model.yaml")
    train_config = load_yaml("configs/train.yaml")
    device = resolve_device(train_config.get("device", "auto")).selected
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(train_config["cuda_memory_fraction"]))
    ode = model_config["ode"]
    rows = []
    for seed in args.seeds:
        seed_everything(seed)
        metadata = pd.DataFrame({"group": dataset.groups, "stage": dataset.times.astype(str)})
        train_mask, validation_mask, _, split = grouped_split(
            metadata, group_column="group", stage_column="stage", seed=seed
        )
        directory = args.output_dir / f"seed_{seed}"
        split.save(directory / "tables" / "split.json")
        model = build_model("cardiogb", model_config, mech_config)
        config = TrainerConfig(
            epochs=args.epochs,
            learning_rate=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            step_size=float(ode["step_size"]),
            solver=ode["solver"],
            early_stopping_patience=int(train_config["early_stopping_patience"]),
            max_distribution_samples=int(train_config["max_distribution_samples"]),
            max_ode_steps_per_transition=int(train_config["max_ode_steps_per_transition"]),
            mixed_precision=bool(train_config["mixed_precision"]),
            amp_dtype=str(train_config["amp_dtype"]),
            patches_per_transition_per_epoch=int(train_config["batching"]["patches_per_transition_per_epoch"]),
            patch_batch_size=int(train_config["batching"]["patch_batch_size"]),
            force_float32_integration=bool(train_config["force_float32_integration"]),
            mechanistic_learning_rate_scale=float(train_config["mechanistic_learning_rate_scale"]),
            gradient_checkpointing=bool(train_config["gradient_checkpointing"]),
            warm_start_epochs=int(train_config["warm_start_epochs"]),
        )
        weights = LossWeights(
            distribution=float(train_config["loss"]["lambda_distribution"]),
            moments=float(train_config["loss"]["lambda_moments"]),
            wasserstein=float(train_config["loss"]["lambda_wasserstein"]),
            spatial=float(train_config["loss"]["lambda_spatial"]),
            biology=float(train_config["loss"]["lambda_biology"]),
            residual=float(train_config["loss"]["lambda_residual"]),
        )
        trainer = CrossSectionalTrainer(model, device=device, config=config, loss_weights=weights)
        max_nodes = int(train_config["batching"]["max_nodes"])
        history = trainer.fit(
            dataset.transitions(mask=train_mask, k=int(model_config["graph"]["k"]), max_nodes=max_nodes),
            dataset.transitions(mask=validation_mask, k=int(model_config["graph"]["k"]), max_nodes=max_nodes),
            checkpoint_path=directory / "checkpoints" / "cardiogb.pt",
        )
        export_table(pd.DataFrame(history), directory / "metrics" / "validation_history.csv")
        joint = [entry for entry in history if entry.get("warm_start", 0.0) == 0.0]
        best = min(entry["validation_loss"] for entry in joint)
        row = {
            "seed": seed,
            "best_validation_loss": best,
            "epochs_total": len(history),
            "joint_epochs": len(joint),
            **{f"mechanistic_gate_{name}": float(value) for name, value in zip(dataset.state_names, trainer.model.mechanistic_gate().detach().cpu())},
            **{f"residual_scale_{name}": float(value) for name, value in zip(dataset.state_names, trainer.model.residual_scale().detach().cpu())},
        }
        rows.append(row)
        export_table(pd.DataFrame(rows), args.output_dir / "validation_selection.csv")
        atomic_json(
            {"status": "partial", "test_evaluated": False, "completed_seeds": [item["seed"] for item in rows]},
            args.output_dir / "run_manifest.json",
        )
        torch.cuda.empty_cache() if device == "cuda" else None
    atomic_json(
        {"status": "complete", "test_evaluated": False, "completed_seeds": args.seeds},
        args.output_dir / "run_manifest.json",
    )
    print(json.dumps({"status": "complete", "test_evaluated": False, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
