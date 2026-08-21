from __future__ import annotations

import argparse
import gc
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cardiogb.data.splits import grouped_split
from cardiogb.data.state_dataset import StateDataset
from cardiogb.losses import LossWeights
from cardiogb.metrics.distributional import distribution_metrics
from cardiogb.metrics.uncertainty import distribution_mean_calibration
from cardiogb.models.factory import build_model
from cardiogb.training.ensemble import predict_ensemble
from cardiogb.training.trainer import CrossSectionalTrainer, TrainerConfig
from cardiogb.utils.config import load_yaml
from cardiogb.utils.device import resolve_device
from cardiogb.utils.io import atomic_json, export_table
from cardiogb.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and calibrate a CardioGB deep ensemble")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-members", type=int, default=1)
    args = parser.parse_args()
    dataset = StateDataset.load(args.data)
    model_config = load_yaml("configs/model.yaml")
    mech_config = load_yaml("configs/mechanistic_model.yaml")
    train_config = load_yaml("configs/train.yaml")
    metadata = pd.DataFrame({"group": dataset.groups, "stage": dataset.times.astype(str)})
    seed = int(train_config["seed"] if args.seed is None else args.seed)
    train_mask, validation_mask, test_mask, split = grouped_split(
        metadata,
        group_column="group",
        stage_column="stage",
        seed=seed,
    )
    split.save(args.output_dir / "tables" / "split.json")
    ode = model_config["ode"]
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        step_size=float(ode["step_size"]),
        solver=ode["solver"],
        early_stopping_patience=int(train_config["early_stopping_patience"]),
        max_distribution_samples=int(train_config["max_distribution_samples"]),
        max_ode_steps_per_transition=int(train_config["max_ode_steps_per_transition"]),
        patches_per_transition_per_epoch=int(
            train_config["batching"]["patches_per_transition_per_epoch"]
        ),
        thermal_cooldown_every_epochs=int(
            train_config["thermal_cooldown_every_epochs"]
        ),
        thermal_cooldown_seconds=float(train_config["thermal_cooldown_seconds"]),
    )
    weights = LossWeights(
        distribution=float(train_config["loss"]["lambda_distribution"]),
        spatial=float(train_config["loss"]["lambda_spatial"]),
        biology=float(train_config["loss"]["lambda_biology"]),
        residual=float(train_config["loss"]["lambda_residual"]),
    )
    device = resolve_device(train_config.get("device", "auto")).selected
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(train_config.get("cuda_memory_fraction", 0.65))
        )
    max_nodes = int(train_config["batching"]["max_nodes"])
    cooldown = int(train_config.get("cooldown_seconds", 20))
    train = dataset.transitions(
        mask=train_mask, k=int(model_config["graph"]["k"]), max_nodes=max_nodes
    )
    validation = dataset.transitions(
        mask=validation_mask, k=int(model_config["graph"]["k"]), max_nodes=max_nodes
    )
    models = []
    history_rows = []
    member_seeds = []
    newly_trained = 0
    all_members_ready = all(
        (args.output_dir / "checkpoints" / f"member_{member:02d}.pt").is_file()
        and (args.output_dir / "members" / f"member_{member:02d}.json").is_file()
        for member in range(args.members)
    )
    for member in range(args.members):
        member_seed = seed + member
        member_seeds.append(member_seed)
        checkpoint = args.output_dir / "checkpoints" / f"member_{member:02d}.pt"
        marker = args.output_dir / "members" / f"member_{member:02d}.json"
        history_path = args.output_dir / "members" / f"member_{member:02d}_history.csv"
        seed_everything(member_seed)
        model = build_model("cardiogb", model_config, mech_config)
        if marker.is_file() and checkpoint.is_file():
            if all_members_ready:
                payload = torch.load(checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(payload["model"])
                model.to(device)
                models.append(model)
            if history_path.is_file():
                history_rows.extend(pd.read_csv(history_path).to_dict("records"))
        else:
            trainer = CrossSectionalTrainer(
                model, device=device, config=trainer_config, loss_weights=weights
            )
            history = trainer.fit(train, validation, checkpoint_path=checkpoint)
            model = trainer.model
            newly_trained += 1
            member_history = pd.DataFrame(
                [{"member": member, "member_seed": member_seed, **row} for row in history]
            )
            export_table(member_history, history_path)
            history_rows.extend(member_history.to_dict("records"))
            atomic_json(
                {
                    "status": "complete",
                    "member": member,
                    "seed": member_seed,
                    "epochs_completed": len(history),
                },
                marker,
            )
            models.append(model)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if cooldown:
                time.sleep(cooldown)
        if args.max_new_members and newly_trained >= args.max_new_members:
            export_table(pd.DataFrame(history_rows), args.output_dir / "metrics" / "history.csv")
            members_completed = sum(
                (args.output_dir / "members" / f"member_{index:02d}.json").is_file()
                for index in range(args.members)
            )
            atomic_json(
                {
                    "status": "partial",
                    "device": device,
                    "members_requested": args.members,
                    "members_completed": members_completed,
                    "base_seed": seed,
                    "epochs_requested": args.epochs,
                    "batch_max_nodes": max_nodes,
                    "cuda_memory_fraction": float(
                        train_config.get("cuda_memory_fraction", 0.65)
                    ),
                    "cooldown_seconds": cooldown,
                },
                args.output_dir / "run_manifest.json",
            )
            print(json.dumps({"status": "partial", "members_completed": members_completed}))
            return
    export_table(pd.DataFrame(history_rows), args.output_dir / "metrics" / "history.csv")
    grouped = {}
    for transition in dataset.transitions(
        mask=test_mask, k=int(model_config["graph"]["k"]), max_nodes=max_nodes
    ):
        step_size = max(
            float(ode["step_size"]),
            abs(transition.t1 - transition.t0) / int(train_config["max_ode_steps_per_transition"]),
        )
        mean, _, stacked = predict_ensemble(
            models,
            transition.source_states,
            transition.graph,
            transition.t0,
            transition.t1,
            step_size=step_size,
            method=ode["solver"],
        )
        key = transition.evaluation_group or transition.name
        entry = grouped.setdefault(
            key,
            {"means": [], "ensembles": [], "target": transition.target_states.numpy()},
        )
        entry["means"].append(mean.numpy())
        entry["ensembles"].append(stacked.numpy())
    rows = []
    for name, entry in grouped.items():
        mean = np.concatenate(entry["means"], axis=0)
        stacked = np.concatenate(entry["ensembles"], axis=1)
        calibration = distribution_mean_calibration(stacked, entry["target"])
        rows.append(
            {
                "transition": name,
                **distribution_metrics(mean, entry["target"]),
                "state_mean_interval_coverage": calibration["coverage"],
            }
        )
        atomic_json(
            calibration,
            args.output_dir / "uncertainty" / f"{name}_calibration.json",
        )
    export_table(pd.DataFrame(rows), args.output_dir / "metrics" / "ensemble_test.csv")
    atomic_json(
        {
            "status": "complete",
            "device": device,
            "members": args.members,
            "base_seed": seed,
            "member_seeds": member_seeds,
            "epochs_requested": args.epochs,
            "batch_max_nodes": max_nodes,
            "cuda_memory_fraction": float(train_config.get("cuda_memory_fraction", 0.65)),
            "cooldown_seconds": cooldown,
            "transitions": len(rows),
        },
        args.output_dir / "run_manifest.json",
    )
    print(json.dumps({"device": device, "members": args.members, "transitions": len(rows)}))


if __name__ == "__main__":
    main()
