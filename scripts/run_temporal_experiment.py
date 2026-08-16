from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cardiogb.data.state_dataset import StateDataset
from cardiogb.losses import LossWeights
from cardiogb.models.factory import build_model
from cardiogb.protocols import evaluate_transitions, extrapolation_masks, interpolation_masks
from cardiogb.training import CrossSectionalTrainer, TrainerConfig
from cardiogb.utils.config import load_yaml
from cardiogb.utils.device import resolve_device
from cardiogb.utils.io import atomic_json, export_table
from cardiogb.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal interpolation/extrapolation protocol")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", choices=["interpolation", "extrapolation"], required=True)
    parser.add_argument("--time", type=float, required=True, help="held-out stage or forecast cutoff")
    parser.add_argument("--validation-time", type=float)
    parser.add_argument("--model", default="cardiogb")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = StateDataset.load(args.data)
    model_config = load_yaml("configs/model.yaml")
    train_config = load_yaml("configs/train.yaml")
    seed = int(train_config["seed"])
    seed_everything(seed)
    if args.protocol == "interpolation":
        masks = interpolation_masks(dataset, args.time, args.validation_time)
        training = dataset.transitions(mask=masks["train"], k=int(model_config["graph"]["k"]))
        validation_time = float(dataset.times[masks["validation"]][0])
        prior_validation = max(time for time in np.unique(dataset.times[masks["train"]]) if time < validation_time)
        validation = [dataset.transition_between(prior_validation, validation_time, k=int(model_config["graph"]["k"]))]
        prior_test = max(time for time in np.unique(dataset.times[masks["train"]]) if time < args.time)
        testing = [dataset.transition_between(prior_test, args.time, k=int(model_config["graph"]["k"]))]
    else:
        masks = extrapolation_masks(dataset, args.time)
        training = dataset.transitions(mask=masks["train"], k=int(model_config["graph"]["k"]))
        prior = float(np.max(dataset.times[masks["train"]]))
        validation = [dataset.transition_between(prior, args.time, k=int(model_config["graph"]["k"]))]
        testing = [
            dataset.transition_between(args.time, future, k=int(model_config["graph"]["k"]))
            for future in sorted(np.unique(dataset.times[masks["test"]]))
        ]
    model = build_model(args.model, model_config, load_yaml("configs/mechanistic_model.yaml"))
    device = resolve_device(train_config.get("device", "auto")).selected
    ode = model_config["ode"]
    trainer = CrossSectionalTrainer(
        model,
        device=device,
        config=TrainerConfig(
            epochs=args.epochs,
            learning_rate=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            step_size=float(ode["step_size"]),
            solver=ode["solver"],
            early_stopping_patience=int(train_config["early_stopping_patience"]),
            max_distribution_samples=int(train_config["max_distribution_samples"]),
            max_ode_steps_per_transition=int(train_config["max_ode_steps_per_transition"]),
        ),
        loss_weights=LossWeights(
            distribution=float(train_config["loss"]["lambda_distribution"]),
            spatial=float(train_config["loss"]["lambda_spatial"]),
            biology=float(train_config["loss"]["lambda_biology"]),
            residual=float(train_config["loss"]["lambda_residual"]),
        ),
    )
    history = trainer.fit(
        training, validation, checkpoint_path=args.output_dir / "checkpoints" / f"{args.model}.pt"
    )
    metrics = evaluate_transitions(
        trainer.model,
        testing,
        device=device,
        step_size=float(ode["step_size"]),
        solver=ode["solver"],
        max_steps_per_transition=int(train_config["max_ode_steps_per_transition"]),
    )
    for row in metrics:
        row.update(model=args.model, protocol=args.protocol, held_out_or_cutoff=args.time)
    export_table(pd.DataFrame(history), args.output_dir / "metrics" / "history.csv")
    export_table(pd.DataFrame(metrics), args.output_dir / "metrics" / "test.csv")
    atomic_json(
        {"protocol": args.protocol, "time": args.time, "seed": seed, "masks": {key: int(value.sum()) for key, value in masks.items()}},
        args.output_dir / "tables" / "split.json",
    )
    print(json.dumps({"device": device, "epochs": len(history), "test_transitions": len(metrics)}))


if __name__ == "__main__":
    main()
