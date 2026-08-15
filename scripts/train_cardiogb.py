from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

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


def train_model(data_path: Path, model_name: str, output_dir: Path) -> dict[str, object]:
    model_config = load_yaml("configs/model.yaml")
    mech_config = load_yaml("configs/mechanistic_model.yaml")
    train_config = load_yaml("configs/train.yaml")
    seed_everything(int(train_config["seed"]))
    dataset = StateDataset.load(data_path)
    metadata = pd.DataFrame({"group": dataset.groups, "stage": dataset.times.astype(str)})
    train_mask, validation_mask, test_mask, definition = grouped_split(
        metadata,
        group_column="group",
        stage_column="stage",
        seed=int(train_config["seed"]),
    )
    definition.save(output_dir / "tables" / "split.json")
    model = build_model(model_name, model_config, mech_config)
    device = resolve_device(train_config.get("device", "auto")).selected
    ode = model_config["ode"]
    trainer = CrossSectionalTrainer(
        model,
        device=device,
        config=TrainerConfig(
            epochs=int(train_config["epochs"]),
            learning_rate=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            step_size=float(ode["step_size"]),
            solver=ode["solver"],
            early_stopping_patience=int(train_config["early_stopping_patience"]),
            mixed_precision=bool(train_config["mixed_precision"]),
        ),
        loss_weights=LossWeights(
            distribution=float(train_config["loss"]["lambda_distribution"]),
            spatial=float(train_config["loss"]["lambda_spatial"]),
            biology=float(train_config["loss"]["lambda_biology"]),
            residual=float(train_config["loss"]["lambda_residual"]),
        ),
    )
    history = trainer.fit(
        dataset.transitions(mask=train_mask, k=int(model_config["graph"]["k"])),
        dataset.transitions(mask=validation_mask, k=int(model_config["graph"]["k"])),
        checkpoint_path=output_dir / "checkpoints" / f"{model_name}.pt",
    )
    export_table(pd.DataFrame(history), output_dir / "metrics" / f"{model_name}_history.csv")
    evaluation = evaluate_transitions(
        trainer.model,
        dataset.transitions(mask=test_mask, k=int(model_config["graph"]["k"])),
        device=device,
        step_size=float(ode["step_size"]),
        solver=ode["solver"],
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
        export_table(
            pd.DataFrame(parameter_rows), output_dir / "tables" / f"{model_name}_parameters.csv"
        )
    return {
        "device": device,
        "epochs": len(history),
        "test_nodes": int(test_mask.sum()),
        "test_transitions": len(evaluation),
        "metrics": str(metrics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="cardiogb")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.model == "persistence":
        raise ValueError("Use train_baseline.py for the parameter-free persistence baseline")
    print(json.dumps(train_model(args.data, args.model, args.output_dir)))


if __name__ == "__main__":
    main()
