from __future__ import annotations

import argparse
import gc
import time
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from cardiogb.ablations import shuffle_graph
from cardiogb.data.splits import grouped_split
from cardiogb.data.state_dataset import StateDataset
from cardiogb.losses import LossWeights
from cardiogb.models import CardioGB, NeuralODEFunc
from cardiogb.models.factory import build_model
from cardiogb.models.mechanistic import MechanisticODE
from cardiogb.protocols import evaluate_transitions
from cardiogb.training import CrossSectionalTrainer, TrainerConfig
from cardiogb.utils.config import load_yaml
from cardiogb.utils.device import resolve_device
from cardiogb.utils.io import atomic_json, export_table
from cardiogb.utils.seed import seed_everything


VARIANTS = (
    "mechanism",
    "graph",
    "shuffled_graph",
    "residual_penalty",
    "state_definition",
)


def make_model(variant: str, model_config: dict, mechanistic_config: dict):
    if variant == "mechanism":
        return build_model("graph_neural_ode", model_config, mechanistic_config)
    if variant == "graph":
        gnn = model_config["gnn"]
        residual = NeuralODEFunc(
            state_dim=len(model_config["states"]),
            hidden_dim=int(gnn["hidden_dim"]),
            layers=int(gnn["layers"]),
            dropout=float(gnn["dropout"]),
        )
        return CardioGB(MechanisticODE.from_config(mechanistic_config), residual)
    return build_model("cardiogb", model_config, mechanistic_config)


def shuffled(transitions, seed: int):
    return [replace(item, graph=shuffle_graph(item.graph, seed=seed + index)) for index, item in enumerate(transitions)]


def run_variant(
    data_path: Path,
    variant: str,
    seed: int,
    epochs: int,
    output: Path,
) -> None:
    seed_everything(seed)
    dataset = StateDataset.load(data_path)
    model_config = load_yaml("configs/model.yaml")
    mechanistic_config = load_yaml("configs/mechanistic_model.yaml")
    train_config = load_yaml("configs/train.yaml")
    metadata = pd.DataFrame({"group": dataset.groups, "stage": dataset.times.astype(str)})
    train_mask, validation_mask, test_mask, split = grouped_split(
        metadata, group_column="group", stage_column="stage", seed=seed
    )
    split.save(output / "tables" / "split.json")
    k = int(model_config["graph"]["k"])
    max_nodes = int(train_config["batching"]["max_nodes"])
    train = dataset.transitions(mask=train_mask, k=k, max_nodes=max_nodes)
    validation = dataset.transitions(mask=validation_mask, k=k, max_nodes=max_nodes)
    test = dataset.transitions(mask=test_mask, k=k, max_nodes=max_nodes)
    if variant == "shuffled_graph":
        train = shuffled(train, seed)
        validation = shuffled(validation, seed + 100)
        test = shuffled(test, seed + 200)
    ode = model_config["ode"]
    trainer_config = TrainerConfig(
        epochs=epochs,
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        step_size=float(ode["step_size"]),
        solver=ode["solver"],
        early_stopping_patience=int(train_config["early_stopping_patience"]),
        max_distribution_samples=int(train_config["max_distribution_samples"]),
        max_ode_steps_per_transition=int(train_config["max_ode_steps_per_transition"]),
        mixed_precision=bool(train_config["mixed_precision"]),
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
        residual=0.0 if variant == "residual_penalty" else float(train_config["loss"]["lambda_residual"]),
    )
    device = resolve_device(train_config.get("device", "auto")).selected
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(train_config.get("cuda_memory_fraction", 0.65))
        )
    trainer = CrossSectionalTrainer(
        make_model(variant, model_config, mechanistic_config),
        device=device,
        config=trainer_config,
        loss_weights=weights,
    )
    history = trainer.fit(
        train,
        validation,
        checkpoint_path=output / "checkpoints" / f"{variant}.pt",
    )
    export_table(pd.DataFrame(history), output / "metrics" / "history.csv")
    rows = evaluate_transitions(
        trainer.model,
        test,
        device=device,
        step_size=float(ode["step_size"]),
        solver=ode["solver"],
        max_steps_per_transition=int(train_config["max_ode_steps_per_transition"]),
    )
    for row in rows:
        row.update({"ablation": variant, "seed": seed})
    export_table(pd.DataFrame(rows), output / "metrics" / "test.csv")
    atomic_json(
        {"status": "complete", "ablation": variant, "seed": seed, "epochs_completed": len(history), "device": device},
        output / "done.json",
    )


def aggregate(output: Path, reference: Path | None, seeds: list[int]) -> None:
    frames = []
    if reference is not None:
        for seed in seeds:
            path = reference / f"seed_{seed}" / "metrics" / "cardiogb_test.csv"
            if path.is_file():
                frame = pd.read_csv(path)
                frame["seed"] = seed
                frame["ablation"] = "full"
                frames.append(frame)
    for variant in VARIANTS:
        for seed in seeds:
            path = output / variant / f"seed_{seed}" / "metrics" / "test.csv"
            if path.is_file():
                frames.append(pd.read_csv(path))
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    export_table(combined, output / "tables" / "ablation_metrics.csv")
    metrics = [name for name in ("mmd", "moment_error", "sliced_wasserstein") if name in combined]
    summary = combined.groupby("ablation", observed=True)[metrics].agg(["mean", "std", "median"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    export_table(summary.reset_index(), output / "tables" / "ablation_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable real-data CardioGB ablation sweep")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rank-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20260815, 20260820)))
    parser.add_argument("--max-new-models", type=int, default=1)
    args = parser.parse_args()
    train_config = load_yaml("configs/train.yaml")
    cooldown = int(train_config.get("cooldown_seconds", 20))
    manifest = {
        "epochs_requested": args.epochs,
        "seeds": args.seeds,
        "ablations": list(VARIANTS),
        "batch_max_nodes": int(train_config["batching"]["max_nodes"]),
        "cuda_memory_fraction": float(train_config.get("cuda_memory_fraction", 0.65)),
        "cooldown_seconds": cooldown,
        "completed": [],
    }
    newly_trained = 0
    for variant in VARIANTS:
        data_path = args.rank_data if variant == "state_definition" else args.data
        for seed in args.seeds:
            run_dir = args.output_dir / variant / f"seed_{seed}"
            if not (run_dir / "done.json").is_file():
                run_variant(data_path, variant, seed, args.epochs, run_dir)
                newly_trained += 1
            manifest["completed"].append({"ablation": variant, "seed": seed})
            atomic_json(manifest, args.output_dir / "run_manifest.json")
            aggregate(args.output_dir, args.reference_dir, args.seeds)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if cooldown:
                time.sleep(cooldown)
            if args.max_new_models and newly_trained >= args.max_new_models:
                manifest["status"] = "partial"
                manifest["new_models_this_batch"] = newly_trained
                atomic_json(manifest, args.output_dir / "run_manifest.json")
                print(json.dumps(manifest, indent=2))
                return
    manifest["status"] = "complete"
    atomic_json(manifest, args.output_dir / "run_manifest.json")
    aggregate(args.output_dir, args.reference_dir, args.seeds)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
