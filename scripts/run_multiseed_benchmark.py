from __future__ import annotations

import argparse
import gc
import time
import json
from pathlib import Path

import pandas as pd
import torch

from cardiogb.data.splits import grouped_split
from cardiogb.data.state_dataset import StateDataset
from cardiogb.metrics import distribution_metrics
from cardiogb.utils.io import atomic_json, export_table
from train_cardiogb import train_model


LEARNED_MODELS = ("mechanistic_ode", "neural_ode", "graph_neural_ode", "cardiogb")


def persistence_for_seed(dataset: StateDataset, seed: int, output: Path) -> None:
    metadata = pd.DataFrame({"group": dataset.groups, "stage": dataset.times.astype(str)})
    _, _, test_mask, split = grouped_split(
        metadata, group_column="group", stage_column="stage", seed=seed
    )
    split.save(output / "tables" / "split.json")
    rows = []
    for transition in dataset.transitions(mask=test_mask):
        rows.append(
            {
                "seed": seed,
                "model": "persistence",
                "transition": transition.name,
                **distribution_metrics(
                    transition.source_states.numpy(), transition.target_states.numpy()
                ),
            }
        )
    export_table(pd.DataFrame(rows), output / "metrics" / "persistence_test.csv")


def aggregate(output: Path, seeds: list[int]) -> None:
    frames = []
    for seed in seeds:
        directory = output / f"seed_{seed}"
        for model in ("persistence", *LEARNED_MODELS):
            path = directory / "metrics" / f"{model}_test.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            frame["seed"] = seed
            frame["model"] = model
            frames.append(frame)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    export_table(combined, output / "tables" / "benchmark_metrics.csv")
    metrics = [name for name in ("mmd", "moment_error", "sliced_wasserstein") if name in combined]
    summary = combined.groupby("model", observed=True)[metrics].agg(["mean", "std", "median"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    export_table(summary.reset_index(), output / "tables" / "benchmark_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable grouped-holdout multi-seed benchmark")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20260815, 20260820)))
    parser.add_argument(
        "--max-new-models",
        type=int,
        default=1,
        help="Stop after this many newly trained learned models; use 0 for no limit.",
    )
    args = parser.parse_args()
    dataset = StateDataset.load(args.data)
    train_config = __import__("cardiogb.utils.config", fromlist=["load_yaml"]).load_yaml("configs/train.yaml")
    cooldown = int(train_config.get("cooldown_seconds", 20))
    manifest = {
        "epochs_requested": args.epochs,
        "batch_max_nodes": int(train_config["batching"]["max_nodes"]),
        "cuda_memory_fraction": float(train_config.get("cuda_memory_fraction", 0.65)),
        "cooldown_seconds": cooldown,
        "seeds": args.seeds,
        "models": ["persistence", *LEARNED_MODELS],
        "split": "grouped biological-unit, stratified by stage",
        "completed": [],
    }
    newly_trained = 0
    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        persistence = seed_dir / "metrics" / "persistence_test.csv"
        if not persistence.is_file():
            persistence_for_seed(dataset, seed, seed_dir)
        manifest["completed"].append({"seed": seed, "model": "persistence"})
        atomic_json(manifest, args.output_dir / "run_manifest.json")
        for model in LEARNED_MODELS:
            metrics = seed_dir / "metrics" / f"{model}_test.csv"
            if not metrics.is_file():
                train_model(
                    args.data,
                    model,
                    seed_dir,
                    epochs_override=args.epochs,
                    seed_override=seed,
                )
                newly_trained += 1
            manifest["completed"].append({"seed": seed, "model": model})
            atomic_json(manifest, args.output_dir / "run_manifest.json")
            aggregate(args.output_dir, args.seeds)
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
    aggregate(args.output_dir, args.seeds)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
