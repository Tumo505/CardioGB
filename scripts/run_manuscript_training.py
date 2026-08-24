from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cardiogb.data.state_dataset import StateDataset
from cardiogb.losses import LossWeights
from cardiogb.models.factory import build_model
from cardiogb.protocols import evaluate_transitions, interpolation_masks
from cardiogb.training import CrossSectionalTrainer, TrainerConfig
from cardiogb.utils.config import load_yaml
from cardiogb.utils.device import resolve_device
from cardiogb.utils.io import atomic_json, export_table
from cardiogb.utils.seed import seed_everything

INTERPOLATION_STAGES = (0.5, 3.0, 14.0)  # configs/experiments/e2_interpolation.yaml
EXTRAPOLATION_CUTOFFS = (3.0, 7.0, 14.0)


def trainer_for(seed: int, epochs: int, gradient_checkpoint_interval: int = 2):
    seed_everything(seed)
    model_config = load_yaml("configs/model.yaml")
    train_config = load_yaml("configs/train.yaml")
    model = build_model("cardiogb", model_config, load_yaml("configs/mechanistic_model.yaml"))
    device = resolve_device(train_config.get("device", "auto")).selected
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(train_config.get("cuda_memory_fraction", 0.78)))
    ode = model_config["ode"]
    trainer = CrossSectionalTrainer(
        model,
        device=device,
        config=TrainerConfig(
            epochs=epochs,
            learning_rate=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            step_size=float(ode["step_size"]),
            solver=ode["solver"],
            early_stopping_patience=int(train_config["early_stopping_patience"]),
            max_distribution_samples=int(train_config["max_distribution_samples"]),
            max_ode_steps_per_transition=int(train_config["max_ode_steps_per_transition"]),
            mixed_precision=bool(train_config["mixed_precision"]),
            amp_dtype=str(train_config.get("amp_dtype", "bfloat16")),
            patches_per_transition_per_epoch=int(train_config["batching"]["patches_per_transition_per_epoch"]),
            thermal_cooldown_every_epochs=int(train_config["thermal_cooldown_every_epochs"]),
            thermal_cooldown_seconds=float(train_config["thermal_cooldown_seconds"]),
        patch_batch_size=int(train_config["batching"]["patch_batch_size"]),
        force_float32_integration=bool(train_config["force_float32_integration"]),
        mechanistic_learning_rate_scale=float(
            train_config["mechanistic_learning_rate_scale"]
        ),
        warm_start_epochs=int(train_config.get("warm_start_epochs", 0)),
        gradient_checkpointing=bool(train_config.get("gradient_checkpointing", True)),
        gradient_checkpoint_interval=gradient_checkpoint_interval,
        ),
        loss_weights=LossWeights(
            distribution=float(train_config["loss"]["lambda_distribution"]),
        moments=float(train_config["loss"]["lambda_moments"]),
        wasserstein=float(train_config["loss"]["lambda_wasserstein"]),
            spatial=float(train_config["loss"]["lambda_spatial"]),
            biology=float(train_config["loss"]["lambda_biology"]),
            residual=float(train_config["loss"]["lambda_residual"]),
        ),
    )
    return trainer, model_config, train_config, device


def replicate_number(group: str) -> int:
    match = re.search(r"_C([123])$", str(group))
    if match is None:
        raise ValueError(f"cannot extract registered replicate from group {group}")
    return int(match.group(1))


def build_case(dataset: StateDataset, protocol: str, case: float | int, *, k: int, max_nodes: int):
    times = np.unique(dataset.times)
    if protocol == "e2_interpolation":
        masks = interpolation_masks(dataset, float(case))
        train = dataset.transitions(mask=masks["train"], k=k, max_nodes=max_nodes)
        validation_time = float(np.unique(dataset.times[masks["validation"]])[0])
        train_times = np.unique(dataset.times[masks["train"]])
        validation_prior = float(np.max(train_times[train_times < validation_time]))
        test_prior = float(np.max(train_times[train_times < float(case)]))
        validation = dataset.transition_patches_between(
            validation_prior, validation_time, source_mask=masks["train"], target_mask=masks["validation"],
            k=k, max_nodes=max_nodes, name=f"{validation_prior:g}_to_{validation_time:g}",
        )
        test = dataset.transition_patches_between(
            test_prior, float(case), source_mask=masks["train"], target_mask=masks["test"],
            k=k, max_nodes=max_nodes, name=f"{test_prior:g}_to_{float(case):g}",
        )
        split = {
            "protocol": protocol, "held_out_stage": float(case), "validation_stage": validation_time,
            "train_stages": sorted(map(float, np.unique(dataset.times[masks["train"]]))),
            "test_stage": float(case),
        }
        return train, validation, test, split
    if protocol == "e3_extrapolation":
        cutoff = float(case)
        train_mask = dataset.times <= cutoff
        validation_replicate = 2
        fitting_mask = train_mask & np.asarray([replicate_number(x) != validation_replicate for x in dataset.groups])
        validation_mask = train_mask & np.asarray([replicate_number(x) == validation_replicate for x in dataset.groups])
        train = dataset.transitions(mask=fitting_mask, k=k, max_nodes=max_nodes)
        validation = dataset.transitions(mask=validation_mask, k=k, max_nodes=max_nodes)
        test = []
        for future in times[times > cutoff]:
            test.extend(dataset.transition_patches_between(cutoff, float(future), k=k, max_nodes=max_nodes, name=f"{cutoff:g}_to_{future:g}"))
        split = {
            "protocol": protocol, "cutoff_stage": cutoff,
            "train_stages": sorted(map(float, times[times <= cutoff])),
            "validation_replicate": validation_replicate,
            "forecast_stages": sorted(map(float, times[times > cutoff])),
            "test_uses_observed_cutoff_initial_state": True,
        }
        return train, validation, test, split
    if protocol == "e4_group_cv":
        fold = int(case)
        test_replicate = fold
        validation_replicate = fold % 3 + 1
        replicate = np.asarray([replicate_number(x) for x in dataset.groups])
        test_mask = replicate == test_replicate
        validation_mask = replicate == validation_replicate
        train_mask = ~(test_mask | validation_mask)
        split = {
            "protocol": protocol, "fold": fold, "train_replicate": int(np.unique(replicate[train_mask])[0]),
            "validation_replicate": validation_replicate, "test_replicate": test_replicate,
            "experimental_unit": "biological replicate (C1/C2/C3) within every stage",
        }
        return (
            dataset.transitions(mask=train_mask, k=k, max_nodes=max_nodes),
            dataset.transitions(mask=validation_mask, k=k, max_nodes=max_nodes),
            dataset.transitions(mask=test_mask, k=k, max_nodes=max_nodes),
            split,
        )
    raise ValueError(protocol)


def run_one(dataset, data_path: Path, protocol: str, case, seed: int, epochs: int, output: Path):
    checkpoint_interval = 1 if protocol in {"e2_interpolation", "e3_extrapolation"} else 2
    trainer, model_config, train_config, device = trainer_for(
        seed, epochs, gradient_checkpoint_interval=checkpoint_interval
    )
    k = int(model_config["graph"]["k"])
    max_nodes = int(train_config["batching"]["max_nodes"])
    train, validation, test, split = build_case(dataset, protocol, case, k=k, max_nodes=max_nodes)
    if not train or not validation or not test:
        raise ValueError(f"empty transition set for {protocol} case {case}")
    atomic_json({**split, "seed": seed}, output / "tables" / "split.json")
    checkpoint = output / "checkpoints" / "cardiogb.pt"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = trainer.fit(train, validation, checkpoint_path=checkpoint)
    export_table(pd.DataFrame(history), output / "metrics" / "history.csv")
    ode = model_config["ode"]
    metrics = evaluate_transitions(
        trainer.model, test, device=device, step_size=float(ode["step_size"]), solver=ode["solver"],
        max_steps_per_transition=int(train_config["max_ode_steps_per_transition"]),
    )
    for row in metrics:
        row.update(protocol=protocol, case=float(case), seed=seed, horizon_days=float(row["t1"] - row["t0"]))
    export_table(pd.DataFrame(metrics), output / "metrics" / "test.csv")
    parameters = trainer.model.mechanistic_model.constrained_parameters()
    parameter_rows = [
        {"parameter": name, "value": float(value.detach().cpu()), "seed": seed, "case": case}
        for name, value in parameters.items()
    ]
    parameter_rows.extend(
        {"parameter": f"mechanistic_gate_{state}", "value": float(value), "seed": seed, "case": case}
        for state, value in zip(dataset.state_names, trainer.model.mechanistic_gate().detach().cpu())
    )
    parameter_rows.extend(
        {"parameter": f"residual_scale_{state}", "value": float(value), "seed": seed, "case": case}
        for state, value in zip(dataset.state_names, trainer.model.residual_scale().detach().cpu())
    )
    export_table(pd.DataFrame(parameter_rows), output / "tables" / "parameters.csv")
    atomic_json(
        {
            "status": "complete", "protocol": protocol, "case": case, "seed": seed,
            "epochs_requested": epochs, "epochs_completed": len(history), "device": device,
            "train_transitions": len(train), "validation_transitions": len(validation), "test_transitions": len(metrics),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
            "max_nodes": max_nodes,
        },
        output / "done.json",
    )


def aggregate(root: Path, protocol: str):
    frames, parameters = [], []
    for path in root.glob("case_*/seed_*/metrics/test.csv"):
        frames.append(pd.read_csv(path))
    for path in root.glob("case_*/seed_*/tables/parameters.csv"):
        parameters.append(pd.read_csv(path))
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        export_table(combined, root / "tables" / "all_metrics.csv")
        metrics = [x for x in ("mmd", "moment_error", "sliced_wasserstein") if x in combined]
        summary = combined.groupby(["case", "transition"], observed=True)[metrics].agg(["mean", "std", "median", "count"])
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        export_table(summary.reset_index(), root / "tables" / "summary.csv")
    if parameters:
        export_table(pd.concat(parameters, ignore_index=True), root / "tables" / "all_parameters.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable manuscript-grade E2/E3/E4 CardioGB matrix")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", choices=["e2_interpolation", "e3_extrapolation", "e4_group_cv"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20260815, 20260820)))
    parser.add_argument("--max-new-runs", type=int, default=0, help="zero means run the whole remaining matrix")
    args = parser.parse_args()
    dataset = StateDataset.load(args.data)
    cases = {
        "e2_interpolation": INTERPOLATION_STAGES,
        "e3_extrapolation": EXTRAPOLATION_CUTOFFS,
        "e4_group_cv": (1, 2, 3),
    }[args.protocol]
    config = load_yaml("configs/train.yaml")
    manifest = {
        "status": "partial", "protocol": args.protocol, "cases": list(cases), "seeds": args.seeds,
        "epochs_requested": args.epochs, "max_nodes": int(config["batching"]["max_nodes"]), "completed": [],
    }
    new_runs = 0
    for case in cases:
        label = f"{float(case):g}".replace(".", "p")
        for seed in args.seeds:
            directory = args.output_dir / f"case_{label}" / f"seed_{seed}"
            trained_this_iteration = not (directory / "done.json").is_file()
            if trained_this_iteration:
                run_one(dataset, args.data, args.protocol, case, seed, args.epochs, directory)
                new_runs += 1
            manifest["completed"].append({"case": case, "seed": seed})
            aggregate(args.output_dir, args.protocol)
            atomic_json(manifest, args.output_dir / "run_manifest.json")
            if trained_this_iteration:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(float(config.get("cooldown_seconds", 20)))
            if args.max_new_runs and new_runs >= args.max_new_runs:
                manifest["new_runs_this_invocation"] = new_runs
                atomic_json(manifest, args.output_dir / "run_manifest.json")
                print(json.dumps(manifest, indent=2))
                return
    manifest["status"] = "complete"
    manifest["new_runs_this_invocation"] = new_runs
    aggregate(args.output_dir, args.protocol)
    atomic_json(manifest, args.output_dir / "run_manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()