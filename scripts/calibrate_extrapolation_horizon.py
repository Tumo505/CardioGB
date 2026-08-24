from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cardiogb.data.state_dataset import StateDataset
from cardiogb.metrics.distributional import distribution_metrics
from cardiogb.metrics.forecast_calibration import (
    calibrate_forecast,
    fit_displacement_scale,
    horizon_displacement_scale,
)
from cardiogb.models.factory import build_model
from cardiogb.ode.integration import integrate_model
from cardiogb.utils.config import load_yaml
from cardiogb.utils.device import resolve_device
from cardiogb.utils.io import atomic_json, export_table

from run_manuscript_training import build_case


@torch.no_grad()
def grouped_predictions(model, transitions, *, device, step_size, solver, max_steps):
    model = model.to(device).eval()
    grouped = {}
    for transition in transitions:
        graph = transition.graph.to(device) if hasattr(transition.graph, "to") else transition.graph
        source = transition.source_states.to(device)
        effective_step = max(step_size, abs(transition.t1 - transition.t0) / max_steps)
        predicted = integrate_model(
            model, source, graph, transition.t0, transition.t1,
            step_size=effective_step, method=solver,
        ).cpu().numpy()
        key = transition.evaluation_group or transition.name
        entry = grouped.setdefault(
            key,
            {
                "t0": transition.t0,
                "t1": transition.t1,
                "source": [],
                "predicted": [],
                "target": transition.target_states.numpy(),
            },
        )
        entry["source"].append(transition.source_states.numpy())
        entry["predicted"].append(predicted)
    for entry in grouped.values():
        entry["source"] = np.concatenate(entry["source"], axis=0)
        entry["predicted"] = np.concatenate(entry["predicted"], axis=0)
    return grouped


def validation_parameters(grouped):
    source, predicted, observed, horizons = [], [], [], []
    for entry in grouped.values():
        source.append(entry["source"].mean(axis=0))
        predicted.append(entry["predicted"].mean(axis=0))
        observed.append(entry["target"].mean(axis=0))
        horizons.append(abs(float(entry["t1"] - entry["t0"])))
    return (
        fit_displacement_scale(np.asarray(source), np.asarray(predicted), np.asarray(observed)),
        max(horizons),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only conservative E3 forecast calibration")
    parser.add_argument("--data", type=Path, default=Path("data/processed/zebrafish_states.npz"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/e3_extrapolation_revised"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/e3_extrapolation_horizon_calibrated"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    dataset = StateDataset.load(args.data)
    model_config = load_yaml("configs/model.yaml")
    train_config = load_yaml("configs/train.yaml")
    mech_config = load_yaml("configs/mechanistic_model.yaml")
    device = resolve_device(args.device).selected
    k = int(model_config["graph"]["k"])
    max_nodes = int(train_config["batching"]["max_nodes"])
    ode = model_config["ode"]
    max_steps = int(train_config["max_ode_steps_per_transition"])
    source_manifest = json.loads((args.checkpoint_root / "run_manifest.json").read_text())
    if source_manifest.get("status") != "complete":
        raise RuntimeError("registered E3 checkpoint matrix must be complete before calibration")

    rows = []
    for item in source_manifest["completed"]:
        case, seed = float(item["case"]), int(item["seed"])
        checkpoint = args.checkpoint_root / f"case_{case:g}" / f"seed_{seed}" / "checkpoints" / "cardiogb.pt"
        model = build_model("cardiogb", model_config, mech_config)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
        _, validation, test, _ = build_case(dataset, "e3_extrapolation", case, k=k, max_nodes=max_nodes)
        validation_grouped = grouped_predictions(
            model, validation, device=device, step_size=float(ode["step_size"]),
            solver=ode["solver"], max_steps=max_steps,
        )
        validation_scale, maximum_validated_horizon = validation_parameters(validation_grouped)
        test_grouped = grouped_predictions(
            model, test, device=device, step_size=float(ode["step_size"]),
            solver=ode["solver"], max_steps=max_steps,
        )
        for transition, entry in test_grouped.items():
            horizon = abs(float(entry["t1"] - entry["t0"]))
            scale = horizon_displacement_scale(horizon, maximum_validated_horizon, validation_scale)
            calibrated = calibrate_forecast(entry["source"], entry["predicted"], scale)
            raw = distribution_metrics(entry["predicted"], entry["target"])
            conservative = distribution_metrics(calibrated, entry["target"])
            persistence = distribution_metrics(entry["source"], entry["target"])
            row = {
                "case": case, "seed": seed, "transition": transition,
                "t0": entry["t0"], "t1": entry["t1"], "horizon_days": horizon,
                "validation_displacement_scale": validation_scale,
                "maximum_validated_horizon_days": maximum_validated_horizon,
                "applied_displacement_scale": scale,
            }
            row.update(conservative)
            row.update({f"raw_{key}": value for key, value in raw.items()})
            row.update({f"persistence_{key}": value for key, value in persistence.items()})
            rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    export_table(frame, args.output_dir / "tables" / "all_metrics.csv")
    summary = frame.groupby(["case", "transition"], observed=True)[
        ["mmd", "moment_error", "sliced_wasserstein", "raw_mmd", "raw_moment_error",
         "raw_sliced_wasserstein", "persistence_mmd", "persistence_moment_error",
         "persistence_sliced_wasserstein"]
    ].agg(["mean", "std", "median", "count"])
    summary.columns = [f"{left}_{right}" for left, right in summary.columns]
    export_table(summary.reset_index(), args.output_dir / "tables" / "summary.csv")
    atomic_json(
        {
            "status": "complete", "source_protocol": "e3_extrapolation",
            "calibration": "validation least-squares displacement scale multiplied by validated-horizon/test-horizon",
            "test_outcomes_used_for_calibration": False, "rows": len(frame), "device": device,
        },
        args.output_dir / "run_manifest.json",
    )
    print(json.dumps({"status": "complete", "rows": len(frame), "device": device}, indent=2))


if __name__ == "__main__":
    main()
