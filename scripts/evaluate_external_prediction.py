from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cardiogb.data.splits import grouped_split
from cardiogb.data.state_dataset import StateDataset
from cardiogb.metrics.distributional import distribution_metrics
from cardiogb.models.factory import build_model
from cardiogb.ode.integration import integrate_model
from cardiogb.training.ensemble_weighting import fit_simplex_weights
from cardiogb.training.robust_ensemble import aggregate_members, select_aggregation
from cardiogb.utils.config import load_yaml
from cardiogb.utils.device import resolve_device
from cardiogb.utils.io import atomic_json, export_table


def load_members(directory: Path, device: str, expected_members: int = 5) -> tuple[list[torch.nn.Module], np.ndarray, list[float]]:
    model_config = load_yaml("configs/model.yaml")
    mech_config = load_yaml("configs/mechanistic_model.yaml")
    checkpoints = sorted(directory.glob("member_*.pt"))
    if len(checkpoints) != expected_members:
        raise ValueError(f"expected exactly {expected_members} ensemble checkpoints in {directory}, found {len(checkpoints)}")
    models, losses = [], []
    for checkpoint_path in checkpoints:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model("cardiogb", model_config, mech_config)
        model.load_state_dict(payload["model"])
        models.append(model.to(device).eval())
        losses.append(float(payload.get("validation_loss", np.nan)))
    values = np.asarray(losses, dtype=float)
    if np.isfinite(values).all() and np.ptp(values) > 0:
        temperature = max(float(np.median(values)), 1e-8)
        logits = -(values - values.min()) / temperature
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
    else:
        weights = np.repeat(1.0 / len(models), len(models))
    return models, weights, losses


@torch.no_grad()
def member_predictions(models, transition, *, step_size: float, solver: str, max_steps: int) -> np.ndarray:
    step = max(step_size, abs(transition.t1 - transition.t0) / max_steps)
    outputs = []
    for model in models:
        device = next(model.parameters()).device
        graph = transition.graph.to(device) if hasattr(transition.graph, "to") else transition.graph
        outputs.append(
            integrate_model(
                model,
                transition.source_states.to(device),
                graph,
                transition.t0,
                transition.t1,
                step_size=step,
                method=solver,
            ).cpu().numpy()
        )
    return np.stack(outputs)


def grouped_predictions(models, transitions, weights, ode, max_steps, aggregation_method="simplex"):
    grouped = {}
    for transition in transitions:
        prediction = member_predictions(
            models,
            transition,
            step_size=float(ode["step_size"]),
            solver=ode["solver"],
            max_steps=max_steps,
        )
        key = transition.evaluation_group or transition.name
        item = grouped.setdefault(
            key,
            {"members": [], "sources": [], "target": transition.target_states.numpy(), "t0": transition.t0, "t1": transition.t1},
        )
        item["members"].append(prediction)
        item["sources"].append(transition.source_states.numpy())
    for item in grouped.values():
        item["members"] = np.concatenate(item["members"], axis=1)
        item["sources"] = np.concatenate(item["sources"], axis=0)
        item["mean"] = aggregate_members(item["members"], aggregation_method, weights)
        item["std"] = np.sqrt(
            np.tensordot(weights, (item["members"] - item["mean"]) ** 2, axes=(0, 0))
        )
        item["aggregation_method"] = aggregation_method
    return grouped


def conformal_factor(grouped: dict, weights: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = []
    absolute_scores = []
    for item in grouped.values():
        predicted_mean = item["mean"].mean(axis=0)
        member_means = item["members"].mean(axis=1)
        spread = np.sqrt(np.tensordot(weights, (member_means - predicted_mean) ** 2, axes=(0, 0)))
        observed_mean = item["target"].mean(axis=0)
        absolute = np.abs(observed_mean - predicted_mean)
        scores.append(absolute / np.maximum(spread, 1e-6))
        absolute_scores.append(absolute)
    scores = np.asarray(scores)
    absolute_scores = np.asarray(absolute_scores)
    quantile = min(1.0, np.ceil((len(scores) + 1) * confidence) / len(scores))
    return (
        np.quantile(scores, quantile, axis=0, method="higher"),
        np.quantile(absolute_scores, quantile, axis=0, method="higher"),
        scores,
        absolute_scores,
    )


def evaluate(grouped: dict, weights: np.ndarray, scale: float, additive_radius: float, state_names, protocol: str):
    metric_rows, state_rows = [], []
    for name, item in grouped.items():
        member_means = item["members"].mean(axis=1)
        weighted_state_mean = aggregate_members(member_means, item["aggregation_method"], weights)
        weighted_state_std = np.sqrt(
            np.tensordot(weights, (member_means - weighted_state_mean) ** 2, axes=(0, 0))
        )
        target_mean = item["target"].mean(axis=0)
        multiplicative_radius = scale * np.maximum(weighted_state_std, 1e-6)
        raw_radius = 1.96 * weighted_state_std
        radius = np.asarray(additive_radius, dtype=float)
        if radius.ndim == 0:
            radius = np.repeat(radius, len(target_mean))
        if radius.shape != target_mean.shape:
            raise ValueError(f"conformal radius shape {radius.shape} does not match state means {target_mean.shape}")
        covered = np.abs(target_mean - weighted_state_mean) <= radius
        raw_covered = np.abs(target_mean - weighted_state_mean) <= raw_radius
        multiplicative_covered = np.abs(target_mean - weighted_state_mean) <= multiplicative_radius
        weighted_metrics = distribution_metrics(item["mean"], item["target"])
        equal_metrics = distribution_metrics(item["members"].mean(axis=0), item["target"])
        persistence_metrics = distribution_metrics(item["sources"], item["target"])
        metric_rows.append(
            {
                "protocol": protocol,
                "transition": name,
                "t0": item["t0"],
                "t1": item["t1"],
                "horizon_days": item["t1"] - item["t0"],
                "raw_coverage": float(raw_covered.mean()),
                "calibrated_coverage": float(covered.mean()),
                "multiplicative_calibrated_coverage": float(multiplicative_covered.mean()),
                **weighted_metrics,
                **{f"equal_{key}": value for key, value in equal_metrics.items()},
                **{f"persistence_{key}": value for key, value in persistence_metrics.items()},
            }
        )
        for index, state in enumerate(state_names):
            state_rows.append(
                {
                    "protocol": protocol,
                    "transition": name,
                    "t0": item["t0"],
                    "t1": item["t1"],
                    "horizon_days": item["t1"] - item["t0"],
                    "state": state,
                    "prediction_mean": float(weighted_state_mean[index]),
                    "observed_mean": float(target_mean[index]),
                    "ensemble_std": float(weighted_state_std[index]),
                    "absolute_error": float(abs(target_mean[index] - weighted_state_mean[index])),
                    "raw_interval_radius": float(raw_radius[index]),
                    "interval_radius": float(radius[index]),
                    "multiplicative_interval_radius": float(multiplicative_radius[index]),
                    "raw_covered": bool(raw_covered[index]),
                    "covered": bool(covered[index]),
                }
            )
    return metric_rows, state_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrated ensemble and zero-shot mouse prediction")
    parser.add_argument("--zebrafish", type=Path, required=True)
    parser.add_argument("--mouse", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260815)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--expected-members", type=int, default=5)
    parser.add_argument(
        "--additional",
        nargs="*",
        default=[],
        metavar="NAME=NPZ",
        help="Additional locked external StateDataset files to evaluate without retraining.",
    )
    args = parser.parse_args()

    zebrafish = StateDataset.load(args.zebrafish)
    mouse = StateDataset.load(args.mouse)
    if zebrafish.state_names != mouse.state_names:
        raise ValueError("mouse and zebrafish state definitions must match")
    config = load_yaml("configs/train.yaml")
    model_config = load_yaml("configs/model.yaml")
    device = resolve_device(config.get("device", "auto")).selected
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(config.get("cuda_memory_fraction", 0.78)))
    models, weights, validation_losses = load_members(args.checkpoints, device, args.expected_members)
    metadata = pd.DataFrame({"group": zebrafish.groups, "stage": zebrafish.times.astype(str)})
    _, validation_mask, test_mask, split = grouped_split(
        metadata, group_column="group", stage_column="stage", seed=args.split_seed
    )
    split.save(args.output_dir / "tables" / "zebrafish_split.json")
    k = int(model_config["graph"]["k"])
    max_nodes = int(config["batching"]["max_nodes"])
    max_steps = int(config["max_ode_steps_per_transition"])
    ode = model_config["ode"]

    validation_transitions = zebrafish.transitions(
        mask=validation_mask, k=k, max_nodes=max_nodes
    )
    validation = grouped_predictions(models, validation_transitions, weights, ode, max_steps)
    design, observed, validation_groups = [], [], []
    for item in validation.values():
        member_means = item["members"].mean(axis=1).T
        design.append(member_means)
        observed.append(item["target"].mean(axis=0))
        transition_group = f'{item["t0"]}_to_{item["t1"]}'
        validation_groups.extend([transition_group] * len(member_means))
    design = np.concatenate(design)
    observed = np.concatenate(observed)
    weights = fit_simplex_weights(design, observed)
    aggregation_method, aggregation_scores = select_aggregation(
        design, observed, np.asarray(validation_groups)
    )
    validation = grouped_predictions(models, validation_transitions, weights, ode, max_steps, aggregation_method)
    scale, additive_radius, scores, absolute_scores = conformal_factor(validation, weights, args.confidence)
    zebra_test = grouped_predictions(
        models, zebrafish.transitions(mask=test_mask, k=k, max_nodes=max_nodes), weights, ode, max_steps, aggregation_method
    )
    adjacent = grouped_predictions(
        models, mouse.transitions(k=k, max_nodes=max_nodes), weights, ode, max_steps, aggregation_method
    )
    origin = float(np.min(mouse.times))
    horizons = [
        mouse.transition_patches_between(origin, float(future), k=k, max_nodes=max_nodes, name=f"{origin:g}_to_{future:g}")
        for future in sorted(np.unique(mouse.times[mouse.times > origin]))
    ]
    direct = grouped_predictions(models, [x for group in horizons for x in group], weights, ode, max_steps, aggregation_method)

    additional_predictions = []
    for specification in args.additional:
        if "=" not in specification:
            raise ValueError(f"additional dataset must be NAME=NPZ, received {specification!r}")
        name, path = specification.split("=", 1)
        external = StateDataset.load(path)
        if external.state_names != zebrafish.state_names:
            raise ValueError(f"{name} state definitions do not match the registered six-state model")
        predictions = grouped_predictions(
            models,
            external.transitions(k=k, max_nodes=max_nodes),
            weights,
            ode,
            max_steps,
            aggregation_method,
        )
        additional_predictions.append((predictions, f"{name}_zero_shot_adjacent"))

    metric_rows, state_rows = [], []
    protocols = [
        (zebra_test, "zebrafish_internal_test"),
        (adjacent, "mouse_zero_shot_adjacent"),
        (direct, "mouse_zero_shot_direct_horizon"),
        *additional_predictions,
    ]
    for grouped, protocol in protocols:
        metrics, states = evaluate(grouped, weights, scale, additive_radius, zebrafish.state_names, protocol)
        metric_rows.extend(metrics)
        state_rows.extend(states)
    state_frame = pd.DataFrame(state_rows)
    mouse_states = state_frame[state_frame["protocol"].str.startswith("mouse")]
    uncertainty_error = float(mouse_states[["ensemble_std", "absolute_error"]].corr(method="spearman").iloc[0, 1])
    direct_states = state_frame[state_frame["protocol"] == "mouse_zero_shot_direct_horizon"]
    uncertainty_horizon = float(direct_states[["horizon_days", "ensemble_std"]].corr(method="spearman").iloc[0, 1])
    error_horizon = float(direct_states[["horizon_days", "absolute_error"]].corr(method="spearman").iloc[0, 1])
    export_table(pd.DataFrame(metric_rows), args.output_dir / "metrics" / "external_prediction.csv")
    export_table(state_frame, args.output_dir / "tables" / "state_mean_predictions.csv")
    export_table(
        pd.DataFrame({"member": np.arange(len(models)), "validation_loss": validation_losses, "weight": weights}),
        args.output_dir / "tables" / "ensemble_weights.csv",
    )
    export_table(
        pd.DataFrame([{"method": key, "validation_group_cv_mse": value, "selected": key == aggregation_method} for key, value in aggregation_scores.items()]),
        args.output_dir / "tables" / "aggregation_selection.csv",
    )
    atomic_json(
        {
            "status": "complete",
            "device": device,
            "members": len(models),
            "weighting": "nonnegative simplex weights fitted on zebrafish validation state means",
            "aggregation_selection": "leave-one-validation-transition-out comparison of simplex, equal mean, coordinate median, and trimmed mean",
            "selected_aggregation": aggregation_method,
            "aggregation_validation_mse": aggregation_scores,
            "ensemble_weights": weights.tolist(),
            "interval_method": "primary pathway-state-conditional additive split-conformal calibration on zebrafish validation transition means; pathway-conditional multiplicative diagnostic retained",
            "confidence": args.confidence,
            "conformal_scale": scale.tolist(),
            "conformal_additive_radius": additive_radius.tolist(),
            "calibration_scores": scores.tolist(),
            "absolute_calibration_scores": absolute_scores.tolist(),
            "mouse_retraining": False,
            "mouse_use": "exploratory direct cross-species transfer on matched six-state representation",
            "additional_external_datasets": args.additional,
            "uncertainty_error_spearman": uncertainty_error,
            "uncertainty_horizon_spearman": uncertainty_horizon,
            "error_horizon_spearman": error_horizon,
            "limitations": [
                "the original registered protocol did not assume species-independent prediction",
                "mouse contains one spatial sample per stage, so transitions are not independent biological replicates",
                "time and state-score alignment across species are assumptions tested here, not established facts",
            ],
        },
        args.output_dir / "run_manifest.json",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "device": device,
                "conformal_scale": np.asarray(scale).tolist(),
                "conformal_additive_radius": np.asarray(additive_radius).tolist(),
                "rows": len(metric_rows),
            },
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
