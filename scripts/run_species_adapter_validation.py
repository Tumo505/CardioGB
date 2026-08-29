"""Few-shot mouse adaptation with frozen zebrafish CardioGB dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cardiogb.data.state_dataset import StateDataset
from cardiogb.losses.distribution import moment_matching, rbf_mmd
from cardiogb.metrics.distributional import distribution_metrics
from cardiogb.models.factory import build_model
from cardiogb.models.species_adapter import SpeciesAdapter, SpeciesAdaptedForecaster
from cardiogb.utils.config import load_yaml
from cardiogb.utils.device import resolve_device
from cardiogb.utils.io import atomic_json, export_table
from cardiogb.utils.seed import seed_everything


def subsample(values: torch.Tensor, limit: int) -> torch.Tensor:
    if len(values) <= limit:
        return values
    index = torch.linspace(0, len(values) - 1, steps=limit, device=values.device).round().long()
    return values[index]


def selected_patches(transitions, maximum: int):
    grouped = {}
    for transition in transitions:
        grouped.setdefault(transition.evaluation_group or transition.name, []).append(transition)
    selected = []
    for patches in grouped.values():
        if len(patches) <= maximum:
            selected.extend(patches)
        else:
            indices = torch.linspace(0, len(patches) - 1, steps=maximum).round().long().tolist()
            selected.extend(patches[index] for index in indices)
    return selected


def load_frozen_member(path: Path, device: str):
    model = build_model(
        "cardiogb", load_yaml("configs/model.yaml"), load_yaml("configs/mechanistic_model.yaml")
    )
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), float(payload.get("validation_loss", np.nan))


def fit_adapter(
    checkpoint: Path,
    calibration,
    validation,
    *,
    device: str,
    epochs: int,
    learning_rate: float,
    regularization: float,
    sample_limit: int,
    step_size: float,
    solver: str,
) -> tuple[SpeciesAdaptedForecaster, list[dict[str, float]], float]:
    shared, source_validation_loss = load_frozen_member(checkpoint, device)
    adapter = SpeciesAdapter(6, ("zebrafish", "mouse")).to(device)
    forecaster = SpeciesAdaptedForecaster(shared, adapter).to(device)
    forecaster.freeze_shared_dynamics(True)
    parameters = [
        *adapter.encoders["mouse"].parameters(),
        *adapter.decoders["mouse"].parameters(),
        adapter.raw_time_scales["mouse"],
    ]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    best = float("inf")
    best_state = None
    history = []

    def epoch_loss(transitions, training: bool) -> float:
        forecaster.adapter.train(training)
        forecaster.shared_model.eval()
        losses = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for transition in transitions:
                if training:
                    optimizer.zero_grad(set_to_none=True)
                graph = transition.graph.to(device)
                prediction = forecaster.forecast(
                    transition.source_states.to(device),
                    graph,
                    transition.t0,
                    transition.t1,
                    species="mouse",
                    step_size=step_size,
                    method=solver,
                    checkpoint_steps=1 if training else False,
                )
                predicted = subsample(prediction, sample_limit)
                observed = subsample(transition.target_states.to(device), sample_limit)
                loss = rbf_mmd(predicted, observed) + 0.25 * moment_matching(predicted, observed)
                loss = loss + regularization * adapter.regularization()
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                    optimizer.step()
                losses.append(float(loss.detach()))
        return float(np.mean(losses))

    for epoch in range(epochs):
        train_loss = epoch_loss(calibration, True)
        validation_loss = epoch_loss(validation, False)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best:
            best = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in adapter.state_dict().items()}
    if best_state is not None:
        adapter.load_state_dict(best_state)
    forecaster.eval()
    return forecaster, history, source_validation_loss


@torch.no_grad()
def predict(forecaster, transitions, device: str, step_size: float, solver: str):
    grouped = {}
    for transition in transitions:
        key = transition.evaluation_group or transition.name
        graph = transition.graph.to(device)
        prediction = forecaster.forecast(
            transition.source_states.to(device),
            graph,
            transition.t0,
            transition.t1,
            species="mouse",
            step_size=step_size,
            method=solver,
        ).cpu().numpy()
        item = grouped.setdefault(
            key,
            {
                "prediction": [],
                "source": [],
                "target": transition.target_states.numpy(),
                "t0": transition.t0,
                "t1": transition.t1,
            },
        )
        item["prediction"].append(prediction)
        item["source"].append(transition.source_states.numpy())
    for item in grouped.values():
        item["prediction"] = np.concatenate(item["prediction"])
        item["source"] = np.concatenate(item["source"])
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--sample-limit", type=int, default=1024)
    parser.add_argument("--max-patches-per-transition", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    seed_everything(args.seed)

    config = load_yaml("configs/train.yaml")
    model_config = load_yaml("configs/model.yaml")
    device = resolve_device(config.get("device", "auto")).selected
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(config.get("cuda_memory_fraction", 0.998)))
    dataset = StateDataset.load(args.mouse)
    times = sorted(np.unique(dataset.times).astype(float).tolist())
    if len(times) < 5:
        raise ValueError("species adaptation requires at least five stages for train/validation/test separation")
    validation_end = times[-3]
    test_start = times[-3]
    all_transitions = dataset.transitions(
        k=int(model_config["graph"]["k"]),
        max_nodes=int(config["batching"]["max_nodes"]),
    )
    calibration = [item for item in all_transitions if item.t1 < validation_end]
    validation = [item for item in all_transitions if item.t1 == validation_end]
    test = [item for item in all_transitions if item.t0 >= test_start]
    calibration = selected_patches(calibration, args.max_patches_per_transition)
    validation = selected_patches(validation, args.max_patches_per_transition)
    test = selected_patches(test, args.max_patches_per_transition)
    if not calibration or not validation or not test:
        raise ValueError("stage split produced an empty calibration, validation, or test set")

    checkpoints = sorted(args.checkpoints.glob("member_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no member_*.pt checkpoints found in {args.checkpoints}")
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    member_outputs = []
    parameter_rows = []
    for member, checkpoint in enumerate(checkpoints):
        forecaster, history, source_loss = fit_adapter(
            checkpoint,
            calibration,
            validation,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            regularization=args.regularization,
            sample_limit=args.sample_limit,
            step_size=float(model_config["ode"]["step_size"]),
            solver=str(model_config["ode"]["solver"]),
        )
        member_outputs.append(
            predict(
                forecaster,
                test,
                device,
                float(model_config["ode"]["step_size"]),
                str(model_config["ode"]["solver"]),
            )
        )
        export_table(pd.DataFrame(history), args.output_dir / "history" / f"member_{member}.csv")
        torch.save(
            {"adapter": forecaster.adapter.state_dict(), "source_checkpoint": str(checkpoint)},
            args.output_dir / "checkpoints" / f"member_{member}_mouse_adapter.pt",
        )
        parameter_rows.append(
            {
                "member": member,
                "source_validation_loss": source_loss,
                "mouse_time_scale": float(forecaster.adapter.time_scale("mouse").detach().cpu()),
                "adapter_regularization": float(forecaster.adapter.regularization().detach().cpu()),
            }
        )

    rows = []
    for transition in member_outputs[0]:
        members = np.stack([output[transition]["prediction"] for output in member_outputs])
        ensemble = np.median(members, axis=0)
        item = member_outputs[0][transition]
        adapted = distribution_metrics(ensemble, item["target"])
        persistence = distribution_metrics(item["source"], item["target"])
        rows.append(
            {
                "transition": transition,
                "t0": item["t0"],
                "t1": item["t1"],
                "horizon_days": item["t1"] - item["t0"],
                **adapted,
                **{f"persistence_{name}": value for name, value in persistence.items()},
            }
        )
    export_table(pd.DataFrame(rows), args.output_dir / "metrics" / "heldout_mouse_adapter.csv")
    export_table(pd.DataFrame(parameter_rows), args.output_dir / "tables" / "adapter_parameters.csv")
    atomic_json(
        {
            "status": "complete",
            "protocol": "freeze zebrafish dynamics; fit mouse observation/time adapters on early stages; evaluate later stages",
            "calibration_times": sorted({(item.t0, item.t1) for item in calibration}),
            "validation_times": sorted({(item.t0, item.t1) for item in validation}),
            "test_times": sorted({(item.t0, item.t1) for item in test}),
            "members": len(checkpoints),
            "zero_shot": False,
        },
        args.output_dir / "run_manifest.json",
    )
    print(json.dumps({"members": len(checkpoints), "test_transitions": len(rows), "device": device}))


if __name__ == "__main__":
    main()
