from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import linregress

from cardiogb.models.mechanistic import MechanisticODE
from cardiogb.models.neural_ode import NeuralODEFunc
from cardiogb.synthetic.recovery import recover_hidden_mechanism, recover_mechanistic_parameters
from cardiogb.synthetic.simulator import simulate_system
from cardiogb.utils.config import load_yaml
from cardiogb.utils.io import atomic_json, export_table
from cardiogb.utils.seed import seed_everything


def percentile_ci(values, rng, n_resamples=10000, confidence=0.95):
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(values.mean()), float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))


def e5(noise: float, seed: int, epochs: int, config):
    seed_everything(seed)
    truth = MechanisticODE.from_config(config)
    simulated = simulate_system(
        truth, observation_times=[0.0, 0.25, 0.5, 1.0, 3.0], num_entities=128,
        noise_std=noise, seed=seed, step_size=0.05,
    )
    fitted = MechanisticODE.from_config(config)
    generator = torch.Generator().manual_seed(seed + 100000)
    with torch.no_grad():
        for parameter in fitted.raw_parameters.values():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.35)
    recovered = recover_mechanistic_parameters(
        fitted, simulated.times, simulated.observations, simulated.true_parameters,
        epochs=epochs, learning_rate=0.03, step_size=0.05,
    )
    rows = []
    for name, true_value in recovered.true_parameters.items():
        inferred = recovered.inferred_parameters[name]
        rows.append({
            "experiment": "E5", "noise_std": noise, "seed": seed, "parameter": name,
            "true_value": true_value, "inferred_value": inferred,
            "absolute_error": abs(inferred - true_value),
            "relative_error": abs(inferred - true_value) / (abs(true_value) + 1e-8),
            **recovered.metrics,
            "final_loss": recovered.loss_history[-1],
        })
    return rows


def e6(noise: float, seed: int, epochs: int, config):
    seed_everything(seed)
    truth = MechanisticODE.from_config(config)
    simulated = simulate_system(
        truth, observation_times=[0.0, 0.25, 0.5, 1.0, 3.0], num_entities=128,
        noise_std=noise, hidden_mechanism=True, hidden_strength=0.4, seed=seed, step_size=0.05,
    )
    residual = NeuralODEFunc(state_dim=6, hidden_dim=32, layers=2, time_dependent=False)
    recovered = recover_hidden_mechanism(
        residual, simulated.observations, simulated.hidden_mechanism_values,
        epochs=epochs, learning_rate=3e-3,
    )
    with torch.no_grad():
        states = simulated.observations.reshape(-1, 6)
        target = simulated.hidden_mechanism_values.reshape(-1, 6)[:, 3].numpy()
        prediction = residual(0.0, states, None)[:, 3].numpy()
        regression = linregress(target, prediction)
        perturbed = states.clone()
        delta = 0.05
        perturbed[:, 2] = (perturbed[:, 2] + delta).clamp(0, 1)
        response = (residual(0.0, perturbed, None)[:, 3] - residual(0.0, states, None)[:, 3]).numpy()
    return {
        "experiment": "E6", "noise_std": noise, "seed": seed,
        "correlation": recovered.correlation, "rmse": recovered.rmse,
        "regression_slope": float(regression.slope), "regression_r2": float(regression.rvalue**2),
        "regression_p": float(regression.pvalue),
        "fibroblast_perturbation_mean_response": float(response.mean()),
        "fibroblast_perturbation_positive_fraction": float((response > 0).mean()),
        "final_loss": recovered.loss_history[-1],
    }


def summarize(frame: pd.DataFrame, metrics: list[str], seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for noise, subset in frame.groupby("noise_std", observed=True):
        for metric in metrics:
            values = subset.groupby("seed", observed=True)[metric].mean().to_numpy()
            estimate, lower, upper = percentile_ci(values, rng)
            rows.append({
                "noise_std": noise, "metric": metric, "estimate": estimate,
                "ci_lower": lower, "ci_upper": upper, "confidence": 0.95,
                "n_seeds": len(values), "resampling_unit": "simulation seed",
            })
    return pd.DataFrame(rows)


def parameter_stability(frame: pd.DataFrame):
    rows = []
    for (noise, parameter), subset in frame.groupby(["noise_std", "parameter"], observed=True):
        values = subset["inferred_value"].to_numpy()
        rows.append({
            "noise_std": noise, "parameter": parameter, "mean": float(values.mean()),
            "std": float(values.std(ddof=1)), "coefficient_of_variation": float(values.std(ddof=1) / max(abs(values.mean()), 1e-8)),
            "median_relative_error": float(subset["relative_error"].median()),
            "n_seeds": len(values),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Replicated E5/E6 recovery across registered noise levels")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20260815, 20260820)))
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1])
    args = parser.parse_args()
    config = load_yaml("configs/mechanistic_model.yaml")
    e5_rows, e6_rows = [], []
    manifest = {"status": "partial", "noise_std": args.noise, "seeds": args.seeds, "epochs": args.epochs, "completed": []}
    for noise in args.noise:
        for seed in args.seeds:
            marker = args.output_dir / "runs" / f"noise_{noise:g}" / f"seed_{seed}.json"
            if marker.is_file():
                payload = json.loads(marker.read_text(encoding="utf-8"))
                e5_rows.extend(payload["e5"])
                e6_rows.append(payload["e6"])
            else:
                rows5 = e5(noise, seed, args.epochs, config)
                row6 = e6(noise, seed, args.epochs, config)
                atomic_json({"e5": rows5, "e6": row6}, marker)
                e5_rows.extend(rows5)
                e6_rows.append(row6)
            manifest["completed"].append({"noise_std": noise, "seed": seed})
            atomic_json(manifest, args.output_dir / "run_manifest.json")
    e5_frame, e6_frame = pd.DataFrame(e5_rows), pd.DataFrame(e6_rows)
    export_table(e5_frame, args.output_dir / "tables" / "e5_parameter_recovery.csv")
    export_table(e6_frame, args.output_dir / "tables" / "e6_hidden_recovery.csv")
    export_table(
        summarize(e5_frame, ["parameter_rmse", "parameter_mae", "parameter_correlation", "relative_error"], args.seeds[0]),
        args.output_dir / "tables" / "e5_confidence_intervals.csv",
    )
    export_table(
        summarize(e6_frame, ["correlation", "rmse", "regression_slope", "regression_r2", "fibroblast_perturbation_mean_response"], args.seeds[0]),
        args.output_dir / "tables" / "e6_confidence_intervals.csv",
    )
    export_table(parameter_stability(e5_frame), args.output_dir / "tables" / "parameter_stability.csv")
    manifest["status"] = "complete"
    manifest["n_runs_per_experiment"] = len(args.noise) * len(args.seeds)
    manifest["confidence_interval"] = "percentile bootstrap over simulation seeds, 10000 resamples"
    atomic_json(manifest, args.output_dir / "run_manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()