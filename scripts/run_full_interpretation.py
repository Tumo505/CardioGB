from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from cardiogb.data.graphs import build_spatial_knn_graph
from cardiogb.data.state_dataset import StateDataset
from cardiogb.interpretation.attribution import integrated_gradients
from cardiogb.interpretation.mechanistic_insufficiency import mechanistic_insufficiency
from cardiogb.models.factory import build_model
from cardiogb.models.message_passing import TorchGraph
from cardiogb.utils.config import load_yaml
from cardiogb.utils.device import resolve_device
from cardiogb.utils.io import atomic_json, export_table


def bh_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def bootstrap_mean(values, seed, n=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n, len(values)), replace=True).mean(1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def graph_for(dataset, index, section, k, device):
    spatial = build_spatial_knn_graph(dataset.coordinates[index], np.repeat(str(section), len(index)), k=k)
    edge_index, edge_attr = spatial.to_torch(device)
    return TorchGraph(edge_index, edge_attr)


def main():
    parser = argparse.ArgumentParser(description="Full ensemble E7 insufficiency, attribution, and parameter stability")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ig-steps", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    dataset = StateDataset.load(args.data)
    checkpoints = sorted(args.checkpoint_root.glob("seed_*/checkpoints/cardiogb.pt"))
    if len(checkpoints) < 2:
        raise ValueError(f"need multiple CardioGB checkpoints under {args.checkpoint_root}")
    model_config = load_yaml("configs/model.yaml")
    mech_config = load_yaml("configs/mechanistic_model.yaml")
    train_config = load_yaml("configs/train.yaml")
    device = resolve_device(train_config.get("device", "auto")).selected
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(train_config.get("cuda_memory_fraction", 0.78)))
    spot_frames, attribution_rows, parameter_rows = [], [], []
    sections = sorted(np.unique(dataset.sections).astype(str))
    for member, checkpoint_path in enumerate(checkpoints):
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model("cardiogb", model_config, mech_config)
        model.load_state_dict(payload["model"])
        model = model.to(device).eval()
        seed = int(checkpoint_path.parents[1].name.replace("seed_", ""))
        for name, value in model.mechanistic_model.constrained_parameters().items():
            parameter_rows.append({"member": member, "seed": seed, "parameter": name, "value": float(value.detach().cpu())})
        for state, value in zip(dataset.state_names, model.mechanistic_gate().detach().cpu()):
            parameter_rows.append({"member": member, "seed": seed, "parameter": f"mechanistic_gate_{state}", "value": float(value)})
        for state, value in zip(dataset.state_names, model.residual_scale().detach().cpu()):
            parameter_rows.append({"member": member, "seed": seed, "parameter": f"residual_scale_{state}", "value": float(value)})
        for section_number, section in enumerate(sections):
            index = np.flatnonzero(dataset.sections.astype(str) == section)
            stage = float(np.unique(dataset.times[index])[0])
            graph = graph_for(dataset, index, section, args.k, device)
            states = torch.as_tensor(dataset.states[index], dtype=torch.float32, device=device)
            with torch.no_grad():
                terms = model.vector_field(stage, states, graph)
                mi = mechanistic_insufficiency(terms["mechanistic"], terms["residual"]).cpu().numpy()
            frame = pd.DataFrame({
                "member": member, "seed": seed, "spot_index": index, "mi": mi,
                "stage_days": dataset.times[index], "domain": dataset.domains[index],
                "section": section, "biological_unit": dataset.groups[index],
            })
            for state_index, state_name in enumerate(dataset.state_names):
                frame[state_name] = dataset.states[index, state_index]
            spot_frames.append(frame)
            for target, target_name in enumerate(dataset.state_names):
                attribution = integrated_gradients(
                    model.residual_model, stage, states, graph, target_state=target, steps=args.ig_steps
                ).abs().mean(dim=0).cpu().numpy()
                for source, value in zip(dataset.state_names, attribution):
                    attribution_rows.append({
                        "member": member, "seed": seed, "stage_days": stage, "section": section,
                        "biological_unit": str(dataset.groups[index[0]]), "target_state": target_name,
                        "input_state": source, "mean_absolute_integrated_gradient": float(value),
                        "claim_scope": "model-derived association; hypothesis generating",
                    })
            if device == "cuda" and section_number % 10 == 0:
                torch.cuda.empty_cache()
    spots = pd.concat(spot_frames, ignore_index=True)
    attribution = pd.DataFrame(attribution_rows)
    parameters = pd.DataFrame(parameter_rows)
    export_table(spots, args.output_dir / "tables" / "mi_spots_all_members.parquet")
    export_table(attribution, args.output_dir / "tables" / "residual_attribution_all_sections.csv")
    export_table(parameters, args.output_dir / "tables" / "mechanistic_parameters.csv")
    unit = spots.groupby(["member", "seed", "biological_unit", "stage_days"], observed=True)[["mi", *dataset.state_names]].mean().reset_index()
    export_table(unit, args.output_dir / "tables" / "mi_biological_units.csv")
    unit_ensemble = unit.groupby(["biological_unit", "stage_days"], observed=True)[["mi", *dataset.state_names]].mean().reset_index()
    association = []
    for state in dataset.state_names:
        result = spearmanr(unit_ensemble["mi"], unit_ensemble[state])
        association.append({"state": state, "spearman": float(result.statistic), "p_value": float(result.pvalue), "n_biological_units": len(unit_ensemble)})
    association = pd.DataFrame(association)
    association["p_adjust_bh"] = bh_adjust(association["p_value"])
    export_table(association, args.output_dir / "tables" / "mi_pathway_associations.csv")
    stage_rows = []
    for stage, subset in unit_ensemble.groupby("stage_days", observed=True):
        estimate, lower, upper = bootstrap_mean(subset["mi"], int(20260815 + stage * 100))
        stage_rows.append({"stage_days": stage, "mi_mean": estimate, "ci_lower": lower, "ci_upper": upper, "n_biological_units": len(subset)})
    export_table(pd.DataFrame(stage_rows), args.output_dir / "tables" / "mi_stage_bootstrap.csv")
    domain = spots.groupby(["member", "biological_unit", "stage_days", "domain"], observed=True)["mi"].mean().reset_index()
    export_table(domain, args.output_dir / "tables" / "mi_domain_biological_units.csv")
    stability_rows = []
    for parameter, subset in parameters.groupby("parameter", observed=True):
        values = subset["value"].to_numpy()
        estimate, lower, upper = bootstrap_mean(values, 20260815)
        stability_rows.append({
            "parameter": parameter, "mean": estimate, "ci_lower": lower, "ci_upper": upper,
            "std": float(values.std(ddof=1)), "coefficient_of_variation": float(values.std(ddof=1) / max(abs(values.mean()), 1e-8)),
            "n_members": len(values),
        })
    export_table(pd.DataFrame(stability_rows), args.output_dir / "tables" / "parameter_stability.csv")
    matrix = parameters.pivot(index="seed", columns="parameter", values="value")
    correlation = matrix.corr()
    correlation_long = correlation.rename_axis("parameter_1").reset_index().melt(
        id_vars="parameter_1", var_name="parameter_2", value_name="pearson_correlation"
    )
    export_table(correlation_long, args.output_dir / "tables" / "parameter_pair_correlations.csv")
    standardized = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, 1)
    singular = np.linalg.svd(standardized.to_numpy(), compute_uv=False)
    tolerance = singular.max() * max(standardized.shape) * np.finfo(float).eps if len(singular) else 0.0
    positive = singular[singular > tolerance]
    effective_rank = int(len(positive))
    condition_number = float(positive.max() / positive.min()) if len(positive) > 1 else np.nan
    off_diagonal = correlation.to_numpy()[~np.eye(len(correlation), dtype=bool)]
    export_table(
        pd.DataFrame({
            "diagnostic": ["effective_rank", "condition_number", "maximum_absolute_pair_correlation", "parameters", "seeds"],
            "value": [effective_rank, condition_number, float(np.nanmax(np.abs(off_diagonal))), matrix.shape[1], matrix.shape[0]],
        }),
        args.output_dir / "tables" / "parameter_identifiability_diagnostics.csv",
    )
    atomic_json(
        {
            "status": "complete", "device": device, "members": len(checkpoints), "sections": len(sections),
            "spots_per_member": len(dataset.states), "ig_steps": args.ig_steps,
            "parameter_stability_rank": effective_rank,
            "parameter_stability_condition_number": condition_number,
            "maximum_absolute_parameter_correlation": float(np.nanmax(np.abs(off_diagonal))),
            "parameter_stability_singular_values": singular.tolist(),
            "identifiability_scope": "real-data cross-seed stability only; ground-truth identifiability is evaluated in E5",
            "statistical_unit": "biological unit; ensemble members are averaged before pathway association tests",
        },
        args.output_dir / "run_manifest.json",
    )
    print(json.dumps({"status": "complete", "members": len(checkpoints), "sections": len(sections), "attribution_rows": len(attribution)}, indent=2))


if __name__ == "__main__":
    main()