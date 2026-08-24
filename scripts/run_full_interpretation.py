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


def local_mechanistic_sensitivities(model, t, states):
    """Sensitivity of the gated mechanistic vector field to constrained rates."""
    mechanism = model.mechanistic_model
    state_names = mechanism.state_names
    state_mean = states.detach().mean(dim=0)
    gate = model.mechanistic_gate().detach().to(device=states.device, dtype=states.dtype)
    values = {
        (parameter, target): states.new_zeros(())
        for parameter in mechanism.raw_parameters
        for target in state_names
    }
    for interaction in mechanism.interactions:
        target = state_names[interaction.target]
        values[(interaction.parameter, target)] = values[(interaction.parameter, target)] + (
            gate[interaction.target] * interaction.sign * state_mean[interaction.source]
        )
    for decay in mechanism.decays:
        target = state_names[decay.state]
        source = state_mean[decay.state]
        derivative = -source if decay.relaxation_target is None else decay.relaxation_target - source
        values[(decay.parameter, target)] = values[(decay.parameter, target)] + gate[decay.state] * derivative
    injury = mechanism.injury_input(t, states)
    activation = state_names[mechanism.activation_index]
    values[(mechanism.injury_parameter, activation)] = (
        values[(mechanism.injury_parameter, activation)] + gate[mechanism.activation_index] * injury
    )
    return [
        {
            "parameter": parameter,
            "target_state": target,
            "signed_local_sensitivity": float(value.detach().cpu()),
            "absolute_local_sensitivity": float(value.detach().abs().cpu()),
        }
        for (parameter, target), value in values.items()
    ]


def parameter_records(model, *, member, seed, source, case):
    fit_id = f"{source}:case_{case}:seed_{seed}" if case is not None else f"{source}:seed_{seed}"
    rows = []
    for name, value in model.mechanistic_model.constrained_parameters().items():
        rows.append({"fit_id": fit_id, "source": source, "case": case, "member": member, "seed": seed, "parameter": name, "value": float(value.detach().cpu())})
    for state, value in zip(model.mechanistic_model.state_names, model.mechanistic_gate().detach().cpu()):
        rows.append({"fit_id": fit_id, "source": source, "case": case, "member": member, "seed": seed, "parameter": f"mechanistic_gate_{state}", "value": float(value)})
    for state, value in zip(model.mechanistic_model.state_names, model.residual_scale().detach().cpu()):
        rows.append({"fit_id": fit_id, "source": source, "case": case, "member": member, "seed": seed, "parameter": f"residual_scale_{state}", "value": float(value)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Full ensemble E7 insufficiency, attribution, and parameter stability")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--fold-checkpoint-root", type=Path, default=Path("results/e4_group_cv_full"))
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
    spot_frames, attribution_rows, parameter_rows, sensitivity_rows = [], [], [], []
    sections = sorted(np.unique(dataset.sections).astype(str))
    for member, checkpoint_path in enumerate(checkpoints):
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model("cardiogb", model_config, mech_config)
        model.load_state_dict(payload["model"])
        model = model.to(device).eval()
        seed = int(checkpoint_path.parents[1].name.replace("seed_", ""))
        parameter_rows.extend(parameter_records(model, member=member, seed=seed, source="E1", case=None))
        split_path = checkpoint_path.parents[1] / "tables" / "split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        role_by_group = {
            str(group): role
            for role, key in (("train", "train_samples"), ("validation", "validation_samples"), ("test", "test_samples"))
            for group in split[key]
        }
        for section_number, section in enumerate(sections):
            index = np.flatnonzero(dataset.sections.astype(str) == section)
            stage = float(np.unique(dataset.times[index])[0])
            graph = graph_for(dataset, index, section, args.k, device)
            states = torch.as_tensor(dataset.states[index], dtype=torch.float32, device=device)
            section_groups = np.unique(dataset.groups[index].astype(str))
            if len(section_groups) != 1:
                raise ValueError(f"section {section} spans multiple biological units: {section_groups.tolist()}")
            biological_unit = str(section_groups[0])
            split_role = role_by_group.get(biological_unit)
            if split_role is None:
                raise ValueError(f"biological unit {biological_unit} is absent from {split_path}")
            with torch.no_grad():
                terms = model.vector_field(stage, states, graph)
                mi = mechanistic_insufficiency(terms["mechanistic"], terms["residual"]).cpu().numpy()
            frame = pd.DataFrame({
                "member": member, "seed": seed, "spot_index": index, "mi": mi,
                "stage_days": dataset.times[index], "domain": dataset.domains[index],
                "section": section, "biological_unit": dataset.groups[index], "split_role": split_role,
            })
            for state_index, state_name in enumerate(dataset.state_names):
                frame[state_name] = dataset.states[index, state_index]
            spot_frames.append(frame)
            if split_role == "test":
                for sensitivity in local_mechanistic_sensitivities(model, stage, states):
                    sensitivity_rows.append({
                        "member": member, "seed": seed, "stage_days": stage,
                        "section": section, "biological_unit": biological_unit,
                        **sensitivity,
                    })
            for target, target_name in enumerate(dataset.state_names):
                attribution = integrated_gradients(
                    model.residual_model, stage, states, graph, target_state=target, steps=args.ig_steps
                ).abs().mean(dim=0).cpu().numpy()
                for source, value in zip(dataset.state_names, attribution):
                    attribution_rows.append({
                        "member": member, "seed": seed, "stage_days": stage, "section": section,
                        "biological_unit": biological_unit, "split_role": split_role, "target_state": target_name,
                        "input_state": source, "mean_absolute_integrated_gradient": float(value),
                        "claim_scope": "model-derived association; hypothesis generating",
                    })
            if device == "cuda" and section_number % 10 == 0:
                torch.cuda.empty_cache()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    fold_manifest_path = args.fold_checkpoint_root / "run_manifest.json"
    if not fold_manifest_path.is_file():
        raise FileNotFoundError(f"full E4 manifest required for seed-and-fold stability: {fold_manifest_path}")
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    if fold_manifest.get("status") != "complete":
        raise RuntimeError("E4 must be complete before real-data fold stability is computed")
    fold_checkpoints = sorted(args.fold_checkpoint_root.glob("case_*/seed_*/checkpoints/cardiogb.pt"))
    if len(fold_checkpoints) != len(fold_manifest.get("completed", [])):
        raise RuntimeError("E4 checkpoint count does not match its completed manifest")
    for fold_member, fold_checkpoint in enumerate(fold_checkpoints):
        payload = torch.load(fold_checkpoint, map_location="cpu", weights_only=False)
        model = build_model("cardiogb", model_config, mech_config)
        model.load_state_dict(payload["model"])
        seed = int(fold_checkpoint.parents[1].name.replace("seed_", ""))
        case = int(fold_checkpoint.parents[2].name.replace("case_", ""))
        parameter_rows.extend(parameter_records(model, member=fold_member, seed=seed, source="E4", case=case))
        del model
    spots = pd.concat(spot_frames, ignore_index=True)
    attribution = pd.DataFrame(attribution_rows)
    parameters = pd.DataFrame(parameter_rows)
    export_table(spots, args.output_dir / "tables" / "mi_spots_all_members.parquet")
    export_table(attribution, args.output_dir / "tables" / "residual_attribution_all_sections.csv")
    export_table(parameters, args.output_dir / "tables" / "mechanistic_parameters.csv")
    sensitivities = pd.DataFrame(sensitivity_rows)
    if sensitivities.empty:
        raise RuntimeError("no held-out local mechanistic sensitivities were computed")
    export_table(sensitivities, args.output_dir / "tables" / "parameter_local_sensitivity_sections.parquet")
    sensitivity_unit = sensitivities.groupby(
        ["member", "seed", "biological_unit", "parameter", "target_state"], observed=True
    )[["signed_local_sensitivity", "absolute_local_sensitivity"]].mean().reset_index()
    sensitivity_ensemble = sensitivity_unit.groupby(
        ["biological_unit", "parameter", "target_state"], observed=True
    )[["signed_local_sensitivity", "absolute_local_sensitivity"]].mean().reset_index()
    sensitivity_summary_rows = []
    for (parameter, target), subset in sensitivity_ensemble.groupby(["parameter", "target_state"], observed=True):
        estimate, lower, upper = bootstrap_mean(
            subset["absolute_local_sensitivity"].to_numpy(),
            20260815 + len(sensitivity_summary_rows),
        )
        sensitivity_summary_rows.append({
            "parameter": parameter, "target_state": target,
            "mean_absolute_local_sensitivity": estimate,
            "ci_lower": lower, "ci_upper": upper,
            "mean_signed_local_sensitivity": float(subset["signed_local_sensitivity"].mean()),
            "n_biological_units": int(subset["biological_unit"].nunique()),
            "sensitivity_definition": "partial derivative of gated mechanistic vector field with respect to constrained rate",
        })
    sensitivity_summary = pd.DataFrame(sensitivity_summary_rows)
    export_table(sensitivity_summary, args.output_dir / "tables" / "parameter_local_sensitivity.csv")
    sensitivity_profile = sensitivity_summary.pivot(
        index="parameter", columns="target_state", values="mean_absolute_local_sensitivity"
    ).fillna(0.0)
    sensitivity_correlation = sensitivity_profile.T.corr()
    sensitivity_correlation_long = sensitivity_correlation.rename_axis("parameter_1").reset_index().melt(
        id_vars="parameter_1", var_name="parameter_2", value_name="local_sensitivity_profile_correlation"
    )
    export_table(sensitivity_correlation_long, args.output_dir / "tables" / "parameter_local_sensitivity_correlations.csv")
    local_singular = np.linalg.svd(sensitivity_profile.to_numpy(), compute_uv=False)
    local_tolerance = local_singular.max() * max(sensitivity_profile.shape) * np.finfo(float).eps if len(local_singular) else 0.0
    local_rank = int(np.count_nonzero(local_singular > local_tolerance))
    local_off_diagonal = sensitivity_correlation.to_numpy()[~np.eye(len(sensitivity_correlation), dtype=bool)]
    local_max_correlation = float(np.nanmax(np.abs(local_off_diagonal)))
    local_profile_norm = np.linalg.norm(sensitivity_profile.to_numpy(), axis=1)
    all_units = spots.groupby(["member", "seed", "split_role", "biological_unit", "stage_days"], observed=True)[["mi", *dataset.state_names]].mean().reset_index()
    export_table(all_units, args.output_dir / "tables" / "mi_biological_units_all_roles.csv")
    unit = all_units[all_units["split_role"] == "test"].copy()
    if unit.empty:
        raise RuntimeError("no cross-fitted held-out biological units were available for E7 inference")
    export_table(unit, args.output_dir / "tables" / "mi_biological_units.csv")
    heldout_attribution = attribution[attribution["split_role"] == "test"].copy()
    export_table(heldout_attribution, args.output_dir / "tables" / "residual_attribution_heldout.csv")
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
    heldout_spots = spots[spots["split_role"] == "test"]
    domain = heldout_spots.groupby(["member", "biological_unit", "stage_days", "domain"], observed=True)["mi"].mean().reset_index()
    export_table(domain, args.output_dir / "tables" / "mi_domain_biological_units.csv")
    stability_rows = []
    for parameter, subset in parameters.groupby("parameter", observed=True):
        values = subset["value"].to_numpy()
        estimate, lower, upper = bootstrap_mean(values, 20260815)
        stability_rows.append({
            "parameter": parameter, "mean": estimate, "ci_lower": lower, "ci_upper": upper,
            "std": float(values.std(ddof=1)), "coefficient_of_variation": float(values.std(ddof=1) / max(abs(values.mean()), 1e-8)),
            "sign_consistency": float(max(np.mean(values >= 0), np.mean(values <= 0))),
            "n_members": len(values), "n_e1_fits": int((subset["source"] == "E1").sum()),
            "n_e4_fits": int((subset["source"] == "E4").sum()),
        })
    export_table(pd.DataFrame(stability_rows), args.output_dir / "tables" / "parameter_stability.csv")
    protocol_stability_rows = []
    for (source, parameter), subset in parameters.groupby(["source", "parameter"], observed=True):
        values = subset["value"].to_numpy()
        estimate, lower, upper = bootstrap_mean(values, 20260815)
        protocol_stability_rows.append({
            "source": source, "parameter": parameter, "mean": estimate, "ci_lower": lower,
            "ci_upper": upper, "std": float(values.std(ddof=1)),
            "coefficient_of_variation": float(values.std(ddof=1) / max(abs(values.mean()), 1e-8)),
            "sign_consistency": float(max(np.mean(values >= 0), np.mean(values <= 0))),
            "n_fits": len(values), "n_folds": int(subset["case"].nunique()) if source == "E4" else 0,
        })
    export_table(pd.DataFrame(protocol_stability_rows), args.output_dir / "tables" / "parameter_stability_by_protocol.csv")
    matrix = parameters.pivot(index="fit_id", columns="parameter", values="value")
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
            "diagnostic": [
                "effective_rank", "condition_number", "maximum_absolute_pair_correlation",
                "parameters", "fits", "local_sensitivity_effective_rank",
                "maximum_absolute_local_sensitivity_profile_correlation",
                "minimum_local_sensitivity_profile_norm",
            ],
            "value": [
                effective_rank, condition_number, float(np.nanmax(np.abs(off_diagonal))),
                matrix.shape[1], matrix.shape[0], local_rank, local_max_correlation,
                float(local_profile_norm.min()),
            ],
        }),
        args.output_dir / "tables" / "parameter_identifiability_diagnostics.csv",
    )
    atomic_json(
        {
            "status": "complete", "device": device, "members": len(checkpoints), "sections": len(sections),
            "spots_per_member": len(dataset.states), "ig_steps": args.ig_steps,
            "heldout_spot_rows": int(len(heldout_spots)),
            "heldout_biological_units": int(unit_ensemble["biological_unit"].nunique()),
            "fold_parameter_fits": len(fold_checkpoints),
            "parameter_stability_rank": effective_rank,
            "parameter_stability_condition_number": condition_number,
            "maximum_absolute_parameter_correlation": float(np.nanmax(np.abs(off_diagonal))),
            "parameter_stability_singular_values": singular.tolist(),
            "local_sensitivity_effective_rank": local_rank,
            "maximum_absolute_local_sensitivity_profile_correlation": local_max_correlation,
            "local_sensitivity_singular_values": local_singular.tolist(),
            "identifiability_scope": "real-data cross-seed stability only; ground-truth identifiability is evaluated in E5",
            "statistical_unit": "cross-fitted held-out biological unit; only model-unit pairs assigned to the test split enter pathway association tests",
            "descriptive_scope": "all train, validation, and test roles are retained in the all-role spot and biological-unit exports",
            "parameter_stability_scope": "10 E1 initialization/split fits plus 15 E4 grouped-fold fits",
        },
        args.output_dir / "run_manifest.json",
    )
    print(json.dumps({"status": "complete", "members": len(checkpoints), "sections": len(sections), "attribution_rows": len(attribution)}, indent=2))


if __name__ == "__main__":
    main()