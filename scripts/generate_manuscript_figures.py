from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from cardiogb.data.state_dataset import StateDataset


STATE_LABELS = {
    "I": "Inflammation",
    "A": "Activation",
    "F": "Fibroblast/ECM",
    "C": "CM regeneration",
    "V": "Vascularisation",
    "M": "Mature myocardium",
}
HUMAN_STATE_LABELS = {
    "inflammation": "Inflammation",
    "activation": "Activation",
    "fibroblast_ecm": "Fibroblast/ECM",
    "cardiomyocyte_regeneration": "CM regeneration",
    "vascularisation": "Vascularisation",
    "mature_myocardium": "Mature myocardium",
}
COLORS = ["#276FBF", "#D1495B", "#2A9D8F", "#F4A261", "#7B2CBF", "#607D3B"]


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required final figure input is missing: {path}")
    return path


def save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def style(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)


def mean_ci(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().to_numpy(float)
    mean = float(values.mean())
    half_width = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, half_width


def figure_1(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(13.0, 4.8))
    axis.set_axis_off()
    boxes = [
        (0.02, "Spatial transcriptomics\nSix pathway states"),
        (0.215, "Bounded mechanistic ODE\nInterpretable rates"),
        (0.41, "Scaled graph residual\nLocal missing dynamics"),
        (0.605, "Projected RK4\nStable forecasts"),
        (0.80, "Registered validation\nE1–E11"),
    ]
    box_width = 0.17
    for index, (x, label) in enumerate(boxes):
        color = "#E8F1FA" if index != 2 else "#FBEAEC"
        axis.add_patch(plt.Rectangle((x, 0.38), box_width, 0.25, facecolor=color, edgecolor="#243447", linewidth=1.5))
        axis.text(x + box_width / 2, 0.505, label, ha="center", va="center", fontsize=8.5, linespacing=1.25)
        if index < len(boxes) - 1:
            axis.annotate("", xy=(boxes[index + 1][0] - 0.004, 0.505), xytext=(x + box_width + 0.004, 0.505), arrowprops={"arrowstyle": "->", "lw": 1.5})
    axis.text(0.50, 0.78, r"$\dot{x}=f_{mech}(x;\theta)+s\odot\tanh(f_{GNN}(x,G))$", ha="center", fontsize=14)
    axis.text(0.50, 0.16, "Grouped biological-unit splits  •  multi-seed inference  •  frozen zebrafish→mouse transfer  •  human-MI translation", ha="center", fontsize=9)
    axis.set_title("CardioGB study design and grey-box architecture", fontsize=13, pad=12)
    save(figure, output / "Figure_1_framework_overview.png")


def supplementary_data_qc(dataset: StateDataset, output: Path) -> None:
    frame = pd.DataFrame(dataset.states, columns=dataset.state_names)
    frame["stage"] = dataset.times
    frame["section"] = dataset.sections
    unit = frame.groupby(["stage", "section"], observed=True)[list(dataset.state_names)].mean().reset_index()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    for color, state in zip(COLORS, dataset.state_names):
        summary = unit.groupby("stage", observed=True)[state].agg(["mean", "sem"]).reset_index()
        axes[0].plot(summary["stage"], summary["mean"], marker="o", label=STATE_LABELS[state], color=color)
        axes[0].fill_between(summary["stage"], summary["mean"] - 1.96 * summary["sem"], summary["mean"] + 1.96 * summary["sem"], alpha=0.13, color=color)
    axes[0].set(xlabel="Days post-amputation", ylabel="Pathway state score", ylim=(0, 1), title="A  Biological-unit pathway trajectories")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    counts = pd.Series(dataset.times).value_counts().sort_index()
    axes[1].bar(counts.index.astype(str), counts.values, color="#637A91")
    axes[1].set(xlabel="Days post-amputation", ylabel="Spots", title="B  Observations by stage")
    for axis in axes:
        style(axis)
    save(figure, output / "Supplementary_Figure_S1_data_and_pathway_qc.png")


def figure_2(root: Path, output: Path) -> None:
    data = pd.read_csv(require(root / "final_full_multiseed" / "tables" / "benchmark_metrics.csv"))
    metrics = [("mmd", "MMD"), ("sliced_wasserstein", "Sliced Wasserstein"), ("moment_error", "Moment error")]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.0))
    order = ["cardiogb", "graph_neural_ode", "neural_ode", "mechanistic_ode", "persistence"]
    seed_level = data.groupby(["seed", "model"], observed=True)[[x[0] for x in metrics]].mean().reset_index()
    for panel, (axis, (metric, label)) in enumerate(zip(axes.flat[:3], metrics)):
        summary = seed_level.groupby("model", observed=True)[metric].apply(mean_ci)
        present = [name for name in order if name in summary.index]
        means = [summary[name][0] for name in present]
        errors = [summary[name][1] for name in present]
        display = [name.replace("_", " ") for name in present[::-1]]
        colors = ["#D1495B" if name == "cardiogb" else "#78909C" for name in present[::-1]]
        axis.barh(display, means[::-1], xerr=errors[::-1], color=colors)
        axis.set(xlabel=f"{label} (lower is better)", title=chr(65 + panel))
        style(axis)
    transition_order = sorted(data["transition"].unique(), key=lambda value: tuple(float(x) for x in value.split("_to_")))
    heat = data.pivot_table(index="model", columns="transition", values="sliced_wasserstein", aggfunc="mean").reindex(index=order, columns=transition_order)
    image = axes[1, 1].imshow(heat, aspect="auto", cmap="viridis_r")
    axes[1, 1].set_xticks(range(len(heat.columns)), [x.replace("_to_", "→") for x in heat.columns], rotation=45, ha="right", fontsize=7)
    axes[1, 1].set_yticks(range(len(heat.index)), [x.replace("_", " ") for x in heat.index], fontsize=7)
    axes[1, 1].set(xlabel="Held-out transition (days)", title="D  Transition-resolved sliced Wasserstein")
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046, label="Error")
    figure.suptitle("E1: grouped-holdout predictive benchmark (mean ± 95% seed CI)")
    figure.tight_layout()
    save(figure, output / "Figure_2_main_predictive_benchmark.png")

def figure_3(root: Path, output: Path) -> None:
    e2 = pd.read_csv(require(root / "e2_interpolation_revised" / "tables" / "all_metrics.csv"))
    calibration = pd.read_csv(require(root / "e3_extrapolation_horizon_calibrated" / "tables" / "all_metrics.csv"))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    e2_seed = e2.groupby(["case", "seed"], observed=True)["sliced_wasserstein"].mean().reset_index()
    for case, subset in e2_seed.groupby("case", observed=True):
        axes[0].scatter(np.repeat(case, len(subset)), subset["sliced_wasserstein"], alpha=0.55, color="#78909C")
    e2_summary = e2_seed.groupby("case", observed=True)["sliced_wasserstein"].mean()
    axes[0].plot(e2_summary.index, e2_summary.values, color="#D1495B", marker="o")
    axes[0].set(xlabel="Held-out stage (days)", ylabel="Sliced Wasserstein", title="A  E2 interpolation")
    for axis, metric, label, title in (
        (axes[1], "sliced_wasserstein", "Sliced Wasserstein", "B  E3 extrapolation"),
        (axes[2], "moment_error", "Moment error", "C  E3 stability"),
    ):
        for column, method, color, marker in (
            (metric, "calibrated CardioGB", "#D1495B", "o"),
            (f"raw_{metric}", "raw CardioGB", "#3949AB", "s"),
            (f"persistence_{metric}", "persistence", "#78909C", "^"),
        ):
            summary = calibration.groupby("horizon_days", observed=True)[column].mean().reset_index()
            axis.plot(summary["horizon_days"], summary[column], marker=marker, label=method, color=color)
        axis.set_yscale("symlog", linthresh=1e-3)
        axis.set(xlabel="Forecast horizon (days)", ylabel=label, title=title)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes:
        style(axis)
    save(figure, output / "Figure_3_interpolation_extrapolation.png")


def figure_4(root: Path, output: Path) -> None:
    e5 = pd.read_csv(require(root / "synthetic_recovery_full" / "tables" / "e5_parameter_recovery.csv"))
    e6 = pd.read_csv(require(root / "synthetic_recovery_full" / "tables" / "e6_hidden_recovery.csv"))
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    cmap = plt.get_cmap("viridis")
    noise_values = sorted(e5["noise_std"].unique())
    for index, noise in enumerate(noise_values):
        subset = e5[e5["noise_std"] == noise]
        axes[0, 0].scatter(subset["true_value"], subset["inferred_value"], s=18, alpha=0.55, color=cmap(index / max(1, len(noise_values) - 1)), label=f"σ={noise:g}")
    limits = [min(e5["true_value"].min(), e5["inferred_value"].min()), max(e5["true_value"].max(), e5["inferred_value"].max())]
    axes[0, 0].plot(limits, limits, "--", color="#263238", linewidth=1)
    axes[0, 0].set(xlabel="True parameter", ylabel="Inferred parameter", title="A  True versus inferred rates")
    axes[0, 0].legend(frameon=False, fontsize=7)
    recovery = e5.groupby(["noise_std", "seed"], observed=True)["parameter_correlation"].first().reset_index()
    summary = recovery.groupby("noise_std", observed=True)["parameter_correlation"].agg(["mean", "sem"]).reset_index()
    axes[0, 1].errorbar(summary["noise_std"], summary["mean"], yerr=1.96 * summary["sem"], marker="o", capsize=3, color="#276FBF")
    axes[0, 1].axhline(0, color="#90A4AE", linewidth=0.8)
    axes[0, 1].set(xlabel="Noise SD", ylabel="Parameter correlation", title="B  E5 recovery across noise")
    for axis, metric, ylabel, title, color in (
        (axes[1, 0], "correlation", "Hidden-field correlation", "C  E6 omitted-mechanism recovery", "#2A9D8F"),
        (axes[1, 1], "rmse", "Hidden-field RMSE", "D  E6 recovery error", "#D1495B"),
    ):
        summary = e6.groupby("noise_std", observed=True)[metric].agg(["mean", "sem"]).reset_index()
        axis.errorbar(summary["noise_std"], summary["mean"], yerr=1.96 * summary["sem"], marker="o", capsize=3, color=color)
        axis.set(xlabel="Noise SD", ylabel=ylabel, title=title)
    for axis in axes.flat:
        style(axis)
    figure.suptitle("Synthetic parameter and hidden-mechanism recovery")
    figure.tight_layout()
    save(figure, output / "Figure_4_synthetic_system_identification.png")


def figure_5(dataset: StateDataset, root: Path, output: Path) -> None:
    base = root / "e7_full_interpretation" / "tables"
    spots = pd.read_parquet(require(base / "mi_spots_all_members.parquet"))
    domains = pd.read_csv(require(base / "mi_domain_biological_units.csv"))
    stage_summary = pd.read_csv(require(base / "mi_stage_bootstrap.csv"))
    heldout = spots[spots["split_role"] == "test"].copy()
    if heldout.empty:
        raise ValueError("Figure 5 requires held-out E7 spot-level insufficiency")
    averaged = heldout.groupby("spot_index", observed=True)["mi"].mean().reset_index()
    averaged["spot_index"] = averaged["spot_index"].astype(int)
    indices = averaged["spot_index"].to_numpy()
    averaged["stage_days"] = dataset.times[indices]
    averaged["section"] = dataset.sections[indices].astype(str)
    averaged["x"] = dataset.coordinates[indices, 0]
    averaged["y"] = dataset.coordinates[indices, 1]
    requested = [0.0, 1.0, 3.0, 7.0, 14.0, 28.0]
    selected = []
    for stage in requested:
        stage_frame = averaged[np.isclose(averaged["stage_days"], stage)]
        if stage_frame.empty:
            raise ValueError(f"no held-out E7 spots at stage {stage:g}")
        section = stage_frame.groupby("section", observed=True).size().sort_values(ascending=False).index[0]
        selected.append(stage_frame[stage_frame["section"] == section])
    all_mi = pd.concat(selected, ignore_index=True)["mi"].to_numpy(float)
    vmin, vmax = np.quantile(all_mi, [0.02, 0.98])
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    map_axes = list(axes.flat[:6])
    image = None
    for axis, stage, frame in zip(map_axes, requested, selected):
        image = axis.scatter(frame["x"], frame["y"], c=frame["mi"], s=4, cmap="magma", vmin=vmin, vmax=vmax, linewidths=0)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_title(f"{stage:g} dpa | {frame['section'].iloc[0]}", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    domain_unit = domains.groupby(["biological_unit", "stage_days", "domain"], observed=True)["mi"].mean().reset_index()
    domain_order = domain_unit.groupby("domain", observed=True)["biological_unit"].nunique().sort_values(ascending=False).head(10).index
    axes[1, 2].boxplot(
        [domain_unit.loc[domain_unit["domain"] == name, "mi"] for name in domain_order],
        tick_labels=domain_order,
        vert=False,
        showfliers=False,
    )
    axes[1, 2].set(xlabel="Mechanistic insufficiency", title="G  Held-out tissue domains")
    axes[1, 2].tick_params(axis="y", labelsize=6)
    axes[1, 3].plot(stage_summary["stage_days"], stage_summary["mi_mean"], marker="o", color="#7B2CBF")
    axes[1, 3].fill_between(stage_summary["stage_days"], stage_summary["ci_lower"], stage_summary["ci_upper"], color="#7B2CBF", alpha=0.18)
    axes[1, 3].set(xlabel="Days post-amputation", ylabel="Mechanistic insufficiency", title="H  Biological-unit stage profile")
    for axis in (axes[1, 2], axes[1, 3]):
        style(axis)
    figure.suptitle("Held-out spatial and tissue-domain mechanistic insufficiency")
    figure.subplots_adjust(top=0.90, right=0.90, wspace=0.35, hspace=0.28)
    color_axis = figure.add_axes([0.92, 0.54, 0.012, 0.30])
    figure.colorbar(image, cax=color_axis, label="Mechanistic insufficiency")
    save(figure, output / "Figure_5_mechanistic_insufficiency_maps.png")

def figure_6(root: Path, output: Path) -> None:
    base = root / "e7_full_interpretation" / "tables"
    stage = pd.read_csv(require(base / "mi_stage_bootstrap.csv"))
    attribution = pd.read_csv(require(base / "residual_attribution_heldout.csv"))
    sensitivity = pd.read_csv(require(base / "parameter_local_sensitivity.csv"))
    stability = pd.read_csv(require(base / "parameter_stability.csv"))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.0))
    axes[0, 0].plot(stage["stage_days"], stage["mi_mean"], marker="o", color="#7B2CBF")
    axes[0, 0].fill_between(stage["stage_days"], stage["ci_lower"], stage["ci_upper"], color="#7B2CBF", alpha=0.18)
    axes[0, 0].set(xlabel="Days post-amputation", ylabel="Mechanistic insufficiency", title="A  Held-out stage profile")
    matrix = attribution.groupby(["target_state", "input_state"], observed=True)["mean_absolute_integrated_gradient"].mean().unstack(fill_value=0)
    image = axes[0, 1].imshow(matrix, aspect="auto", cmap="magma")
    axes[0, 1].set_xticks(range(len(matrix.columns)), matrix.columns)
    axes[0, 1].set_yticks(range(len(matrix.index)), matrix.index)
    axes[0, 1].set(xlabel="Input pathway", ylabel="Residual target", title="B  Neighbour-to-target attribution")
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046)
    stage_effect = attribution.groupby(["stage_days", "target_state"], observed=True)["mean_absolute_integrated_gradient"].mean().reset_index()
    for color, state_name in zip(COLORS, STATE_LABELS):
        subset = stage_effect[stage_effect["target_state"] == state_name]
        axes[1, 0].plot(subset["stage_days"], subset["mean_absolute_integrated_gradient"], marker="o", label=state_name, color=color)
    axes[1, 0].set(xlabel="Days post-amputation", ylabel="Mean absolute attribution", title="C  Stage-specific residual effects")
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)
    shown = stability.replace([np.inf, -np.inf], np.nan).dropna(subset=["coefficient_of_variation"]).sort_values("coefficient_of_variation").tail(12)
    axes[1, 1].barh(shown["parameter"], shown["coefficient_of_variation"], color="#607D3B")
    axes[1, 1].set(xlabel="Cross-fit coefficient of variation", title="D  Parameter stability")
    sensitivity_rank = sensitivity.groupby("parameter", observed=True)["mean_absolute_local_sensitivity"].mean().sort_values(ascending=False)
    if len(sensitivity_rank):
        axes[1, 1].text(0.98, 0.02, f"Most locally influential: {sensitivity_rank.index[0]}", transform=axes[1, 1].transAxes, ha="right", fontsize=7)
    for axis in (axes[0, 0], axes[1, 0], axes[1, 1]):
        style(axis)
    figure.suptitle("Residual biological interpretation and real-data identifiability")
    figure.tight_layout()
    save(figure, output / "Figure_6_residual_biological_interpretation.png")


def figure_7(root: Path, output: Path) -> None:
    external = root / "external_predictive_validation_revised"
    states = pd.read_csv(require(external / "tables" / "state_mean_predictions.csv"))
    tests = pd.read_csv(require(external / "tables" / "uncertainty_inferential_tests.csv"))
    e4 = pd.read_csv(require(root / "e4_group_cv_full" / "tables" / "all_metrics.csv"))
    e8 = pd.read_csv(require(root / "final_full_ablations" / "tables" / "ablation_metrics.csv"))
    direct = states[states["protocol"] == "mouse_zero_shot_direct_horizon"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.0))
    for color, (state_name, part) in zip(COLORS, direct.groupby("state", observed=True)):
        axes[0, 0].scatter(part["horizon_days"], part["ensemble_std"], label=state_name, color=color)
        axes[0, 1].scatter(part["ensemble_std"], part["absolute_error"], label=state_name, color=color)
    axes[0, 0].set(xlabel="Horizon (days)", ylabel="Ensemble SD", title="A  Uncertainty versus horizon")
    axes[0, 1].set(xlabel="Ensemble SD", ylabel="Absolute error", title="B  Uncertainty versus error")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    significant = int((tests["p_adjust_bh"] < 0.05).sum())
    axes[0, 1].text(0.98, 0.03, f"BH-significant tests: {significant}/{len(tests)}", transform=axes[0, 1].transAxes, ha="right", fontsize=8)
    fold = e4.groupby(["case", "seed"], observed=True)["sliced_wasserstein"].mean().reset_index()
    fold_cases = sorted(fold["case"].unique())
    axes[1, 0].boxplot([fold.loc[fold["case"] == case, "sliced_wasserstein"] for case in fold_cases], tick_labels=[str(case) for case in fold_cases])
    axes[1, 0].set(xlabel="Held-out biological-replicate fold", ylabel="Sliced Wasserstein", title="C  E4 grouped cross-validation")
    e8_seed = e8.groupby(["ablation", "seed"], observed=True)["sliced_wasserstein"].mean().reset_index()
    order = e8_seed.groupby("ablation", observed=True)["sliced_wasserstein"].mean().sort_values().index
    axes[1, 1].boxplot([e8_seed.loc[e8_seed["ablation"] == name, "sliced_wasserstein"] for name in order], tick_labels=order, vert=False, showfliers=False)
    axes[1, 1].set(xlabel="Sliced Wasserstein", title="D  E8 component ablations")
    axes[1, 1].tick_params(axis="y", labelsize=6)
    for axis in axes.flat:
        style(axis)
    figure.suptitle("Uncertainty, grouped robustness, and ablation evidence")
    figure.tight_layout()
    save(figure, output / "Figure_7_uncertainty_robustness_ablations.png")


def figure_8(root: Path, output: Path) -> None:
    conservation_root = root / "mouse_validation_revised"
    conservation = pd.read_csv(require(conservation_root / "pathway_conservation.csv"))
    matched = pd.read_csv(require(conservation_root / "matched_stage_scores.csv"))
    external_root = root / "external_predictive_validation_revised"
    metrics = pd.read_csv(require(external_root / "metrics" / "external_prediction.csv"))
    states = pd.read_csv(require(external_root / "tables" / "state_mean_predictions.csv"))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.0))
    conservation = conservation.set_index("state").reindex(STATE_LABELS)
    axes[0, 0].bar([STATE_LABELS[x] for x in conservation.index], conservation["spearman"], color=COLORS)
    axes[0, 0].axhline(0, color="#78909C", linewidth=0.8)
    axes[0, 0].tick_params(axis="x", rotation=35, labelsize=7)
    axes[0, 0].set(ylabel="Spearman correlation", ylim=(-1.05, 1.05), title="A  Matched-phase pathway conservation")
    for species, linestyle in (("zebrafish", "-"), ("mouse", "--")):
        subset = matched[matched["species"] == species].sort_values("stage_days")
        phase = np.arange(1, len(subset) + 1)
        for color, state_name in zip(COLORS, STATE_LABELS):
            axes[0, 1].plot(phase, subset[state_name], linestyle=linestyle, marker="o", color=color, alpha=0.85, label=state_name if species == "zebrafish" else None)
    axes[0, 1].set_xticks([1, 2, 3, 4])
    axes[0, 1].set(xlabel="Matched repair phase", ylabel="Mean pathway score", title="B  Cross-species repair dynamics")
    pathway_legend = axes[0, 1].legend(frameon=False, fontsize=6, ncol=2, loc="upper right", title="Pathway", title_fontsize=7)
    axes[0, 1].add_artist(pathway_legend)
    species_handles = [
        Line2D([0], [0], color="#455A64", linewidth=1.8, linestyle="-", label="Zebrafish"),
        Line2D([0], [0], color="#455A64", linewidth=1.8, linestyle="--", label="Mouse"),
    ]
    axes[0, 1].legend(handles=species_handles, frameon=False, fontsize=7, loc="lower left", title="Species", title_fontsize=7)
    mouse = metrics[metrics["protocol"].str.startswith("mouse")].copy()
    positions = np.arange(len(mouse))
    axes[1, 0].plot(positions, mouse["sliced_wasserstein"], marker="o", label="CardioGB", color="#D1495B")
    axes[1, 0].plot(positions, mouse["persistence_sliced_wasserstein"], marker="s", label="Persistence", color="#78909C")
    axes[1, 0].set_xticks(positions, mouse["transition"], rotation=55, ha="right", fontsize=6)
    axes[1, 0].set(ylabel="Sliced Wasserstein", title="C  Frozen zero-shot mouse prediction")
    axes[1, 0].legend(frameon=False, fontsize=8)
    direct = states[states["protocol"] == "mouse_zero_shot_direct_horizon"]
    for color, (state_name, part) in zip(COLORS, direct.groupby("state", observed=True)):
        axes[1, 1].errorbar(part["observed_mean"], part["prediction_mean"], yerr=part["interval_radius"], fmt="o", alpha=0.7, color=color, label=state_name)
    lo = min(direct["observed_mean"].min(), direct["prediction_mean"].min())
    hi = max(direct["observed_mean"].max(), direct["prediction_mean"].max())
    axes[1, 1].plot([lo, hi], [lo, hi], "--", color="#263238", linewidth=1)
    axes[1, 1].set(xlabel="Observed mouse state mean", ylabel="Frozen CardioGB prediction", title="D  State-level predictions and conformal intervals")
    axes[1, 1].legend(frameon=False, fontsize=6, ncol=2)
    for axis in axes.flat:
        style(axis)
    figure.suptitle("Mouse cross-species conservation and external predictive validation")
    figure.tight_layout()
    save(figure, output / "Figure_8_mouse_external_validation.png")


def supplementary_human(root: Path, output: Path) -> None:
    base = root / "human_snatac_validation_revised" / "tables"
    patient = pd.read_csv(require(base / "patient_pathway_accessibility.csv"))
    posthoc = pd.read_csv(require(base / "patient_group_posthoc_effects.csv"))
    pathways = [name for name in HUMAN_STATE_LABELS if name in patient.columns]
    if len(pathways) != 6:
        missing = sorted(set(HUMAN_STATE_LABELS) - set(pathways))
        raise ValueError(f"human supplementary figure requires all six pathway columns; missing {missing}")
    groups = sorted(patient["patient_group"].astype(str).unique())
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=False)
    for axis, pathway, color in zip(axes.flat, pathways, COLORS):
        values = [patient.loc[patient["patient_group"].astype(str) == group, pathway] for group in groups]
        axis.boxplot(values, tick_labels=groups, showfliers=False)
        for index, series in enumerate(values, start=1):
            axis.scatter(np.repeat(index, len(series)), series, s=13, alpha=0.65, color=color)
        significant = posthoc[(posthoc["pathway"] == pathway) & (posthoc["p_adjust_bh_global"] < 0.05)]
        axis.set_title(f"{HUMAN_STATE_LABELS[pathway]} | BH contrasts: {len(significant)}", fontsize=9)
        axis.tick_params(axis="x", rotation=35, labelsize=7)
        style(axis)
    figure.suptitle("Human-MI snATAC pathway accessibility at the patient level")
    figure.tight_layout()
    save(figure, output / "Supplementary_Figure_S2_human_MI_snatac.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate registered CardioGB manuscript Figures 1–8")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/manuscript"))
    args = parser.parse_args()
    dataset = StateDataset.load(args.data)
    figure_1(args.output_dir)
    figure_2(args.results_root, args.output_dir)
    figure_3(args.results_root, args.output_dir)
    figure_4(args.results_root, args.output_dir)
    figure_5(dataset, args.results_root, args.output_dir)
    figure_6(args.results_root, args.output_dir)
    figure_7(args.results_root, args.output_dir)
    figure_8(args.results_root, args.output_dir)
    supplementary_data_qc(dataset, args.output_dir)
    supplementary_human(args.results_root, args.output_dir)
    print(f"generated registered Figures 1–8 and supplementary figures in {args.output_dir}")


if __name__ == "__main__":
    main()
