from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cardiogb.data.state_dataset import StateDataset


STATE_LABELS = {
    "I": "Inflammation",
    "A": "Activation",
    "F": "Fibroblast/ECM",
    "C": "CM regeneration",
    "V": "Vascularisation",
    "M": "Mature myocardium",
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
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def figure_1(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 4.6))
    axis.set_axis_off()
    boxes = [
        (0.02, "Spatial transcriptomics\n6 pathway states"),
        (0.22, "Bounded mechanistic ODE\ninterpretable parameters"),
        (0.43, "Scaled graph residual\nlocal missing dynamics"),
        (0.64, "Projected RK4\nstable forecasts"),
        (0.83, "Validation\nE1–E10"),
    ]
    for index, (x, label) in enumerate(boxes):
        color = "#E8F1FA" if index != 2 else "#FBEAEC"
        axis.add_patch(plt.Rectangle((x, 0.38), 0.15, 0.25, facecolor=color, edgecolor="#243447", linewidth=1.5))
        axis.text(x + 0.075, 0.505, label, ha="center", va="center", fontsize=9)
        if index < len(boxes) - 1:
            axis.annotate("", xy=(boxes[index + 1][0], 0.505), xytext=(x + 0.15, 0.505), arrowprops={"arrowstyle": "->", "lw": 1.5})
    axis.text(0.33, 0.78, r"$\dot{x}=f_{mech}(x;\theta)+s\odot\tanh(f_{GNN}(x,G))$", ha="center", fontsize=14)
    axis.text(0.50, 0.16, "Grouped biological-unit splits  •  multi-seed inference  •  zebrafish→mouse frozen transfer  •  human-MI translation", ha="center", fontsize=9)
    axis.set_title("CardioGB study design and grey-box architecture", fontsize=13, pad=12)
    save(figure, output / "Figure_1_study_design.png")


def figure_2(dataset: StateDataset, output: Path) -> None:
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
    save(figure, output / "Figure_2_data_and_pathway_qc.png")


def figure_3(root: Path, output: Path) -> None:
    data = pd.read_csv(require(root / "final_full_multiseed" / "tables" / "benchmark_metrics.csv"))
    metrics = [("mmd", "MMD"), ("sliced_wasserstein", "Sliced Wasserstein"), ("moment_error", "Moment error")]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    order = ["cardiogb", "graph_neural_ode", "neural_ode", "mechanistic_ode", "persistence"]
    seed_level = data.groupby(["seed", "model"], observed=True)[[x[0] for x in metrics]].mean().reset_index()
    for axis, (metric, label) in zip(axes, metrics):
        summary = seed_level.groupby("model", observed=True)[metric].apply(mean_ci)
        present = [name for name in order if name in summary.index]
        means = [summary[name][0] for name in present]
        errors = [summary[name][1] for name in present]
        axis.barh(present[::-1], means[::-1], xerr=errors[::-1], color=["#D1495B" if name == "cardiogb" else "#78909C" for name in present[::-1]])
        axis.set_xlabel(f"{label} (lower is better)")
        style(axis)
    axes[0].set_title("A")
    axes[1].set_title("B")
    axes[2].set_title("C")
    figure.suptitle("E1: grouped-holdout predictive benchmark (mean ± 95% seed CI)")
    save(figure, output / "Figure_3_E1_benchmark.png")


def figure_4(root: Path, output: Path) -> None:
    e2 = pd.read_csv(require(root / "e2_interpolation_revised" / "tables" / "all_metrics.csv"))
    e3 = pd.read_csv(require(root / "e3_extrapolation_revised" / "tables" / "all_metrics.csv"))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    e2_seed = e2.groupby(["case", "seed"], observed=True)["sliced_wasserstein"].mean().reset_index()
    for case, subset in e2_seed.groupby("case", observed=True):
        axes[0].scatter(np.repeat(case, len(subset)), subset["sliced_wasserstein"], alpha=0.6)
    e2_summary = e2_seed.groupby("case")["sliced_wasserstein"].mean()
    axes[0].plot(e2_summary.index, e2_summary.values, color="#D1495B", marker="o")
    axes[0].set(xlabel="Held-out stage (days)", ylabel="Sliced Wasserstein", title="A  E2 interpolation")
    for axis, metric, label in ((axes[1], "sliced_wasserstein", "Sliced Wasserstein"), (axes[2], "moment_error", "Moment error")):
        summary = e3.groupby("horizon_days", observed=True)[metric].agg(["median", "mean"]).reset_index()
        axis.plot(summary["horizon_days"], summary["median"], marker="o", label="median")
        axis.plot(summary["horizon_days"], summary["mean"], marker="s", label="mean", alpha=0.75)
        axis.set_yscale("symlog", linthresh=1e-3)
        axis.set(xlabel="Forecast horizon (days)", ylabel=label)
        axis.legend(frameon=False, fontsize=8)
    axes[1].set_title("B  E3 extrapolation")
    axes[2].set_title("C  E3 stability")
    for axis in axes:
        style(axis)
    save(figure, output / "Figure_4_E2_E3_temporal_generalization.png")


def figure_5(root: Path, output: Path) -> None:
    e4 = pd.read_csv(require(root / "e4_group_cv_full" / "tables" / "all_metrics.csv"))
    e5 = pd.read_csv(require(root / "synthetic_recovery_full" / "tables" / "e5_parameter_recovery.csv"))
    e6 = pd.read_csv(require(root / "synthetic_recovery_full" / "tables" / "e6_hidden_recovery.csv"))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    fold = e4.groupby(["case", "seed"], observed=True)["sliced_wasserstein"].mean().reset_index()
    axes[0].boxplot([part["sliced_wasserstein"] for _, part in fold.groupby("case")], labels=[str(x) for x in sorted(fold["case"].unique())])
    axes[0].set(xlabel="Held-out biological replicate", ylabel="Sliced Wasserstein", title="A  E4 grouped CV")
    recovery = e5.groupby(["noise_std", "seed"], observed=True)["parameter_correlation"].mean().reset_index()
    axes[1].boxplot([part["parameter_correlation"] for _, part in recovery.groupby("noise_std")], labels=[str(x) for x in sorted(recovery["noise_std"].unique())])
    axes[1].set(xlabel="Noise SD", ylabel="Parameter correlation", title="B  E5 parameter recovery")
    hidden = e6.groupby("noise_std", observed=True)["correlation"].agg(["mean", "sem"]).reset_index()
    axes[2].errorbar(hidden["noise_std"], hidden["mean"], yerr=1.96 * hidden["sem"], marker="o", capsize=3)
    axes[2].set(xlabel="Noise SD", ylabel="Hidden-mechanism correlation", title="C  E6 residual recovery")
    for axis in axes:
        style(axis)
    save(figure, output / "Figure_5_E4_E5_E6_robustness.png")


def figure_6(root: Path, output: Path) -> None:
    base = root / "e7_full_interpretation" / "tables"
    stage = pd.read_csv(require(base / "mi_stage_bootstrap.csv"))
    attribution = pd.read_csv(require(base / "residual_attribution_all_sections.csv"))
    stability = pd.read_csv(require(base / "parameter_stability.csv"))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].plot(stage["stage_days"], stage["mi_mean"], marker="o", color="#7B2CBF")
    axes[0].fill_between(stage["stage_days"], stage["ci_lower"], stage["ci_upper"], color="#7B2CBF", alpha=0.18)
    axes[0].set(xlabel="Days post-amputation", ylabel="Mechanistic insufficiency", title="A  E7 insufficiency")
    matrix = attribution.groupby(["target_state", "input_state"], observed=True)["mean_absolute_integrated_gradient"].mean().unstack(fill_value=0)
    image = axes[1].imshow(matrix, aspect="auto", cmap="magma")
    axes[1].set_xticks(range(len(matrix.columns)), matrix.columns)
    axes[1].set_yticks(range(len(matrix.index)), matrix.index)
    axes[1].set(xlabel="Input pathway", ylabel="Residual target", title="B  Residual attribution")
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    shown = stability.sort_values("coefficient_of_variation").tail(12)
    axes[2].barh(shown["parameter"], shown["coefficient_of_variation"], color="#607D3B")
    axes[2].set(xlabel="Cross-seed coefficient of variation", title="C  Parameter stability")
    for axis in (axes[0], axes[2]):
        style(axis)
    save(figure, output / "Figure_6_E7_interpretability_identifiability.png")


def figure_7(root: Path, output: Path) -> None:
    base = root / "external_predictive_validation_revised"
    metrics = pd.read_csv(require(base / "metrics" / "external_prediction.csv"))
    states = pd.read_csv(require(base / "tables" / "state_mean_predictions.csv"))
    tests = pd.read_csv(require(base / "tables" / "uncertainty_inferential_tests.csv"))
    mouse = metrics[metrics["protocol"].str.startswith("mouse")].copy()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    positions = np.arange(len(mouse))
    axes[0].plot(positions, mouse["sliced_wasserstein"], marker="o", label="CardioGB")
    axes[0].plot(positions, mouse["persistence_sliced_wasserstein"], marker="s", label="Persistence")
    axes[0].set_xticks(positions, mouse["transition"], rotation=60, ha="right", fontsize=7)
    axes[0].set(ylabel="Sliced Wasserstein", title="A  Frozen mouse transfer")
    axes[0].legend(frameon=False, fontsize=8)
    direct = states[states["protocol"] == "mouse_zero_shot_direct_horizon"]
    for state, part in direct.groupby("state", observed=True):
        axes[1].scatter(part["horizon_days"], part["ensemble_std"], label=state)
        axes[2].scatter(part["ensemble_std"], part["absolute_error"], label=state)
    axes[1].set(xlabel="Horizon (days)", ylabel="Ensemble SD", title="B  Uncertainty vs horizon")
    axes[2].set(xlabel="Ensemble SD", ylabel="Absolute error", title="C  Uncertainty vs error")
    significant = int((tests["p_adjust_bh"] < 0.05).sum())
    axes[2].text(0.98, 0.03, f"BH-significant tests: {significant}/{len(tests)}", transform=axes[2].transAxes, ha="right", fontsize=8)
    for axis in axes:
        style(axis)
    save(figure, output / "Figure_7_ensemble_external_uncertainty.png")


def figure_8(root: Path, output: Path) -> None:
    base = root / "human_snatac_validation_revised" / "tables"
    patient = pd.read_csv(require(base / "patient_pathway_accessibility.csv"))
    posthoc = pd.read_csv(require(base / "patient_group_posthoc_effects.csv"))
    pathways = [name for name in STATE_LABELS if name in patient.columns]
    groups = sorted(patient["patient_group"].astype(str).unique())
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=False)
    for axis, pathway, color in zip(axes.flat, pathways, COLORS):
        values = [patient.loc[patient["patient_group"].astype(str) == group, pathway] for group in groups]
        axis.boxplot(values, labels=groups, showfliers=False)
        for index, series in enumerate(values, start=1):
            axis.scatter(np.repeat(index, len(series)), series, s=13, alpha=0.65, color=color)
        significant = posthoc[(posthoc["pathway"] == pathway) & (posthoc["p_adjust_bh_global"] < 0.05)]
        axis.set_title(f"{STATE_LABELS[pathway]}  |  BH contrasts: {len(significant)}", fontsize=9)
        axis.tick_params(axis="x", rotation=35, labelsize=7)
        style(axis)
    figure.suptitle("Human-MI snATAC pathway accessibility at the patient level")
    figure.tight_layout()
    save(figure, output / "Figure_8_human_MI_snatac.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final CardioGB manuscript Figures 1–8")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/manuscript"))
    args = parser.parse_args()
    dataset = StateDataset.load(args.data)
    figure_1(args.output_dir)
    figure_2(dataset, args.output_dir)
    figure_3(args.results_root, args.output_dir)
    figure_4(args.results_root, args.output_dir)
    figure_5(args.results_root, args.output_dir)
    figure_6(args.results_root, args.output_dir)
    figure_7(args.results_root, args.output_dir)
    figure_8(args.results_root, args.output_dir)
    print(f"generated Figures 1–8 in {args.output_dir}")


if __name__ == "__main__":
    main()
