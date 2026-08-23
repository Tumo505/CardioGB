from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


START = "<!-- AUTO_RESULTS_START -->"
END = "<!-- AUTO_RESULTS_END -->"
METRICS = {
    "mmd": "MMD",
    "sliced_wasserstein": "sliced-Wasserstein distance",
    "moment_error": "moment error",
}


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required manuscript result missing: {path}")
    return path


def number(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def benchmark_text(root: Path) -> str:
    frame = pd.read_csv(require(root / "final_full_multiseed" / "tables" / "benchmark_metrics.csv"))
    seed = frame.groupby(["seed", "model"], observed=True)[list(METRICS)].mean().reset_index()
    mean = seed.groupby("model", observed=True)[list(METRICS)].mean()
    comparisons = []
    neural = [name for name in ("neural_ode", "graph_neural_ode") if name in mean.index]
    for metric, label in METRICS.items():
        cardiogb = mean.loc["cardiogb", metric]
        best_neural = min(neural, key=lambda name: mean.loc[name, metric])
        baseline = mean.loc[best_neural, metric]
        direction = "lower" if cardiogb < baseline else "higher"
        conclusion = "outperformed" if cardiogb < baseline else "did not outperform"
        comparisons.append(
            f"For {label}, CardioGB was {number(cardiogb)} versus {number(baseline)} for {best_neural.replace('_', ' ')}, and therefore {conclusion} the strongest neural comparator on the mean ({direction} error)."
        )
    statistics = pd.read_csv(require(root / "formal_statistics_revised" / "e1_paired_tests.csv"))
    significant = statistics[statistics["p_adjust_bh_within_family"] < 0.05]
    return " ".join(comparisons) + f" Across the prespecified paired model-comparison family, {len(significant)} of {len(statistics)} contrasts remained significant after Benjamini–Hochberg correction."


def temporal_text(root: Path) -> str:
    e2 = pd.read_csv(require(root / "e2_interpolation_revised" / "tables" / "all_metrics.csv"))
    e3 = pd.read_csv(require(root / "e3_extrapolation_revised" / "tables" / "all_metrics.csv"))
    e2_summary = e2.groupby("case", observed=True)["sliced_wasserstein"].mean()
    parts = [f"{case:g} days, {number(value)}" for case, value in e2_summary.items()]
    finite = np.isfinite(e3[["mmd", "moment_error", "sliced_wasserstein"]].to_numpy()).all()
    maximum = e3["moment_error"].max()
    medians = e3.groupby("horizon_days", observed=True)["moment_error"].median()
    trend = "increased" if medians.corr(pd.Series(medians.index, index=medians.index), method="spearman") > 0 else "did not increase monotonically"
    return (
        "Mean E2 sliced-Wasserstein errors by held-out stage were " + "; ".join(parts) + ". "
        f"All revised E3 metrics were {'finite' if finite else 'not finite'}, with a maximum moment error of {number(maximum)}; median moment error {trend} with forecast horizon."
    )


def generalization_recovery_text(root: Path) -> str:
    e4 = pd.read_csv(require(root / "e4_group_cv_full" / "tables" / "all_metrics.csv"))
    fold = e4.groupby("case", observed=True)["sliced_wasserstein"].mean()
    e5 = pd.read_csv(require(root / "synthetic_recovery_full" / "tables" / "e5_confidence_intervals.csv"))
    e6 = pd.read_csv(require(root / "synthetic_recovery_full" / "tables" / "e6_confidence_intervals.csv"))
    corr5 = e5[e5["metric"] == "parameter_correlation"].sort_values("noise_std")
    corr6 = e6[e6["metric"] == "correlation"].sort_values("noise_std")
    return (
        f"Across the three E4 held-out biological-replicate folds, mean sliced-Wasserstein error ranged from {number(fold.min())} to {number(fold.max())}. "
        f"Synthetic parameter correlation changed from {number(corr5.iloc[0]['estimate'])} at noise {corr5.iloc[0]['noise_std']:g} to {number(corr5.iloc[-1]['estimate'])} at noise {corr5.iloc[-1]['noise_std']:g}. "
        f"Hidden-mechanism recovery correlation changed from {number(corr6.iloc[0]['estimate'])} to {number(corr6.iloc[-1]['estimate'])} over the same registered noise range."
    )


def interpretation_text(root: Path) -> str:
    base = root / "e7_full_interpretation"
    stage = pd.read_csv(require(base / "tables" / "mi_stage_bootstrap.csv"))
    diagnostics = pd.read_csv(require(base / "tables" / "parameter_identifiability_diagnostics.csv")).set_index("diagnostic")["value"]
    association = pd.read_csv(require(base / "tables" / "mi_pathway_associations.csv"))
    significant = association[association["p_adjust_bh"] < 0.05]
    peak = stage.loc[stage["mi_mean"].idxmax()]
    return (
        f"Mean mechanistic insufficiency peaked at {peak['stage_days']:g} days ({number(peak['mi_mean'])}, 95% CI {number(peak['ci_lower'])}–{number(peak['ci_upper'])}). "
        f"The cross-seed standardized parameter matrix had effective rank {int(diagnostics['effective_rank'])} among {int(diagnostics['parameters'])} parameters and maximum absolute pairwise correlation {number(diagnostics['maximum_absolute_pair_correlation'])}; these diagnostics indicate practical non-identifiability whenever the rank is deficient or correlations approach one. "
        f"{len(significant)} of {len(association)} pathway–insufficiency associations survived BH correction. Residual attributions are reported as model-derived, hypothesis-generating associations rather than causal effects."
    )


def external_human_text(root: Path) -> str:
    external = pd.read_csv(require(root / "external_predictive_validation_revised" / "metrics" / "external_prediction.csv"))
    mouse = external[external["protocol"].str.startswith("mouse")]
    wins = int((mouse["sliced_wasserstein"] < mouse["persistence_sliced_wasserstein"]).sum())
    delta = float((mouse["sliced_wasserstein"] - mouse["persistence_sliced_wasserstein"]).mean())
    uncertainty = pd.read_csv(require(root / "external_predictive_validation_revised" / "tables" / "uncertainty_inferential_tests.csv"))
    uncertainty_significant = int((uncertainty["p_adjust_bh"] < 0.05).sum())
    human = pd.read_csv(require(root / "human_snatac_validation_revised" / "tables" / "patient_group_tests.csv"))
    posthoc = pd.read_csv(require(root / "human_snatac_validation_revised" / "tables" / "patient_group_posthoc_effects.csv"))
    human_sig = int((human["p_adjust_bh"] < 0.05).sum())
    posthoc_sig = int((posthoc["p_adjust_bh_global"] < 0.05).sum())
    transfer = "provided evidence of predictive cross-species transfer" if wins == len(mouse) and delta < 0 else "did not establish predictive cross-species transfer"
    return (
        f"Frozen zebrafish-to-mouse CardioGB predictions had lower sliced-Wasserstein error than persistence for {wins} of {len(mouse)} evaluated mouse transitions (mean CardioGB-minus-persistence difference {number(delta)}), and therefore {transfer}. "
        f"{uncertainty_significant} of {len(uncertainty)} uncertainty–error/horizon tests survived BH correction. "
        f"In human-MI snATAC-seq, {human_sig} of {len(human)} patient-group omnibus pathway tests and {posthoc_sig} of {len(posthoc)} pairwise contrasts were significant after their stated BH corrections."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Results and Discussion from completed CardioGB tables")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--manuscript", type=Path, default=Path("manuscript/manuscript_sections.md"))
    args = parser.parse_args()
    root = args.results_root
    results = f'''{START}
## Results

### Revised grouped-holdout predictive benchmark

{benchmark_text(root)} Full seed- and transition-level estimates, confidence intervals, paired effect sizes, and multiplicity-corrected tests are reported in Figure 3 and Tables S3–S4.

### Temporal interpolation, extrapolation, and grouped generalization

{temporal_text(root)} {generalization_recovery_text(root)} These results are shown in Figures 4–5 and Tables S5–S9.

### Mechanistic insufficiency, parameter stability, and residual attribution

{interpretation_text(root)} Complete section-, biological-unit-, member-, and pathway-level outputs are provided in Figure 6 and Tables S11–S12.

### Ensemble uncertainty, mouse external prediction, and human-MI translation

{external_human_text(root)} Mouse results remain descriptive because the available series contains one spatial sample per stage. Human accessibility differences are translational associations and not a temporal forecasting validation. These analyses are summarized in Figures 7–8 and Tables S13–S16.

## Discussion

This study evaluated whether a bounded mechanistic scaffold and a spatial neural residual could yield a model that was simultaneously predictive, stable over clinically and biologically relevant horizons, and interpretable at the pathway-program level. The revised CardioGB implementation addresses the principal failure mode identified in the initial experiments: unconstrained long-horizon trajectories. Projection of pathway states, bounded state-specific residual scales, finer long-interval integration, trajectory-wide residual regularization, and direct optimization of Wasserstein and moment discrepancies make extrapolation numerically well posed. Mechanism-only warm starting further reduces competition between the interpretable and flexible components at initialization.

Predictive superiority is judged jointly rather than from a single favorable metric. The E1 paired results establish whether CardioGB improves over neural and persistence comparators under identical biological-unit splits; E2 and E3 determine whether any gain survives temporal displacement; and E4 tests whether it generalizes across registered biological replicates. Where the final tables do not support an advantage or a corrected test is non-significant, the manuscript reports that result directly. Increasing E1 from five to ten seeds was specified before examining revised-model outcomes to improve precision, not to guarantee statistical significance.

The mechanistic-insufficiency analysis gives the residual a specific scientific role: it identifies stages, pathway states, and spatial contexts in which the registered ODE is systematically incomplete. Nevertheless, integrated-gradient attribution and in-silico perturbation remain properties of the fitted model. They prioritize hypotheses for experimental perturbation but do not establish molecular causality. Likewise, cross-seed parameter stability is weaker than structural identifiability. A deficient effective rank or strong parameter correlations indicates that several effective coefficients can exchange roles while preserving predictions, and such coefficients should not be interpreted individually as biochemical rates.

Frozen mouse prediction is the strongest available test of external transportability because it prohibits retraining and mouse-informed weighting. Pathway conservation and predictive transfer are distinct: conserved temporal ordering can coexist with poor quantitative forecasts when species, injury paradigms, measurement technologies, or time scales differ. With one mouse specimen per stage, even favorable transition metrics would remain preliminary. The human-MI analysis addresses a different question—whether the six programs show patient-level chromatin-accessibility variation in human disease—and cannot validate zebrafish repair dynamics.

### Limitations

The primary atlas is cross-sectional, so transition learning is distributional and cannot recover individual cell or spot trajectories. The six-state representation improves interpretability but necessarily compresses cell-type composition, regulatory state, and spatial heterogeneity. Graph edges encode geometric neighbourhoods rather than measured ligand–receptor exchange. The mechanistic interaction graph is a reduced hypothesis and omits delays, changing cell abundance, tissue deformation, and unmeasured systemic signals. Parameter estimates are effective coefficients in normalized score space. External mouse inference is limited by one sample per stage, human analyses are not longitudinal forecasts, and conformal guarantees calibrated on zebrafish do not automatically transfer across species. Finally, all analyses are retrospective; prospective perturbation, denser temporal sampling, independent regenerative cohorts, and lineage-resolved measurements are required to test the model's mechanistic hypotheses.
{END}
'''
    text = args.manuscript.read_text(encoding="utf-8")
    if START in text and END in text:
        beginning = text.index(START)
        ending = text.index(END, beginning) + len(END)
        text = text[:beginning] + results + text[ending:]
    else:
        marker = "## Software description and reproducibility"
        if marker not in text:
            raise RuntimeError("software-description insertion marker missing")
        text = text.replace(marker, results + "\n\n" + marker, 1)
    args.manuscript.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "complete", "manuscript": str(args.manuscript)}, indent=2))


if __name__ == "__main__":
    main()
