"""Biological-unit perturbation statistics for newly added validation cohorts."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from cardiogb.statistics import benjamini_hochberg
from cardiogb.utils.io import export_table


STATES = ("I", "A", "F", "C", "V", "M")


def exact_difference_test(first: np.ndarray, second: np.ndarray) -> float:
    pooled = np.concatenate((first, second))
    observed = abs(float(first.mean() - second.mean()))
    assignments = itertools.combinations(range(len(pooled)), len(first))
    statistics = []
    for indices in assignments:
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(indices)] = True
        statistics.append(abs(float(pooled[mask].mean() - pooled[~mask].mean())))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-12))


def hedges_g(first: np.ndarray, second: np.ndarray) -> float:
    degrees = len(first) + len(second) - 2
    if degrees <= 0:
        return float("nan")
    variance = ((len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1)) / degrees
    if not np.isfinite(variance) or variance <= 0:
        return float("nan")
    correction = 1.0 - 3.0 / (4.0 * (len(first) + len(second)) - 9.0)
    return float(correction * (second.mean() - first.mean()) / np.sqrt(variance))


def compare(
    frame: pd.DataFrame,
    *,
    unit_column: str,
    group_column: str,
    reference: str,
    perturbed: str,
    cohort: str,
) -> pd.DataFrame:
    rows = []
    for state in STATES:
        column = f"{state}_mean"
        reference_values = frame.loc[frame[group_column] == reference, column].to_numpy(float)
        perturbed_values = frame.loc[frame[group_column] == perturbed, column].to_numpy(float)
        inferential = len(reference_values) >= 2 and len(perturbed_values) >= 2
        rows.append(
            {
                "cohort": cohort,
                "contrast": f"{perturbed}_minus_{reference}",
                "state": state,
                "unit": unit_column,
                "n_reference": len(reference_values),
                "n_perturbed": len(perturbed_values),
                "reference_mean": float(reference_values.mean()),
                "perturbed_mean": float(perturbed_values.mean()),
                "mean_difference": float(perturbed_values.mean() - reference_values.mean()),
                "hedges_g": hedges_g(reference_values, perturbed_values) if inferential else float("nan"),
                "p_value": exact_difference_test(reference_values, perturbed_values) if inferential else float("nan"),
                "inferential_status": "exact biological-unit permutation" if inferential else "descriptive only; fewer than two units per group",
            }
        )
    result = pd.DataFrame(rows)
    valid = result["p_value"].notna()
    result.loc[valid, "fdr_bh"] = benjamini_hochberg(result.loc[valid, "p_value"].to_numpy())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/added_validation"))
    parser.add_argument("--output", type=Path, default=Path("results/added_validation/perturbation_statistics.csv"))
    args = parser.parse_args()

    prrx = pd.read_csv(args.input_dir / "gse153170_biological_unit_states.csv")
    prrx["condition"] = prrx["genotype"]
    complement = pd.read_csv(args.input_dir / "gse217828_snrna_biological_unit_states.csv")
    complement["condition"] = complement["group1"].str.extract(r"^(C3KO|WT)", expand=False)
    yap = pd.read_csv(args.input_dir / "gse217828_scrna_biological_unit_states.csv")
    yap["condition"] = yap["sample"]

    results = pd.concat(
        [
            compare(prrx, unit_column="CEL-seq2 library", group_column="condition", reference="WT", perturbed="prrx1b-/-", cohort="GSE153170"),
            compare(complement, unit_column="mouse replicate", group_column="condition", reference="WT", perturbed="C3KO", cohort="GSE217828_snRNA"),
            compare(yap, unit_column="pooled sample", group_column="condition", reference="WT", perturbed="YAP5SA", cohort="GSE217828_scRNA"),
        ],
        ignore_index=True,
    )
    export_table(results, args.output)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
