import numpy as np

from cardiogb.training.robust_ensemble import aggregate_members, select_aggregation


def test_median_and_trimmed_mean_reject_extreme_member():
    values = np.array([[0.2], [0.2], [0.2], [0.2], [1.0]])
    np.testing.assert_allclose(aggregate_members(values, "median"), [0.2])
    np.testing.assert_allclose(aggregate_members(values, "trimmed_mean"), [0.2])
    assert aggregate_members(values, "equal_mean")[0] > 0.2


def test_validation_group_selection_prefers_robust_aggregation_for_outlier():
    observed = np.array([0.2, 0.3, 0.4, 0.5])
    design = np.column_stack(
        [observed, observed, observed, observed, observed + 0.8]
    )
    groups = np.array(["a", "a", "b", "b"])
    selected, scores = select_aggregation(design, observed, groups)
    assert selected in {"median", "trimmed_mean"}
    assert scores[selected] < scores["equal_mean"]
