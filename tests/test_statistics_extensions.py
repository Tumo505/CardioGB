import numpy as np

from cardiogb.statistics import benjamini_hochberg


def test_benjamini_hochberg_is_monotone_in_rank_and_order_preserving() -> None:
    p_values = np.asarray([0.04, 0.001, 0.03, 0.20])
    adjusted = benjamini_hochberg(p_values)
    assert adjusted.shape == p_values.shape
    assert np.all((0 <= adjusted) & (adjusted <= 1))
    ranked = adjusted[np.argsort(p_values)]
    assert np.all(np.diff(ranked) >= -1e-12)
    assert adjusted[1] == adjusted.min()
