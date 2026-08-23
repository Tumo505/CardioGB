import numpy as np

from cardiogb.training.ensemble_weighting import fit_simplex_weights, weighted_summary


def test_validation_simplex_weighting_prefers_accurate_member() -> None:
    observed = np.array([0.1, 0.3, 0.8, 0.5])
    design = np.column_stack((observed, observed + 0.5, 1.0 - observed))
    weights = fit_simplex_weights(design, observed, ridge=1e-6)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)
    assert weights[0] > 0.9
    stacked = np.stack((observed, observed + 0.5, 1.0 - observed))[:, :, None]
    mean, std = weighted_summary(stacked, weights)
    assert mean.shape == std.shape == (4, 1)
    assert np.isfinite(std).all()
