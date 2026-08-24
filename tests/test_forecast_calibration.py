import numpy as np
import pytest

from cardiogb.metrics.forecast_calibration import (
    calibrate_forecast,
    fit_displacement_scale,
    horizon_displacement_scale,
)


def test_validation_scale_recovers_known_shrinkage():
    source = np.array([[0.2, 0.4], [0.5, 0.3]])
    predicted = np.array([[0.8, 0.2], [0.9, 0.7]])
    observed = source + 0.25 * (predicted - source)
    assert fit_displacement_scale(source, predicted, observed) == pytest.approx(0.25)


def test_horizon_calibration_is_bounded_and_shrinks_extrapolation():
    scale = horizon_displacement_scale(8.0, 2.0, 0.8)
    assert scale == pytest.approx(0.2)
    source = np.array([[0.2, 0.8]])
    raw = np.array([[1.2, -0.2]])
    calibrated = calibrate_forecast(source, raw, scale)
    np.testing.assert_allclose(calibrated, [[0.4, 0.6]])


def test_scale_rejects_invalid_horizons():
    with pytest.raises(ValueError):
        horizon_displacement_scale(0.0, 1.0, 0.5)
