from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_human_snatac import robust_unit_scale


def test_robust_unit_scale_is_bounded_monotone_and_finite() -> None:
    values = np.linspace(-3.0, 3.0, 1001)
    scaled = robust_unit_scale(values)
    assert scaled.dtype == np.float32
    assert np.isfinite(scaled).all()
    assert scaled.min() == 0.0
    assert scaled.max() == 1.0
    assert np.all(np.diff(scaled) >= 0.0)


def test_robust_unit_scale_handles_constant_features() -> None:
    assert np.array_equal(robust_unit_scale(np.ones(12)), np.zeros(12, dtype=np.float32))
