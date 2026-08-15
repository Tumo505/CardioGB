"""Parameter recovery metrics."""

from __future__ import annotations

import numpy as np


def parameter_recovery(true: dict[str, float], inferred: dict[str, float]) -> dict[str, float]:
    names = sorted(set(true) & set(inferred))
    if not names:
        raise ValueError("true and inferred parameter sets do not overlap")
    x = np.array([true[name] for name in names])
    y = np.array([inferred[name] for name in names])
    correlation = 0.0 if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])
    return {
        "parameter_rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "parameter_mae": float(np.mean(np.abs(x - y))),
        "parameter_correlation": correlation,
    }
