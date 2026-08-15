"""Training callbacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 25
    min_delta: float = 0.0
    best: float = float("inf")
    bad_epochs: int = 0

    def update(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
