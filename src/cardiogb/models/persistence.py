"""Persistence/null baseline."""

from __future__ import annotations

from torch import Tensor, nn


class PersistenceBaseline(nn.Module):
    def forward(self, states: Tensor, *_args: object, **_kwargs: object) -> Tensor:
        return states

