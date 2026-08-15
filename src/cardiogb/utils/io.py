"""Atomic, manuscript-friendly result exports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def atomic_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def export_table(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".csv":
        frame.to_csv(target, index=False)
    elif target.suffix.lower() == ".parquet":
        frame.to_parquet(target, index=False)
    else:
        raise ValueError("table output must end in .csv or .parquet")
