"""YAML configuration loading with deterministic recursive overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursively merged copy without mutating either input."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject ambiguous non-mapping roots."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a mapping at the root of {config_path}")
    return value


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load one or more YAML files, with later files overriding earlier ones."""
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, load_yaml(path))
    return merged


def resolve_project_path(value: str | Path, project_root: str | Path) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(project_root).expanduser().resolve() / path
    return path.resolve()

