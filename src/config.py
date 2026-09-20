"""YAML configuration loading with optional `extends:` inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from paths import project_root, resolve_under_root


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping.")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving relative `extends:` against the project root."""
    root = project_root()
    cfg_path = resolve_under_root(path, root)
    raw = load_yaml(cfg_path)
    if "extends" in raw:
        parent = load_config(raw["extends"])
        merged = _deep_merge(parent, raw)
    else:
        merged = raw
    merged["_config_path"] = str(cfg_path)
    return merged
