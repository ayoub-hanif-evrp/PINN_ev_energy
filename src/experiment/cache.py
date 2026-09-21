"""Safe experiment caches. Retrain if identity is uncertain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_fingerprint(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def default_cache_payload(
    *,
    profile: str,
    method: str,
    seed: int,
    test_id: str,
    train_ids: list[str],
    val_ids: list[str],
    q_inner: float,
    q_outer: float,
    feature_list: list[str],
    progress_features: list[str],
    physics: dict[str, Any],
    windows: dict[str, Any],
    training: dict[str, Any],
    git_sha: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "profile": profile,
        "method": method,
        "seed": int(seed),
        "test_id": test_id,
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "q_inner": float(q_inner),
        "q_outer": float(q_outer),
        "feature_list": list(feature_list),
        "progress_features": list(progress_features),
        "physics": physics,
        "windows": windows,
        "training": {
            k: training.get(k)
            for k in (
                "optimizer",
                "learning_rate",
                "max_epochs",
                "early_stopping_patience",
                "weight_decay",
                "hidden_layers",
                "activation",
                "residual_limit_kw",
                "mlp_power_center_kw",
                "mlp_power_half_range_kw",
                "lambda_window",
                "lambda_dynamics",
                "lambda_state",
                "lambda_boundary",
                "lambda_prior",
                "huber_state_delta_pp",
            )
        },
        "git_sha": git_sha,
    }
    if extra:
        body.update(extra)
    return body


def cache_path(root: Path, fingerprint: str) -> Path:
    return root / "cache" / f"{fingerprint}.json"


def load_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
