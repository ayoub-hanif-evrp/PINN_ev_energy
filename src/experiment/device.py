"""Device and seed helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("experiment", {}).get("device", "auto")).lower()
    if requested in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
