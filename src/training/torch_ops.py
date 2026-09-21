"""Differentiable trapezoidal energy integration (matches src/units.py)."""

from __future__ import annotations

import torch

SECONDS_PER_HOUR = 3600.0


def torch_interval_increments_kwh(power_kw: torch.Tensor, dt_s: torch.Tensor) -> torch.Tensor:
    if power_kw.shape != dt_s.shape:
        raise ValueError("power_kw and dt_s must have the same shape.")
    de = torch.zeros_like(power_kw)
    if power_kw.numel() >= 2:
        de = de.clone()
        de[:-1] = 0.5 * (power_kw[:-1] + power_kw[1:]) * dt_s[:-1] / SECONDS_PER_HOUR
    return de


def torch_energy_prefix(power_kw: torch.Tensor, dt_s: torch.Tensor) -> torch.Tensor:
    """Length-n prefix with prefix[0] = 0. E(a,b) = prefix[b] - prefix[a]."""
    de = torch_interval_increments_kwh(power_kw, dt_s)
    prefix = torch.zeros_like(power_kw)
    if power_kw.numel() >= 2:
        prefix = prefix.clone()
        prefix[1:] = torch.cumsum(de[:-1], dim=0)
    return prefix


def torch_window_energy(prefix_kwh: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    return prefix_kwh[end] - prefix_kwh[start]
