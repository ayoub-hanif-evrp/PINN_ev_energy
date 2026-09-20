"""Explicit SI unit conversions and trapezoidal energy integration.

n timestamps define n-1 time intervals:

    Δt_i = t_{i+1} - t_i,   i = 0, …, n-2

Power is sampled at the timestamps. Interval energy uses the trapezoid:

    P_{i+1/2} = (P_i + P_{i+1}) / 2
    ΔE_i = P_{i+1/2} * Δt_i / 3600

Prefix energy has length n with prefix[0] = 0, and

    E(a, b) = prefix[b] - prefix[a] = energy from sample a to sample b.
"""

from __future__ import annotations

import numpy as np

KMH_TO_MPS = 1.0 / 3.6
MPS_TO_KMH = 3.6
SECONDS_PER_HOUR = 3600.0
WATTS_PER_KW = 1000.0
EARTH_RADIUS_M = 6_371_000.0
STANDARD_GRAVITY = 9.80665


def kmh_to_mps(speed_kmh: np.ndarray | float) -> np.ndarray:
    return np.asarray(speed_kmh, dtype=float) * KMH_TO_MPS


def mps_to_kmh(speed_mps: np.ndarray | float) -> np.ndarray:
    return np.asarray(speed_mps, dtype=float) * MPS_TO_KMH


def seconds_to_hours(dt_s: np.ndarray | float) -> np.ndarray:
    return np.asarray(dt_s, dtype=float) / SECONDS_PER_HOUR


def watts_to_kw(power_w: np.ndarray | float) -> np.ndarray:
    return np.asarray(power_w, dtype=float) / WATTS_PER_KW


def kw_to_watts(power_kw: np.ndarray | float) -> np.ndarray:
    return np.asarray(power_kw, dtype=float) * WATTS_PER_KW


def interval_increments_kwh(power_kw: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    """Trapezoidal ΔE on each interval i → i+1. Length n; last increment is 0."""
    p = np.asarray(power_kw, dtype=float)
    dt = np.asarray(dt_s, dtype=float)
    if p.shape != dt.shape:
        raise ValueError("power_kw and dt_s must have the same shape (length n).")
    de = np.zeros_like(p)
    if p.size >= 2:
        de[:-1] = 0.5 * (p[:-1] + p[1:]) * dt[:-1] / SECONDS_PER_HOUR
    return de


def energy_kwh_from_power(power_kw: np.ndarray, dt_s: np.ndarray) -> float:
    """Integrate battery power (kW) over n-1 intervals to kWh."""
    return float(np.sum(interval_increments_kwh(power_kw, dt_s)))


def energy_prefix_kwh(power_kw: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    """
    Length-n prefix with prefix[0] = 0.

    prefix[t] = energy from sample 0 to sample t, so
    E(a, b) = prefix[b] - prefix[a].
    """
    de = interval_increments_kwh(power_kw, dt_s)
    prefix = np.zeros(de.size, dtype=float)
    if de.size >= 2:
        prefix[1:] = np.cumsum(de[:-1])
    return prefix


def cumulative_energy_kwh(power_kw: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    """Alias for energy_prefix_kwh (length n, prefix[0] = 0)."""
    return energy_prefix_kwh(power_kw, dt_s)


def window_energy_from_prefix(prefix_kwh: np.ndarray, start: int, end: int) -> float:
    """Energy from sample `start` to sample `end`: prefix[end] - prefix[start]."""
    if end < start:
        raise ValueError("end must be >= start")
    n = len(prefix_kwh)
    if start < 0 or end >= n:
        raise IndexError("window indices out of range for prefix array")
    return float(prefix_kwh[end] - prefix_kwh[start])
