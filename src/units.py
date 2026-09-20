"""Explicit SI unit conversions and energy integration helpers."""

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


def energy_kwh_from_power(power_kw: np.ndarray, dt_s: np.ndarray) -> float:
    """Integrate battery power (kW) over dt (seconds) to kWh (all samples)."""
    p = np.asarray(power_kw, dtype=float)
    dt = np.asarray(dt_s, dtype=float)
    if p.shape != dt.shape:
        raise ValueError("power_kw and dt_s must have the same shape.")
    return float(np.sum(p * dt / SECONDS_PER_HOUR))


def energy_prefix_kwh(power_kw: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    """
    Prefix energy array of length n+1 with prefix[0] = 0.

    prefix[t] = sum_{k=0}^{t-1} P[k] * dt[k] / 3600

    Inclusive window [a, b] (sample indices) therefore has
    E = prefix[b + 1] - prefix[a]
    which equals sum(P[a:b+1] * dt[a:b+1]) / 3600.

    The plan's E[a:b] = cum[b] - cum[a] is this prefix form: use
    cum = prefix and interpret the right index as exclusive in prefix space
    (i.e. sample b inclusive → prefix index b+1).
    """
    p = np.asarray(power_kw, dtype=float)
    dt = np.asarray(dt_s, dtype=float)
    if p.shape != dt.shape:
        raise ValueError("power_kw and dt_s must have the same shape.")
    de = p * dt / SECONDS_PER_HOUR
    return np.concatenate([[0.0], np.cumsum(de)])


def cumulative_energy_kwh(power_kw: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    """Length-n cumulative energy with cum[0] = 0, cum[t] = energy before sample t."""
    prefix = energy_prefix_kwh(power_kw, dt_s)
    return prefix[:-1]


def window_energy_from_prefix(prefix_kwh: np.ndarray, start: int, end: int) -> float:
    """Inclusive sample window [start, end] using a length n+1 prefix array."""
    if end < start:
        raise ValueError("end must be >= start")
    if start < 0 or end + 1 >= len(prefix_kwh):
        raise IndexError("window indices out of range for prefix array")
    return float(prefix_kwh[end + 1] - prefix_kwh[start])
