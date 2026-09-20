"""Quantization-aware weak energy windows.

Observed energy uses SoC endpoints only:
    E_obs(w) = E_battery * (SOC[a] - SOC[b]) / 100

Predicted energy uses a prefix sum of latent power, never a fake
instantaneous P_t = E_battery * (SOC_t - SOC_{t+1}).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from data.preprocessing import ProcessedTrip
from physics.vehicle_model import observed_energy_kwh
from units import energy_prefix_kwh, window_energy_from_prefix


@dataclass
class EnergyWindow:
    trip_id: str
    start: int
    end: int
    scale: str
    e_obs_kwh: float
    dsoc: float
    duration_s: float
    distance_m: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def huber_delta_kwh(
    battery_capacity_kwh: float,
    q: float,
    mode: str = "quantization",
    fixed: float | None = None,
) -> float:
    """Label-precision scale. Not a claim that this equals the BMS process."""
    if mode == "fixed" and fixed is not None:
        return float(fixed)
    if not np.isfinite(q) or q <= 0:
        return float(battery_capacity_kwh) * 0.02 / 100.0
    return float(battery_capacity_kwh) * float(q) / 100.0


def mean_loss_per_scale(scale_to_losses: dict[str, Iterable[float]]) -> float:
    """Average of per-scale means so short windows cannot dominate by count."""
    means: list[float] = []
    for values in scale_to_losses.values():
        arr = np.asarray(list(values), dtype=float)
        if arr.size:
            means.append(float(np.mean(arr)))
    if not means:
        return 0.0
    return float(np.mean(means))


def _soc_change_indices(soc: np.ndarray, q: float) -> np.ndarray:
    if len(soc) == 0:
        return np.array([], dtype=int)
    step = np.abs(np.diff(soc, prepend=soc[0]))
    changed = np.where(step >= max(0.25 * q, 1e-9))[0]
    if 0 not in changed:
        changed = np.concatenate([[0], changed])
    return np.unique(changed)


def _end_index_for_duration(dt: np.ndarray, start: int, duration_s: float) -> int | None:
    n = len(dt)
    acc = 0.0
    for j in range(start + 1, n):
        acc += float(dt[j])
        if acc >= duration_s:
            return j
    return None


def generate_trip_windows(
    trip: ProcessedTrip,
    battery_capacity_kwh: float,
    q: float,
    config: dict[str, Any] | None = None,
    overlapping: bool = True,
) -> list[EnergyWindow]:
    cfg = (config or {}).get("windows", {})
    df = trip.frame
    if "soc" not in df.columns:
        return []
    soc = pd.to_numeric(df["soc"], errors="coerce").to_numpy(dtype=float)
    dt = df["dt_s"].to_numpy(dtype=float)
    s = df["s_m"].to_numpy(dtype=float) if "s_m" in df.columns else np.zeros(len(df))
    n = len(df)
    if n < 2 or not np.isfinite(q) or q <= 0:
        return []

    windows: list[EnergyWindow] = []
    k_list = list(cfg.get("soc_event_k", [1, 2, 3]))
    time_list = list(cfg.get("fixed_time_s", [120, 300, 600]))
    min_abs_dsoc_fixed = float(cfg.get("min_abs_dsoc_for_fixed", 1.0)) * q
    event_stride = int(cfg.get("event_start_stride", 30))

    def add_window(start: int, end: int, scale: str) -> None:
        if end <= start or end >= n or start < 0:
            return
        dsoc = float(soc[start] - soc[end])
        e_obs = observed_energy_kwh(soc[start], soc[end], battery_capacity_kwh)
        duration = float(np.nansum(dt[start + 1 : end + 1]))
        distance = float(s[end] - s[start])
        windows.append(
            EnergyWindow(
                trip_id=trip.trip_id,
                start=int(start),
                end=int(end),
                scale=scale,
                e_obs_kwh=float(e_obs),
                dsoc=dsoc,
                duration_s=duration,
                distance_m=distance,
            )
        )

    # A. SoC-event windows
    starts = _soc_change_indices(soc, q)
    if event_stride > 0:
        stride_starts = np.arange(0, n - 1, event_stride)
        starts = np.unique(np.concatenate([starts, stride_starts]))
    for k in k_list:
        thresh = float(k) * q
        last_end = -1
        for a in starts:
            a = int(a)
            if not overlapping and a < last_end:
                continue
            target = None
            for b in range(a + 1, n):
                if abs(soc[a] - soc[b]) >= thresh:
                    target = b
                    break
            if target is None:
                continue
            add_window(a, target, f"soc_event_k{k}")
            last_end = target

    # B. Fixed-time windows (overlapping uses half-duration stride so they do not explode)
    for duration in time_list:
        last_end = -1
        start = 0
        stride = 1
        if overlapping:
            # Advance by ~half the window in samples using mean dt.
            mean_dt = float(np.nanmean(dt[dt > 0])) if np.any(dt > 0) else 1.0
            stride = max(1, int(round(float(duration) / (2.0 * mean_dt))))
        while start < n - 1:
            if not overlapping and start < last_end:
                start = last_end
                continue
            end = _end_index_for_duration(dt, start, float(duration))
            if end is None:
                break
            if abs(soc[start] - soc[end]) >= min_abs_dsoc_fixed:
                add_window(start, end, f"time_{int(duration)}s")
                last_end = end
            if overlapping:
                start += stride
            else:
                start = end

    # C. Full-trip
    if cfg.get("include_full_trip", True):
        add_window(0, n - 1, "full_trip")

    return windows


def generate_windows(
    trips: list[ProcessedTrip],
    battery_capacity_kwh: float,
    q: float,
    config: dict[str, Any] | None = None,
    overlapping: bool = True,
) -> list[EnergyWindow]:
    out: list[EnergyWindow] = []
    for trip in trips:
        out.extend(
            generate_trip_windows(trip, battery_capacity_kwh, q, config=config, overlapping=overlapping)
        )
    return out


def windows_to_frame(windows: list[EnergyWindow]) -> pd.DataFrame:
    if not windows:
        return pd.DataFrame(
            columns=["trip_id", "start", "end", "scale", "e_obs_kwh", "dsoc", "duration_s", "distance_m"]
        )
    return pd.DataFrame([w.as_dict() for w in windows])


def predicted_window_energy(power_kw: np.ndarray, dt_s: np.ndarray, start: int, end: int) -> float:
    prefix = energy_prefix_kwh(power_kw, dt_s)
    return window_energy_from_prefix(prefix, start, end)


def assert_windows_within_trip(windows: list[EnergyWindow], trip_lengths: dict[str, int]) -> None:
    for w in windows:
        n = trip_lengths[w.trip_id]
        if w.start < 0 or w.end >= n or w.end < w.start:
            raise AssertionError(f"Window {w} escapes trip {w.trip_id} of length {n}")
