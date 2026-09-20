"""SI preprocessing: timestamps, speed, acceleration, distance, grade.

Never writes to original CSVs. Smoothing never crosses trip boundaries because
each trip is processed independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from data.loader import TripRecord
from data.schema import MAIN_MODEL_FEATURES
from paths import project_root, resolve_under_root
from units import EARTH_RADIUS_M, KMH_TO_MPS


def haversine_m(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
    radius_m: float = EARTH_RADIUS_M,
) -> np.ndarray:
    """Great-circle distance in metres."""
    phi1 = np.deg2rad(np.asarray(lat1, dtype=float))
    phi2 = np.deg2rad(np.asarray(lat2, dtype=float))
    dphi = phi2 - phi1
    dlmb = np.deg2rad(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * radius_m * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _odd_window(requested: int, n: int) -> int:
    if n < 3:
        return 0
    w = min(int(requested), n if n % 2 == 1 else n - 1)
    if w < 3:
        return 0
    if w % 2 == 0:
        w -= 1
    return w


def smooth_series(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    n = len(x)
    w = _odd_window(window, n)
    if w == 0:
        return x.copy()
    order = min(int(polyorder), w - 1)
    finite = np.isfinite(x)
    if not finite.all():
        filled = pd.Series(x).interpolate(limit_direction="both").to_numpy()
        x = np.asarray(filled, dtype=float)
    return savgol_filter(x, window_length=w, polyorder=order, mode="interp")


def _clean_date_strings(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    # Pandas may stringify a parsed date as "2021-03-29 00:00:00".
    return s.str.replace(r"\s+\d{1,2}:\d{2}:\d{2}.*$", "", regex=True)


def _clean_time_strings(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^0 days\s+", "", regex=True)
    s = s.str.replace(r"^\d{4}-\d{2}-\d{2}\s+", "", regex=True)
    return s


def epoch_seconds(timestamps) -> np.ndarray:
    """UTC seconds. Unit-safe under pandas datetime64[ns] or datetime64[us]."""
    s = pd.to_datetime(pd.Series(np.asarray(timestamps)), errors="coerce")
    delta = s - pd.Timestamp("1970-01-01")
    return delta.dt.total_seconds().to_numpy(dtype=float)


def parse_trip_timestamps(frame: pd.DataFrame, fallback_dt_s: float = 1.0) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    n = len(frame)
    if n == 0:
        return np.array([], dtype="datetime64[ns]"), notes

    if "date" in frame.columns and "time" in frame.columns:
        date_s = _clean_date_strings(frame["date"])
        time_s = _clean_time_strings(frame["time"])
        combined = date_s + " " + time_s
        ts = pd.Series(pd.NaT, index=frame.index)
        for fmt in (
            "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S.%f",
            "%d/%m/%Y %H:%M:%S",
        ):
            still = ts.isna()
            if not still.any():
                break
            parsed = pd.to_datetime(combined[still], errors="coerce", format=fmt)
            ts.loc[still] = parsed
        n_nat = int(ts.isna().sum())
        if n_nat:
            notes.append(f"timestamp_parse_failures={n_nat}")
    else:
        ts = pd.Series(pd.NaT, index=frame.index)
        notes.append("missing_date_or_time_columns")

    if ts.isna().any() and ts.notna().mean() >= 0.8:
        ts = ts.ffill().bfill()
        notes.append("timestamps_filled_from_neighbors")

    if ts.isna().any():
        start = ts.dropna().iloc[0] if ts.notna().any() else pd.Timestamp("1970-01-01")
        idx = np.arange(n)
        ts = pd.Series(start + pd.to_timedelta(idx, unit="s"))
        notes.append("timestamps_synthesized_from_row_index_at_1Hz")

    return ts.to_numpy(), notes


def compute_dt_seconds(timestamps: np.ndarray, fallback_dt_s: float = 1.0) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(timestamps)
    dt = np.full(n, float(fallback_dt_s), dtype=float)
    stats = {
        "n_nonpositive_dt": 0,
        "n_duplicate_timestamps": 0,
        "n_gaps_gt_2s": 0,
        "fallback_dt_s": float(fallback_dt_s),
    }
    if n == 0:
        return dt, stats
    epoch = epoch_seconds(timestamps)
    if n >= 2:
        diffs = np.diff(epoch)
        dt[1:] = diffs
        dt[0] = diffs[0] if np.isfinite(diffs[0]) and diffs[0] > 0 else float(fallback_dt_s)
        stats["n_nonpositive_dt"] = int(np.sum(~(dt > 0)))
        stats["n_duplicate_timestamps"] = int(np.sum(np.diff(epoch) == 0))
        stats["n_gaps_gt_2s"] = int(np.sum(np.diff(epoch) > 2.0))
        bad = ~(dt > 0) | ~np.isfinite(dt)
        dt[bad] = float(fallback_dt_s)
        if bad.any():
            stats["repaired_nonpositive_or_nan_dt"] = int(bad.sum())
    return dt, stats


def cumulative_distance_m(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(lat)
    ds = np.zeros(n, dtype=float)
    if n >= 2:
        ds[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        ds = np.where(np.isfinite(ds), ds, 0.0)
    s = np.cumsum(ds)
    return ds, s


def estimate_grade(
    altitude_m: np.ndarray,
    s_m: np.ndarray,
    spatial_window_m: float,
    ds_min_m: float,
    grade_clip: float,
    alt_savgol_window: int,
    alt_savgol_polyorder: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Robust GPS grade: smooth altitude, then dh/ds over a spatial interval.

    theta = atan(dh/ds). Never divide by near-zero distance.
    """
    n = len(altitude_m)
    alt_s = smooth_series(altitude_m, alt_savgol_window, alt_savgol_polyorder)
    grade = np.zeros(n, dtype=float)
    n_small_ds = 0
    n_invalid = 0
    last = 0.0
    s = np.asarray(s_m, dtype=float)
    for i in range(n):
        target = s[i] + float(spatial_window_m)
        j = int(np.searchsorted(s, target, side="left"))
        if j <= i:
            j = min(n - 1, i + 1)
        if j >= n:
            j = n - 1
        ds = s[j] - s[i]
        if ds < float(ds_min_m):
            # Look backward if we cannot look forward far enough.
            target_b = s[i] - float(spatial_window_m)
            k = int(np.searchsorted(s, target_b, side="left"))
            k = max(0, min(k, i))
            ds = s[i] - s[k]
            if ds < float(ds_min_m):
                grade[i] = last
                n_small_ds += 1
                continue
            dh = alt_s[i] - alt_s[k]
        else:
            dh = alt_s[j] - alt_s[i]
        if not np.isfinite(dh) or not np.isfinite(ds) or ds <= 0:
            grade[i] = last
            n_invalid += 1
            continue
        g = dh / ds
        last = g
        grade[i] = g

    n_clip = int(np.sum(np.abs(grade) > float(grade_clip)))
    grade_clipped = np.clip(grade, -float(grade_clip), float(grade_clip))
    theta = np.arctan(grade_clipped)
    stats = {
        "n_small_ds": n_small_ds,
        "n_invalid": n_invalid,
        "n_clipped": n_clip,
        "clip_fraction": float(n_clip / n) if n else 0.0,
        "grade_clip": float(grade_clip),
        "spatial_window_m": float(spatial_window_m),
        "ds_min_m": float(ds_min_m),
        "grade_raw_min": float(np.nanmin(grade)) if n else np.nan,
        "grade_raw_max": float(np.nanmax(grade)) if n else np.nan,
    }
    return theta, grade_clipped, stats


def acceleration_from_speed(
    speed_mps: np.ndarray,
    timestamps: np.ndarray,
    dt_s: np.ndarray,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[np.ndarray, np.ndarray]:
    v_smooth = smooth_series(speed_mps, savgol_window, savgol_polyorder)
    epoch = epoch_seconds(timestamps)
    if len(epoch) >= 2 and np.all(np.diff(epoch) > 0):
        acc = np.gradient(v_smooth, epoch)
    else:
        t = np.cumsum(dt_s)
        t = t - t[0]
        # Avoid zero-length time axis.
        if len(t) >= 2 and t[-1] > t[0]:
            acc = np.gradient(v_smooth, t)
        else:
            acc = np.gradient(v_smooth)
    acc = np.asarray(acc, dtype=float)
    acc[~np.isfinite(acc)] = 0.0
    return v_smooth, acc


@dataclass
class ProcessedTrip:
    trip_id: str
    trajectory: str
    source_path: Path
    kind: str
    frame: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))


def preprocess_trip(trip: TripRecord, config: dict[str, Any]) -> ProcessedTrip:
    pre = config.get("preprocessing", {})
    fallback_dt = float(pre.get("timestamp_fallback_dt_s", 1.0))
    frame = trip.frame.copy()
    notes = list(trip.load_notes)

    timestamps, ts_notes = parse_trip_timestamps(frame, fallback_dt)
    notes.extend(ts_notes)
    dt, dt_stats = compute_dt_seconds(timestamps, fallback_dt)
    frame["timestamp"] = pd.to_datetime(timestamps)
    frame["dt_s"] = dt

    if "speed_kmh" not in frame.columns:
        raise ValueError(f"{trip.trip_id}: missing Speed column after canonicalisation")
    speed_kmh = pd.to_numeric(frame["speed_kmh"], errors="coerce").to_numpy(dtype=float)
    speed_kmh = np.where(np.isfinite(speed_kmh), speed_kmh, 0.0)
    frame["speed_kmh"] = speed_kmh
    frame["speed_mps"] = speed_kmh * KMH_TO_MPS

    if "gps_speed_kmh" in frame.columns:
        gps = pd.to_numeric(frame["gps_speed_kmh"], errors="coerce").to_numpy(dtype=float)
        frame["gps_speed_kmh"] = gps
        frame["gps_speed_mps"] = gps * KMH_TO_MPS
    else:
        frame["gps_speed_mps"] = np.nan

    v_smooth, acc = acceleration_from_speed(
        frame["speed_mps"].to_numpy(dtype=float),
        frame["timestamp"].to_numpy(),
        dt,
        int(pre.get("savgol_window", 11)),
        int(pre.get("savgol_polyorder", 2)),
    )
    frame["speed_mps_smooth"] = v_smooth
    frame["acc_mps2"] = acc

    lat = pd.to_numeric(frame.get("lat", np.nan), errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(frame.get("lon", np.nan), errors="coerce").to_numpy(dtype=float)
    alt = pd.to_numeric(frame.get("alt_m", np.nan), errors="coerce").to_numpy(dtype=float)
    lat = pd.Series(lat).ffill().bfill().to_numpy(dtype=float)
    lon = pd.Series(lon).ffill().bfill().to_numpy(dtype=float)
    alt = pd.Series(alt).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    ds, s = cumulative_distance_m(lat, lon)
    # Zero only continent-scale teleports. Analysed GPS often holds a fix for
    # several seconds then jumps 50–150 m; those steps are real distance.
    ds_max = np.maximum(80.0 * dt, 2000.0)
    jump = (ds > ds_max) | ~np.isfinite(ds)
    n_jumps = int(np.sum(jump))
    if n_jumps:
        ds = ds.copy()
        ds[jump] = 0.0
        s = np.cumsum(ds)
        notes.append(f"implausible_gps_jumps_zeroed={n_jumps}")
    frame["ds_m"] = ds
    frame["s_m"] = s
    frame["distance_km"] = s / 1000.0
    frame["gps_jump_flag"] = jump

    theta, grade, grade_stats = estimate_grade(
        alt,
        s,
        spatial_window_m=float(pre.get("grade_spatial_window_m", 40.0)),
        ds_min_m=float(pre.get("grade_ds_min_m", 5.0)),
        grade_clip=float(pre.get("grade_clip", 0.20)),
        alt_savgol_window=int(pre.get("altitude_savgol_window", 21)),
        alt_savgol_polyorder=int(pre.get("altitude_savgol_polyorder", 2)),
    )
    frame["alt_m"] = alt
    frame["grade"] = grade
    frame["theta_rad"] = theta
    frame["theta_deg"] = np.rad2deg(theta)

    if "speed_limit_kmh" in frame.columns:
        sl = pd.to_numeric(frame["speed_limit_kmh"], errors="coerce").to_numpy(dtype=float)
        frame["speed_limit_kmh"] = sl
        frame["speed_limit_mps"] = sl * KMH_TO_MPS
    else:
        frame["speed_limit_mps"] = np.nan

    if "wind_speed_mps" in frame.columns:
        frame["wind_speed_mps"] = pd.to_numeric(frame["wind_speed_mps"], errors="coerce")
    if "temperature_c" in frame.columns:
        frame["temperature_c"] = pd.to_numeric(frame["temperature_c"], errors="coerce")
    if "humidity_pct" in frame.columns:
        frame["humidity_pct"] = pd.to_numeric(frame["humidity_pct"], errors="coerce")
    if "traffic" in frame.columns:
        frame["traffic"] = pd.to_numeric(frame["traffic"], errors="coerce")
    if "soc" in frame.columns:
        frame["soc"] = pd.to_numeric(frame["soc"], errors="coerce")

    stats = {
        "dt": dt_stats,
        "grade": grade_stats,
        "distance_m": float(s[-1]) if len(s) else 0.0,
        "duration_s": float(np.nansum(dt)),
        "n_rows": int(len(frame)),
        "main_features": list(MAIN_MODEL_FEATURES),
        "savgol_window": int(pre.get("savgol_window", 11)),
        "savgol_polyorder": int(pre.get("savgol_polyorder", 2)),
        "wind_used_as_headwind": False,
        "n_implausible_gps_jumps": n_jumps,
    }
    return ProcessedTrip(
        trip_id=trip.trip_id,
        trajectory=trip.trajectory,
        source_path=trip.source_path,
        kind=trip.kind,
        frame=frame,
        notes=notes,
        stats=stats,
    )


def preprocess_trips(trips: list[TripRecord], config: dict[str, Any]) -> list[ProcessedTrip]:
    return [preprocess_trip(t, config) for t in trips]


def _config_fingerprint(config: dict[str, Any]) -> str:
    payload = {
        "preprocessing": config.get("preprocessing", {}),
        "vehicle": config.get("vehicle", {}),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return sha256(blob).hexdigest()[:16]


def cache_processed_trip(trip: ProcessedTrip, config: dict[str, Any]) -> Path:
    root = project_root()
    cache_dir = resolve_under_root(config.get("paths", {}).get("cache_dir", "outputs/cache"), root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = _config_fingerprint(config)
    parquet_path = cache_dir / f"{trip.trip_id}__{fp}.parquet"
    sidecar = cache_dir / f"{trip.trip_id}__{fp}.json"
    trip.frame.to_parquet(parquet_path, index=False)
    sidecar.write_text(
        json.dumps(
            {
                "trip_id": trip.trip_id,
                "trajectory": trip.trajectory,
                "source_path": str(trip.source_path),
                "kind": trip.kind,
                "n_rows": trip.n_rows,
                "notes": trip.notes,
                "stats": trip.stats,
                "config_fingerprint": fp,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return parquet_path
