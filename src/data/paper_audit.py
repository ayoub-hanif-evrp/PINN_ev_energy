"""Paper-facing dataset summary computed from the real loaded trips."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from data.feature_audit import feature_distribution_table
from data.preprocessing import ProcessedTrip
from data.quantization import combine_quantization, estimate_soc_quantization
from data.windows import generate_windows
from physics.vehicle_model import observed_energy_kwh


def paper_dataset_summary(
    trips: list[ProcessedTrip],
    battery_capacity_kwh: float,
    q: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    traj = Counter(t.trajectory for t in trips)
    durs = []
    dist_can = []
    dist_gps = []
    dist_hav = []
    soc_start = []
    soc_end = []
    gaps = []
    soc_inc = 0
    grade_clip = 0
    for t in trips:
        df = t.frame
        durs.append(float(np.nansum(df["dt_s"])))
        dist_can.append(float(df["s_can_m"].iloc[-1]) / 1000.0 if "s_can_m" in df.columns else np.nan)
        dist_gps.append(float(df["s_gps_speed_m"].iloc[-1]) / 1000.0 if "s_gps_speed_m" in df.columns else np.nan)
        dist_hav.append(float(df["s_haversine_m"].iloc[-1]) / 1000.0 if "s_haversine_m" in df.columns else np.nan)
        soc = df["soc"].to_numpy(dtype=float)
        soc_start.append(float(soc[0]))
        soc_end.append(float(soc[-1]))
        soc_inc += int(np.sum(np.diff(soc) > 1e-9))
        dt = df["dt_s"].to_numpy(dtype=float)
        gaps.append(int(np.sum(dt[:-1] > 2.0)) if len(dt) else 0)
        grade_info = t.stats.get("grade") or {}
        grade_clip += int(grade_info.get("n_clipped", 0))
    per_q = [estimate_soc_quantization(t.frame["soc"]) for t in trips]
    cq = combine_quantization(per_q)
    windows = generate_windows(trips, battery_capacity_kwh, q, config=config, overlapping=True)
    by_scale = Counter(w.scale for w in windows)
    by_trip = Counter(w.trip_id for w in windows)
    feats = feature_distribution_table(trips)
    high_missing = feats.loc[feats["missing_fraction"] > 0.5, "feature"].tolist()
    row = {
        "n_trips": len(trips),
        "n_T1": int(traj.get("T1", 0)),
        "n_T2": int(traj.get("T2", 0)),
        "n_T3": int(traj.get("T3", 0)),
        "duration_s_min": float(np.min(durs)) if durs else np.nan,
        "duration_s_max": float(np.max(durs)) if durs else np.nan,
        "duration_s_mean": float(np.mean(durs)) if durs else np.nan,
        "distance_can_km_min": float(np.nanmin(dist_can)) if dist_can else np.nan,
        "distance_can_km_max": float(np.nanmax(dist_can)) if dist_can else np.nan,
        "distance_gps_speed_km_mean": float(np.nanmean(dist_gps)) if dist_gps else np.nan,
        "distance_haversine_km_mean": float(np.nanmean(dist_hav)) if dist_hav else np.nan,
        "soc_start_mean": float(np.mean(soc_start)) if soc_start else np.nan,
        "soc_end_mean": float(np.mean(soc_end)) if soc_end else np.nan,
        "soc_depletion_mean_pp": float(np.mean(np.array(soc_start) - np.array(soc_end))) if soc_start else np.nan,
        "empirical_q": float(cq.get("q", np.nan)),
        "soc_increase_counts": int(soc_inc),
        "timestamp_gaps_gt_2s": int(np.sum(gaps)),
        "grade_clip_samples": int(grade_clip),
        "n_windows_total": int(len(windows)),
        "high_missing_features": ",".join(high_missing),
    }
    for scale, n in sorted(by_scale.items()):
        row[f"windows_{scale}"] = int(n)
    return pd.DataFrame([row])


def trip_characteristics_table(trips: list[ProcessedTrip], battery_capacity_kwh: float) -> pd.DataFrame:
    rows = []
    for t in trips:
        df = t.frame
        soc = df["soc"].to_numpy(dtype=float)
        rows.append(
            {
                "trip_id": t.trip_id,
                "trajectory": t.trajectory,
                "n_rows": t.n_rows,
                "duration_s": float(np.nansum(df["dt_s"])),
                "distance_can_km": float(df["s_can_m"].iloc[-1]) / 1000.0 if "s_can_m" in df.columns else np.nan,
                "distance_gps_speed_km": float(df["s_gps_speed_m"].iloc[-1]) / 1000.0 if "s_gps_speed_m" in df.columns else np.nan,
                "distance_haversine_km": float(df["s_haversine_m"].iloc[-1]) / 1000.0 if "s_haversine_m" in df.columns else np.nan,
                "soc_start": float(soc[0]),
                "soc_end": float(soc[-1]),
                "soc_depletion_pp": float(soc[0] - soc[-1]),
                "e_obs_kwh": observed_energy_kwh(float(soc[0]), float(soc[-1]), battery_capacity_kwh),
            }
        )
    return pd.DataFrame(rows)
