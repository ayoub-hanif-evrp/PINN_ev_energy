"""Feature construction. LAT/LON/Date/Time/SoC are not predictors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from physics.vehicle_model import observed_energy_kwh
from data.preprocessing import ProcessedTrip
from data.schema import (
    ELASTICNET_FEATURES,
    EXCLUDED_MAIN_PREDICTORS,
    MAIN_MODEL_FEATURES,
    METADATA_COLUMNS,
    SOC_LABEL_COLUMNS,
)


def elevation_gain_loss_from_grade(grade: np.ndarray, ds_m: np.ndarray) -> tuple[float, float]:
    """Cumulative ascent/descent from grade × spatial increment (OPTION B).

    Grade is rise/run estimated from Savitzky–Golay-smoothed altitude over a
    spatial window (the same altitude used by the physics model). Haversine
    ``ds_m`` provides the spatial increment. This is *not* the sum of successive
    raw GPS altitude differences, which inflate ascent under high-frequency noise.

        dh_i = grade_i * ds_i
        gain = Σ max(dh_i, 0)
        loss = Σ max(−dh_i, 0)
    """
    dh = np.asarray(grade, dtype=float) * np.asarray(ds_m, dtype=float)
    dh = np.where(np.isfinite(dh), dh, 0.0)
    gain = float(np.sum(np.clip(dh, 0.0, None)))
    loss = float(np.sum(np.clip(-dh, 0.0, None)))
    return gain, loss


def sample_feature_frame(trip: ProcessedTrip, feature_names: tuple[str, ...] | None = None) -> pd.DataFrame:
    names = list(feature_names or MAIN_MODEL_FEATURES)
    missing = [n for n in names if n not in trip.frame.columns]
    if missing:
        raise KeyError(f"{trip.trip_id}: missing feature columns {missing}")
    forbidden = [n for n in names if n in EXCLUDED_MAIN_PREDICTORS or n in SOC_LABEL_COLUMNS]
    if forbidden:
        raise ValueError(f"Main model features must not include {forbidden}")
    return trip.frame[names].copy()


def trip_level_features(trip: ProcessedTrip, battery_capacity_kwh: float = 6.0) -> dict[str, Any]:
    """Trip summaries. SoC fields are labels only; they are not ElasticNet predictors."""
    df = trip.frame
    speed = df["speed_mps"].to_numpy(dtype=float)
    acc = df["acc_mps2"].to_numpy(dtype=float)
    grade = df["grade"].to_numpy(dtype=float) if "grade" in df.columns else np.zeros(len(df))
    dt = df["dt_s"].to_numpy(dtype=float)
    duration = float(np.nansum(dt))
    if "s_can_m" in df.columns:
        distance_m = float(df["s_can_m"].iloc[-1]) if len(df) else 0.0
    else:
        distance_m = float(df["s_m"].iloc[-1]) if len(df) and "s_m" in df.columns else 0.0
    idle = speed < 0.5
    pos_acc = acc > 0.3
    neg_acc = acc < -0.3
    if "ds_m" in df.columns:
        ds = df["ds_m"].to_numpy(dtype=float)
    else:
        ds = np.zeros(len(df), dtype=float)
    elev_gain, elev_loss = elevation_gain_loss_from_grade(grade, ds)

    out: dict[str, Any] = {
        "trip_id": trip.trip_id,
        "trajectory": trip.trajectory,
        "n_rows": trip.n_rows,
        "distance_km": distance_m / 1000.0,
        "duration_s": duration,
        "mean_speed_mps": float(np.nanmean(speed)),
        "std_speed_mps": float(np.nanstd(speed)),
        "max_speed_mps": float(np.nanmax(speed)) if len(speed) else np.nan,
        "idle_fraction": float(np.mean(idle)) if len(speed) else np.nan,
        "mean_positive_acc": float(np.nanmean(np.clip(acc, 0.0, None))),
        "acc_std": float(np.nanstd(acc)),
        "positive_acc_time_fraction": float(np.mean(pos_acc)) if len(acc) else np.nan,
        "braking_time_fraction": float(np.mean(neg_acc)) if len(acc) else np.nan,
        "cumulative_elevation_gain_m": elev_gain,
        "cumulative_elevation_loss_m": elev_loss,
        "mean_abs_grade": float(np.nanmean(np.abs(grade))),
    }
    if "temperature_c" in df.columns:
        out["mean_temperature_c"] = float(pd.to_numeric(df["temperature_c"], errors="coerce").mean())
    if "humidity_pct" in df.columns:
        out["mean_humidity_pct"] = float(pd.to_numeric(df["humidity_pct"], errors="coerce").mean())
    if "wind_speed_mps" in df.columns:
        out["mean_wind_speed_mps"] = float(pd.to_numeric(df["wind_speed_mps"], errors="coerce").mean())
    if "traffic" in df.columns:
        traffic = pd.to_numeric(df["traffic"], errors="coerce")
        out["mean_traffic"] = float(traffic.mean())
        for level in (0, 1, 2):
            out[f"traffic_frac_{level}"] = float((traffic == level).mean())
    if "speed_limit_kmh" in df.columns:
        out["mean_speed_limit_kmh"] = float(pd.to_numeric(df["speed_limit_kmh"], errors="coerce").mean())
    if "soc" in df.columns:
        soc = pd.to_numeric(df["soc"], errors="coerce")
        out["soc_start"] = float(soc.iloc[0])
        out["soc_end"] = float(soc.iloc[-1])
        out["soc_delta"] = float(soc.iloc[0] - soc.iloc[-1])
        out["e_obs_kwh"] = observed_energy_kwh(out["soc_start"], out["soc_end"], battery_capacity_kwh)
    return out


def trip_level_feature_table(
    trips: list[ProcessedTrip],
    battery_capacity_kwh: float = 6.0,
) -> pd.DataFrame:
    return pd.DataFrame([trip_level_features(t, battery_capacity_kwh=battery_capacity_kwh) for t in trips])


def is_soc_derived_column(name: str) -> bool:
    key = name.strip().lower()
    if key in {c.lower() for c in SOC_LABEL_COLUMNS}:
        return True
    return key.startswith("soc") or "dsoc" in key or key.endswith("_soc")


def elasticnet_feature_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """Whitelist-only design matrix. Raises if any SoC-derived column is requested."""
    # Labels may sit in the same table; they must not enter X.
    available = [c for c in ELASTICNET_FEATURES if c in table.columns]
    forbidden = [c for c in available if is_soc_derived_column(c)]
    if forbidden:
        raise ValueError(f"ElasticNet features must not include SoC-derived columns: {forbidden}")
    missing = [c for c in ELASTICNET_FEATURES if c not in table.columns]
    if missing:
        raise KeyError(f"ElasticNet whitelist missing columns: {missing}")
    x = table.loc[:, list(ELASTICNET_FEATURES)].copy()
    leaked_in_x = [c for c in x.columns if is_soc_derived_column(str(c))]
    if leaked_in_x:
        raise ValueError(f"Leakage: SoC-derived columns in ElasticNet X: {leaked_in_x}")
    return x


def elasticnet_target(table: pd.DataFrame, column: str = "e_obs_kwh") -> pd.Series:
    if column not in table.columns:
        raise KeyError(f"Target column {column} is not in the trip table.")
    if column in ELASTICNET_FEATURES:
        raise ValueError(f"Target {column} must not also be an ElasticNet predictor.")
    return table[column]


def assert_no_soc_in_predictors(columns: list[str] | tuple[str, ...]) -> None:
    leaked = [c for c in columns if is_soc_derived_column(str(c)) or str(c) in METADATA_COLUMNS]
    soc_leaked = [c for c in columns if is_soc_derived_column(str(c))]
    if soc_leaked:
        raise AssertionError(f"SoC-derived predictor leakage: {soc_leaked}")
