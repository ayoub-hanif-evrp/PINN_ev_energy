"""Load preprocessed Tesla trips into ProcessedTrip objects (review-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.preprocessing import ProcessedTrip


REQUIRED_COLS = (
    "timestamp",
    "dt_s",
    "soc",
    "speed_mps",
    "acc_mps2",
    "theta_rad",
    "temperature_c",
    "s_m",
    "s_can_m",
    "ds_m",
    "elapsed_s",
    "cumulative_distance_km",
    "grade",
)


def load_tesla_trip_csv(path: Path, vehicle: str) -> ProcessedTrip:
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in frame.columns]
    if missing:
        raise KeyError(f"{path.name}: missing columns {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    # Evaluation-only measured energy columns (never used as training targets/inputs).
    for col in ("batt_voltage_v", "batt_current_a", "p_meas_kw", "speed_kmh"):
        if col not in frame.columns:
            frame[col] = np.nan
    notes = [
        "source=ku_leuven_bev_energy_dynamics",
        "grade_assumption=flat_road_theta_0",
        "current_sign=positive_discharge",
    ]
    e_meas = float("nan")
    if frame["p_meas_kw"].notna().any():
        e_meas = float(np.nansum(frame["p_meas_kw"].to_numpy(dtype=float) * frame["dt_s"].to_numpy(dtype=float) / 3600.0))
    stats = {
        "n_rows": int(len(frame)),
        "e_meas_kwh": e_meas,
        "vehicle": vehicle,
    }
    return ProcessedTrip(
        trip_id=path.stem,
        trajectory=vehicle,
        source_path=path,
        kind="tesla_drive",
        frame=frame,
        notes=notes,
        stats=stats,
    )


def select_pool(index: pd.DataFrame, vehicle: str, pool_size: int, pool_seed: int) -> list[str]:
    """Predetermined stratified sample by energy tertiles (seed fixed before training)."""
    sub = index.loc[index["vehicle"] == vehicle].copy()
    if sub.empty:
        raise ValueError(f"No trips for {vehicle}")
    sub = sub.sort_values("trip_id").reset_index(drop=True)
    # Energy strata from SoC-proxy distance×dsoc to avoid using V×I for selection bias.
    key = sub["dsoc_pp"].to_numpy(dtype=float) * sub["distance_km"].to_numpy(dtype=float)
    try:
        sub["stratum"] = pd.qcut(key, q=3, labels=False, duplicates="drop")
    except ValueError:
        sub["stratum"] = 0
    rng = np.random.default_rng(pool_seed)
    chosen: list[str] = []
    strata = sorted(sub["stratum"].unique())
    per = max(1, pool_size // max(len(strata), 1))
    for s in strata:
        ids = sub.loc[sub["stratum"] == s, "trip_id"].tolist()
        take = min(per, len(ids))
        pick = rng.choice(ids, size=take, replace=False).tolist()
        chosen.extend(str(x) for x in pick)
    # Fill remaining slots deterministically if short.
    remaining = [t for t in sub["trip_id"].tolist() if t not in set(chosen)]
    need = pool_size - len(chosen)
    if need > 0 and remaining:
        extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False).tolist()
        chosen.extend(str(x) for x in extra)
    chosen = sorted(set(chosen))[:pool_size]
    return chosen


def load_tesla_pool(config: dict[str, Any], root: Path) -> tuple[list[ProcessedTrip], dict[str, Any]]:
    tesla = config.get("tesla", {})
    vehicle = str(tesla["vehicle"])
    trips_dir = root / tesla.get("trips_dir", "data/tesla/trips")
    index_path = root / tesla.get("trip_index", "data/tesla/trip_index.csv")
    pool_file = root / tesla.get("pool_file", f"review_results/tesla/pools/{vehicle.lower()}_pool.json")
    pool_size = int(tesla.get("pool_size", 17))
    pool_seed = int(tesla.get("pool_seed", 2026))

    index = pd.read_csv(index_path)
    index = index.loc[index["vehicle"] == vehicle].copy()
    # Stricter QC matching audit protocol.
    index = index.loc[
        (index["duration_s"] / 60.0 >= 15.0)
        & (index["dsoc_pp"] >= 2.0)
        & (index["moving_frac"] >= 0.4)
    ].copy()

    pool_file.parent.mkdir(parents=True, exist_ok=True)
    if pool_file.exists():
        meta = json.loads(pool_file.read_text(encoding="utf-8"))
        trip_ids = list(meta["trip_ids"])
    else:
        trip_ids = select_pool(index, vehicle, pool_size, pool_seed)
        meta = {
            "vehicle": vehicle,
            "pool_size_requested": pool_size,
            "pool_seed": pool_seed,
            "n_trips": len(trip_ids),
            "trip_ids": trip_ids,
            "selection": "stratified_dsoc_x_distance_tertiles_before_training",
            "qc": {"min_duration_min": 15, "min_dsoc_pp": 2.0, "min_moving_frac": 0.4},
        }
        pool_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    trips: list[ProcessedTrip] = []
    for tid in trip_ids:
        path = trips_dir / f"{tid}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        trips.append(load_tesla_trip_csv(path, vehicle))
    return trips, meta
