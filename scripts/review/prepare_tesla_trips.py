#!/usr/bin/env python
"""Extract and preprocess Tesla Model Y driving sessions for review experiments.

Writes gitignored artifacts under data/tesla/. Does not touch results/ or paper configs.

Timestamp convention matches HELECAR-D / src/units.py:
  dt_s[i] = t[i+1] - t[i] for i < n-1; dt_s[n-1] = 0
  energy uses trapezoid on intervals i → i+1 via dt_s[:-1]
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.preprocessing import (  # noqa: E402
    acceleration_from_speed,
    compute_dt_seconds,
    elapsed_seconds,
    trapezoid_path_m,
)
from units import KMH_TO_MPS, energy_kwh_from_power  # noqa: E402


QC_DEFAULT = {
    "min_duration_min": 15.0,
    "min_dsoc_pp": 2.0,
    "min_moving_frac": 0.4,
    "min_coverage": 0.9,
}


def _find_zip(dataset_dir: Path) -> Path:
    zips = list(dataset_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No zip in {dataset_dir}")
    return zips[0]


def _ambient(df: pd.DataFrame) -> pd.Series:
    for col in ("VCFRONT_tempAmbient", "VCRIGHT_tempAmbientRaw"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def preprocess_drive(df: pd.DataFrame, trip_id: str, vehicle: str) -> dict:
    ts = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    order = np.argsort(ts.to_numpy())
    df = df.iloc[order].reset_index(drop=True)
    ts = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce").dt.tz_convert(None)

    # Provisional forward dt for acceleration smoothing before row drops.
    dt_prov, _ = compute_dt_seconds(ts.to_numpy(), fallback_dt_s=1.0)
    # Clip only provisional intervals used for accel; rebuild after cleaning.
    dt_for_acc = dt_prov.copy()
    if len(dt_for_acc) >= 2:
        mid = dt_for_acc[:-1]
        mid = np.clip(np.where(np.isfinite(mid) & (mid > 0), mid, 1.0), 0.2, 5.0)
        dt_for_acc[:-1] = mid

    speed_kmh = pd.to_numeric(df["DI_uiSpeed"], errors="coerce").to_numpy(dtype=float)
    speed_kmh = np.where(np.isfinite(speed_kmh), np.clip(speed_kmh, 0.0, None), 0.0)
    speed_mps = speed_kmh * KMH_TO_MPS
    v_smooth, acc = acceleration_from_speed(
        speed_mps, ts.to_numpy(), dt_for_acc, savgol_window=11, savgol_polyorder=2
    )

    soc = pd.to_numeric(df["SOCave292"], errors="coerce").to_numpy(dtype=float)
    volt = pd.to_numeric(df["BattVoltage132"], errors="coerce").to_numpy(dtype=float)
    current = pd.to_numeric(df["RawBattCurrent132"], errors="coerce").to_numpy(dtype=float)
    # Positive current = discharge (verified in dataset audit). Evaluation only.
    p_meas_kw = volt * current / 1000.0
    amb = _ambient(df).to_numpy(dtype=float)

    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "soc": soc,
            "speed_kmh": speed_kmh,
            "speed_mps": v_smooth,
            "acc_mps2": acc,
            "theta_rad": np.zeros(len(df), dtype=float),
            "grade": np.zeros(len(df), dtype=float),
            "temperature_c": amb,
            "lat": pd.to_numeric(df.get("GPSLatitude04F"), errors="coerce"),
            "lon": pd.to_numeric(df.get("GPSLongitude04F"), errors="coerce"),
            "batt_voltage_v": volt,
            "batt_current_a": current,
            "p_meas_kw": p_meas_kw,
        }
    )

    keep = (
        np.isfinite(frame["soc"])
        & np.isfinite(frame["speed_mps"])
        & np.isfinite(frame["batt_voltage_v"])
        & np.isfinite(frame["batt_current_a"])
        & np.isfinite(frame["temperature_c"])
        & frame["timestamp"].notna()
    )
    frame = frame.loc[keep].reset_index(drop=True)
    if len(frame) < 60:
        raise ValueError(f"{trip_id}: too few valid rows after cleaning")

    # Rebuild forward intervals from remaining timestamps (critical after drops).
    dt, dt_stats = compute_dt_seconds(frame["timestamp"].to_numpy(), fallback_dt_s=1.0)
    if len(dt) >= 2:
        mid = dt[:-1]
        mid = np.clip(np.where(np.isfinite(mid) & (mid > 0), mid, 1.0), 0.2, 5.0)
        dt = dt.copy()
        dt[:-1] = mid
        dt[-1] = 0.0
    frame["dt_s"] = dt

    sp = frame["speed_mps"].to_numpy(dtype=float)
    ds_m, s_m = trapezoid_path_m(sp, dt)
    frame["s_m"] = s_m
    frame["s_can_m"] = s_m
    frame["ds_m"] = ds_m
    frame["elapsed_s"] = elapsed_seconds(dt)
    frame["cumulative_distance_km"] = s_m / 1000.0

    e_meas = energy_kwh_from_power(frame["p_meas_kw"].to_numpy(dtype=float), dt)
    soc_arr = frame["soc"].to_numpy(dtype=float)
    meta = {
        "trip_id": trip_id,
        "vehicle": vehicle,
        "trajectory": vehicle,
        "n_rows": int(len(frame)),
        "duration_s": float(np.sum(dt[:-1])) if len(dt) >= 2 else 0.0,
        "distance_km": float(s_m[-1]) / 1000.0 if len(s_m) else 0.0,
        "soc_start": float(soc_arr[0]),
        "soc_end": float(soc_arr[-1]),
        "dsoc_pp": float(soc_arr[0] - soc_arr[-1]),
        "speed_mean_kmh": float(frame["speed_kmh"].mean()),
        "moving_frac": float((frame["speed_kmh"] > 1.0).mean()),
        "e_meas_kwh": float(e_meas),
        "grade_assumption": "flat_road_theta_0",
        "speed_source": "DI_uiSpeed",
        "current_sign": "positive_current_equals_discharge",
        "dt_convention": "forward_interval_i_to_iplus1_last_zero",
        "dt_stats": dt_stats,
    }
    return {"frame": frame, "meta": meta}


def passes_qc(meta: dict, qc: dict) -> bool:
    return (
        meta["duration_s"] / 60.0 >= qc["min_duration_min"]
        and meta["dsoc_pp"] >= qc["min_dsoc_pp"]
        and meta["moving_frac"] >= qc["min_moving_frac"]
        and meta["n_rows"] >= 60
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="BEV energy dynamics dataset")
    parser.add_argument("--out", default="data/tesla")
    parser.add_argument("--vehicle", choices=["BEV1", "BEV2", "both"], default="both")
    parser.add_argument("--max-trips", type=int, default=0, help="0 = all QC-passing trips")
    args = parser.parse_args()

    dataset_dir = ROOT / args.dataset_dir
    out_dir = ROOT / args.out
    trips_dir = out_dir / "trips"
    trips_dir.mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(_find_zip(dataset_dir))
    drives = [n for n in z.namelist() if "driving sessions/" in n and n.endswith(".csv")]
    vehicles = ["BEV1", "BEV2"] if args.vehicle == "both" else [args.vehicle]

    metas = []
    for vehicle in vehicles:
        files = [n for n in drives if Path(n).name.startswith(f"{vehicle}_")]
        kept = 0
        for n in files:
            trip_id = Path(n).stem
            try:
                raw = pd.read_csv(z.open(n), low_memory=False)
                pack = preprocess_drive(raw, trip_id, vehicle)
            except Exception as exc:  # noqa: BLE001
                print(f"SKIP {trip_id}: {exc}")
                continue
            if not passes_qc(pack["meta"], QC_DEFAULT):
                continue
            pack["frame"].to_csv(trips_dir / f"{trip_id}.csv", index=False)
            metas.append(pack["meta"])
            kept += 1
            if args.max_trips and kept >= args.max_trips:
                break
        print(f"{vehicle}: kept {kept} QC-passing trips")

    meta_df = pd.DataFrame(metas)
    meta_df.to_csv(out_dir / "trip_index.csv", index=False)
    (out_dir / "qc.json").write_text(json.dumps(QC_DEFAULT, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "KU Leuven BEV energy dynamics",
                "doi": "10.48804/8KPDTW",
                "n_trips": int(len(meta_df)),
                "by_vehicle": meta_df.groupby("vehicle").size().to_dict() if len(meta_df) else {},
                "grade_assumption": "flat_road_theta_0",
                "dt_convention": "forward_interval_i_to_iplus1_last_zero",
                "features": ["speed_mps", "acc_mps2", "theta_rad", "temperature_c"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(meta_df)} trips under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
