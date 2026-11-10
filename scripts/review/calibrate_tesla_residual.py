#!/usr/bin/env python
"""Physics-only residual-bound calibration for Tesla review experiments.

CRITICAL: Measured battery power / voltage / current are NEVER used here.

Protocol:
1. Load the frozen evaluation pool IDs (must not be used for L selection).
2. Sample a disjoint calibration corpus from remaining QC-passing trips.
3. On calibration trips only, compute longitudinal P_physics.
4. Choose L = smallest grid value ≥ 0.5 * P95(|P_physics|) among {10,20,40,60,80} kW.
5. Freeze L before any held-out neural evaluation metrics are inspected.

This replaces the earlier leaked protocol that used |P_meas - P_physics| on the
evaluation pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "review"))

from config import load_config  # noqa: E402
from models.baselines import physics_only_predict  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402
from tesla_data import load_tesla_trip_csv, select_pool  # noqa: E402

GRID = (10.0, 20.0, 40.0, 60.0, 80.0)
PHYS_FRACTION = 0.5  # predetermined: residual capacity as fraction of P95(|P_physics|)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--calib-size", type=int, default=17)
    parser.add_argument("--calib-seed", type=int, default=2027)
    args = parser.parse_args()
    config = load_config(args.config)
    vehicle = str(config["tesla"]["vehicle"])
    params = load_vehicle_parameters(config=config)

    trips_dir = ROOT / config["tesla"].get("trips_dir", "data/tesla/trips")
    index = pd.read_csv(ROOT / config["tesla"].get("trip_index", "data/tesla/trip_index.csv"))
    index = index.loc[index["vehicle"] == vehicle].copy()
    index = index.loc[
        (index["duration_s"] / 60.0 >= 15.0)
        & (index["dsoc_pp"] >= 2.0)
        & (index["moving_frac"] >= 0.4)
    ].copy()

    pool_file = ROOT / config["tesla"]["pool_file"]
    eval_ids = set(json.loads(pool_file.read_text(encoding="utf-8"))["trip_ids"])
    remaining = index.loc[~index["trip_id"].isin(eval_ids)].copy()
    if remaining.empty:
        raise SystemExit("No QC trips left outside the evaluation pool for calibration")

    calib_file = ROOT / "review_results" / "tesla" / "pools" / f"{vehicle.lower()}_calib_pool.json"
    if calib_file.exists():
        calib_meta = json.loads(calib_file.read_text(encoding="utf-8"))
        calib_ids = list(calib_meta["trip_ids"])
    else:
        calib_ids = select_pool(remaining, vehicle, args.calib_size, args.calib_seed)
        overlap = set(calib_ids) & eval_ids
        if overlap:
            raise SystemExit(f"Calibration/eval pool overlap: {overlap}")
        calib_meta = {
            "vehicle": vehicle,
            "pool_size_requested": args.calib_size,
            "pool_seed": args.calib_seed,
            "n_trips": len(calib_ids),
            "trip_ids": calib_ids,
            "selection": "stratified_dsoc_x_distance_tertiles_disjoint_from_eval",
            "eval_pool_file": str(pool_file.relative_to(ROOT)).replace("\\", "/"),
            "forbidden_eval_ids": sorted(eval_ids),
            "rule": "physics_only_no_Pmeas",
        }
        calib_file.parent.mkdir(parents=True, exist_ok=True)
        calib_file.write_text(json.dumps(calib_meta, indent=2), encoding="utf-8")

    if set(calib_ids) & eval_ids:
        raise SystemExit("Calibration IDs intersect evaluation pool — abort")

    abs_phys = []
    for tid in calib_ids:
        trip = load_tesla_trip_csv(trips_dir / f"{tid}.csv", vehicle)
        phys = physics_only_predict(trip, params, apply_bounds=False)
        p_phys = np.asarray(phys["p_battery_kw"], dtype=float)
        m = np.isfinite(p_phys)
        if m.any():
            abs_phys.append(np.abs(p_phys[m]))
    if not abs_phys:
        raise SystemExit("No finite P_physics on calibration trips")
    stacked = np.concatenate(abs_phys)
    p95 = float(np.percentile(stacked, 95))
    target = PHYS_FRACTION * p95
    above = [L for L in GRID if L >= target]
    chosen = float(min(above) if above else GRID[-1])

    out = {
        "vehicle": vehicle,
        "n_calib_trips": len(calib_ids),
        "calib_pool_file": str(calib_file.relative_to(ROOT)).replace("\\", "/"),
        "eval_pool_file": str(pool_file.relative_to(ROOT)).replace("\\", "/"),
        "n_eval_trips_excluded": len(eval_ids),
        "abs_Pphysics_p50_kw": float(np.percentile(stacked, 50)),
        "abs_Pphysics_p95_kw": p95,
        "target_L_kw": target,
        "phys_fraction": PHYS_FRACTION,
        "grid_kw": list(GRID),
        "chosen_residual_limit_kw": chosen,
        "rule": "smallest_grid_ge_0.5_times_P95_abs_Pphysics_on_disjoint_calib",
        "uses_measured_power": False,
        "note": "No voltage/current/P_meas used. Eval-pool trips excluded from L selection.",
        "supersedes": "prior residual_calibration that used |Pmeas-Pphysics| on eval pool",
    }
    out_path = ROOT / "review_results" / "tesla" / f"residual_calibration_{vehicle.lower()}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
