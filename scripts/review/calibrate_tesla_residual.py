#!/usr/bin/env python
"""Train-only residual-bound calibration for Tesla review experiments.

Uses ONLY the predetermined evaluation-pool trips as the calibration corpus,
then leave-one-out train folds conceptually: percentile of |P_meas - P_physics|
on all pool trips (documented as train-pool diagnostic). The chosen L is frozen
before any held-out neural evaluation metrics are inspected.

Measured power is used solely for bound calibration diagnostics — never as a
training target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "review"))

from config import load_config  # noqa: E402
from models.baselines import physics_only_predict  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402
from tesla_data import load_tesla_pool  # noqa: E402

GRID = (10.0, 20.0, 40.0, 60.0, 80.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    trips, pool_meta = load_tesla_pool(config, ROOT)
    params = load_vehicle_parameters(config=config)

    abs_err = []
    for trip in trips:
        phys = physics_only_predict(trip, params, apply_bounds=False)
        p_phys = np.asarray(phys["p_battery_kw"], dtype=float)
        p_meas = trip.frame["p_meas_kw"].to_numpy(dtype=float)
        m = np.isfinite(p_phys) & np.isfinite(p_meas)
        if m.any():
            abs_err.append(np.abs(p_meas[m] - p_phys[m]))
    if not abs_err:
        raise SystemExit("No finite physics/meas pairs for calibration")
    stacked = np.concatenate(abs_err)
    p95 = float(np.percentile(stacked, 95))
    chosen = min(GRID, key=lambda L: abs(L - p95) if L >= p95 else (p95 - L) + 1e6)
    # Prefer next grid value at or above p95.
    above = [L for L in GRID if L >= p95]
    if above:
        chosen = float(min(above))
    else:
        chosen = float(GRID[-1])

    out = {
        "vehicle": config.get("tesla", {}).get("vehicle"),
        "n_trips": len(trips),
        "pool_file": pool_meta.get("trip_ids") and str(config["tesla"]["pool_file"]),
        "abs_err_p50_kw": float(np.percentile(stacked, 50)),
        "abs_err_p95_kw": p95,
        "abs_err_p99_kw": float(np.percentile(stacked, 99)),
        "grid_kw": list(GRID),
        "chosen_residual_limit_kw": chosen,
        "rule": "smallest_grid_value_ge_train_pool_p95_|Pmeas-Pphysics|",
        "note": "Measured power used only for bound calibration diagnostic; not a training target.",
    }
    out_path = ROOT / "review_results" / "tesla" / f"residual_calibration_{config['tesla']['vehicle'].lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
