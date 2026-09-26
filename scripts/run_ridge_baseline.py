#!/usr/bin/env python
"""Fit Ridge on existing LOTO folds. Does not retrain neural models."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from data.features import trip_level_feature_table  # noqa: E402
from evaluation.metrics import trip_energy_row  # noqa: E402
from experiment.data import load_processed  # noqa: E402
from experiment.loto import trip_obs  # noqa: E402
from models.ridge import TripRidge  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402


def main() -> int:
    config = load_config("configs/paper.yaml")
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    pred_path = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root) / profile / "per_trip_predictions.csv"
    if not pred_path.exists():
        raise SystemExit(f"Missing neural/physics LOTO predictions: {pred_path}")
    existing = pd.read_csv(pred_path)
    kept = existing.loc[~existing["method"].isin(["elasticnet", "ridge"])].copy()
    trips = load_processed(config)
    params = load_vehicle_parameters(config=config)
    rows = []
    alphas = []
    for test in trips:
        outer = [t for t in trips if t.trip_id != test.trip_id]
        obs = trip_obs(test, params.battery_capacity_kwh)
        train_table = trip_level_feature_table(outer, battery_capacity_kwh=params.battery_capacity_kwh)
        test_table = trip_level_feature_table([test], battery_capacity_kwh=params.battery_capacity_kwh)
        model = TripRidge().fit(train_table)
        model.assert_not_fitted_on(test.trip_id)
        pred = float(model.predict(test_table)[0])
        row = trip_energy_row(
            test.trip_id, test.trajectory, "ridge", 0,
            obs["distance_km"], obs["duration_s"], obs["e_obs_kwh"], pred,
        )
        rows.append(row)
        alphas.append({"trip_id": test.trip_id, "alpha": model.best_params_["alpha"], "n_train": len(outer)})
        print(f"ridge test={test.trip_id} alpha={model.best_params_['alpha']} E_hat={pred:.4f} E_obs={obs['e_obs_kwh']:.4f}", flush=True)
    out = pd.concat([kept, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(pred_path, index=False)
    pd.DataFrame(alphas).to_csv(pred_path.parent / "ridge_alphas.csv", index=False)
    print(f"Wrote Ridge LOTO rows into {pred_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
