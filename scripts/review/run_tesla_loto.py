#!/usr/bin/env python
"""Leave-one-trip-out Tesla review experiment (separate from Twizy paper LOTO)."""

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
from experiment.loto import run_loto  # noqa: E402
from experiment.sanity import write_sanity_report  # noqa: E402
from paths import resolve_under_root, project_root  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402
from physics.vehicle_model import observed_energy_kwh  # noqa: E402
from tesla_data import load_tesla_pool  # noqa: E402


def _apply_calibrated_L(config: dict, vehicle: str) -> float:
    path = ROOT / "review_results" / "tesla" / f"residual_calibration_{vehicle.lower()}.json"
    if path.exists():
        chosen = float(json.loads(path.read_text(encoding="utf-8"))["chosen_residual_limit_kw"])
        config.setdefault("training", {})["residual_limit_kw"] = chosen
        return chosen
    return float(config.get("training", {}).get("residual_limit_kw", 40.0))


def _add_meas_eval(table: pd.DataFrame, trips, params) -> pd.DataFrame:
    """Append independent V×I energy reference columns (evaluation only)."""
    by_id = {t.trip_id: t for t in trips}
    e_meas = []
    e_soc = []
    for _, row in table.iterrows():
        trip = by_id[str(row["trip_id"])]
        e_m = float(trip.stats.get("e_meas_kwh", np.nan))
        soc = trip.frame["soc"].to_numpy(dtype=float)
        e_s = observed_energy_kwh(float(soc[0]), float(soc[-1]), params.battery_capacity_kwh)
        e_meas.append(e_m)
        e_soc.append(e_s)
    out = table.copy()
    out["e_meas_kwh"] = e_meas
    out["e_soc_kwh"] = e_soc
    out["abs_err_vs_meas_kwh"] = (out["predicted_energy_kwh"] - out["e_meas_kwh"]).abs()
    out["abs_err_vs_soc_kwh"] = (out["predicted_energy_kwh"] - out["e_soc_kwh"]).abs()
    out["signed_bias_vs_meas_kwh"] = out["predicted_energy_kwh"] - out["e_meas_kwh"]
    return out


def _summarize(table: pd.DataFrame, ref_col: str, err_col: str) -> pd.DataFrame:
    rows = []
    for method, g in table.groupby("method"):
        y = g[ref_col].to_numpy(dtype=float)
        yhat = g["predicted_energy_kwh"].to_numpy(dtype=float)
        err = g[err_col].to_numpy(dtype=float)
        mae = float(np.nanmean(err))
        rmse = float(np.sqrt(np.nanmean((yhat - y) ** 2)))
        wape = float(100.0 * np.nansum(err) / np.nansum(np.abs(y))) if np.nansum(np.abs(y)) else float("nan")
        bias = float(np.nanmean(yhat - y))
        rows.append(
            {
                "method": method,
                "n_trips": int(g["trip_id"].nunique()),
                "ref": ref_col,
                "mae_kwh": mae,
                "rmse_kwh": rmse,
                "wape_pct": wape,
                "bias_kwh": bias,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=0, help="0=all pool trips as test folds")
    args = parser.parse_args()

    config = load_config(args.config)
    vehicle = str(config["tesla"]["vehicle"])
    L = _apply_calibrated_L(config, vehicle)
    print(f"Using residual_limit_kw={L}")

    trips, pool_meta = load_tesla_pool(config, ROOT)
    test_ids = list(pool_meta["trip_ids"])
    if args.max_folds and args.max_folds < len(test_ids):
        test_ids = test_ids[: args.max_folds]

    table = run_loto(
        config,
        methods=args.methods,
        seeds=args.seeds,
        test_ids=test_ids,
        trips=trips,
    )
    params = load_vehicle_parameters(config=config)
    enriched = _add_meas_eval(table, trips, params)

    profile = str(config.get("experiment", {}).get("profile", "tesla"))
    out_dir = resolve_under_root(config.get("paths", {}).get("loto_dir", "review_results/tesla/loto"), project_root()) / profile
    enriched.to_csv(out_dir / "trip_energy_with_meas_ref.csv", index=False)
    summary_meas = _summarize(enriched, "e_meas_kwh", "abs_err_vs_meas_kwh")
    summary_soc = _summarize(enriched, "e_soc_kwh", "abs_err_vs_soc_kwh")
    summary_meas.to_csv(out_dir / "summary_vs_meas.csv", index=False)
    summary_soc.to_csv(out_dir / "summary_vs_soc.csv", index=False)
    # Agreement between SoC-derived and measured references on the pool.
    agree = enriched.drop_duplicates("trip_id")[["trip_id", "e_soc_kwh", "e_meas_kwh"]].copy()
    agree["abs_diff_kwh"] = (agree["e_soc_kwh"] - agree["e_meas_kwh"]).abs()
    agree.to_csv(out_dir / "soc_vs_meas_agreement.csv", index=False)
    # Model Y can briefly exceed 200 kW battery power; use a Tesla-aware diagnostic.
    absurd_kw = float(config.get("tesla", {}).get("absurd_power_kw", 400.0))
    write_sanity_report(out_dir, table, profile=profile, absurd_power_kw=absurd_kw)
    print(summary_meas.to_string(index=False))
    print(summary_soc.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
