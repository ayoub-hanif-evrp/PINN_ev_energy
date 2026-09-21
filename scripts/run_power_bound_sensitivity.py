#!/usr/bin/env python
"""Sensitivity: unbounded vs physically bounded wheel power.

The frozen paper protocol keeps apply_power_bounds=false. This script:
1. reports physics-only energy and power with and without the wheel clamp;
2. optionally retrains PINN under bounded wheel power in a separate output folder.

It does not replace the main LOTO numbers.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from evaluation.bootstrap import bootstrap_mean_ci, trip_absolute_errors  # noqa: E402
from experiment.loto import run_loto  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402


def _mae_row(pred: pd.DataFrame, method: str, variant: str) -> dict:
    err = trip_absolute_errors(pred, method)
    stats = bootstrap_mean_ci(err.to_numpy(dtype=float))
    signed = pred.loc[pred["method"] == method].groupby("trip_id")["signed_error_kwh"].mean()
    return {
        "variant": variant,
        "method": method,
        "mae_kwh": stats["mean"],
        "mae_ci_low": stats["ci_low"],
        "mae_ci_high": stats["ci_high"],
        "bias_kwh": float(signed.mean()) if len(signed) else float("nan"),
        "n_trips": int(stats["n"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Wheel-power bound sensitivity.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--retrain-pinn", action="store_true", help="Retrain PINN with bounded wheel power.")
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds for optional PINN retrain.")
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    out_dir = resolve_under_root("outputs/sensitivity", root) / profile
    out_dir.mkdir(parents=True, exist_ok=True)

    phys_path = out_dir / "physics_unbounded_vs_bounded.csv"
    rows = []
    if phys_path.exists():
        phys = pd.read_csv(phys_path)
        for variant, sub in phys.groupby("variant"):
            err = sub.groupby("trip_id")["absolute_error_kwh"].mean()
            stats = bootstrap_mean_ci(err.to_numpy(dtype=float))
            rows.append(
                {
                    "variant": variant,
                    "method": "physics",
                    "mae_kwh": stats["mean"],
                    "mae_ci_low": stats["ci_low"],
                    "mae_ci_high": stats["ci_high"],
                    "bias_kwh": float(sub.groupby("trip_id")["signed_error_kwh"].mean().mean()),
                    "n_trips": int(sub["trip_id"].nunique()),
                }
            )

    loto_pred = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root) / profile / "per_trip_predictions.csv"
    if loto_pred.exists():
        pred = pd.read_csv(loto_pred)
        if "pinn" in set(pred["method"]):
            rows.append(_mae_row(pred, "pinn", "unbounded_protocol_pinn"))
        if "physics" in set(pred["method"]):
            rows.append(_mae_row(pred, "physics", "unbounded_protocol_physics"))

    if args.retrain_pinn:
        bounded = deepcopy(config)
        bounded.setdefault("physics", {})["apply_power_bounds"] = True
        bounded.setdefault("experiment", {})["cache"] = True
        seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
        print(f"Retraining PINN with bounded wheel power, seeds={seeds}", flush=True)
        table = run_loto(
            bounded,
            methods=["physics", "pinn"],
            seeds=seeds,
            output_subdir=f"{profile}_wheel_bounds",
        )
        bound_dir = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root) / f"{profile}_wheel_bounds"
        table.to_csv(bound_dir / "per_trip_predictions.csv", index=False)
        for method in ("physics", "pinn"):
            if method in set(table["method"]):
                rows.append(_mae_row(table, method, "bounded_wheel_retrain"))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "power_bound_sensitivity.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
