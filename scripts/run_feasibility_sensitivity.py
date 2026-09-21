#!/usr/bin/env python
"""Routing-oriented battery-feasibility sensitivity on held-out real trips.

Not an EVRP algorithm. No synthetic customers. Methods: Constant, Physics, PINN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from experiment.data import load_processed  # noqa: E402
from manifest import write_manifest  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402


def classify(e_hat: float, e_obs: float, soc_start: float, reserve_pct: float, e_batt: float) -> dict[str, bool | float]:
    budget = float(e_batt) * max(float(soc_start) - float(reserve_pct), 0.0) / 100.0
    model_feasible = float(e_hat) <= budget
    obs_feasible = float(e_obs) <= budget
    return {
        "energy_budget_kwh": budget,
        "model_feasible": model_feasible,
        "observed_feasible": obs_feasible,
        "false_safe": bool(model_feasible and not obs_feasible),
        "overly_conservative": bool((not model_feasible) and obs_feasible),
    }


def run_feasibility(config: dict) -> pd.DataFrame:
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    loto_dir = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root) / profile
    pred_path = loto_dir / "per_trip_predictions.csv"
    if not pred_path.exists():
        raise SystemExit(f"Need LOTO predictions at {pred_path} before feasibility sensitivity.")
    pred = pd.read_csv(pred_path)
    methods = ["constant", "physics", "pinn"]
    reserves = list(config.get("feasibility", {}).get("reserve_soc_pct", [5, 10, 15, 20]))
    params = load_vehicle_parameters(config=config)
    trips = {t.trip_id: t for t in load_processed(config)}
    rows = []
    for method in methods:
        sub = pred.loc[pred["method"] == method]
        if sub.empty:
            continue
        grouped = sub.groupby("trip_id")[["predicted_energy_kwh", "observed_energy_kwh"]].mean()
        for trip_id, rec in grouped.iterrows():
            trip = trips[str(trip_id)]
            soc0 = float(trip.frame["soc"].iloc[0])
            for reserve in reserves:
                decision = classify(float(rec["predicted_energy_kwh"]), float(rec["observed_energy_kwh"]), soc0, float(reserve), params.battery_capacity_kwh)
                rows.append(
                    {
                        "trip_id": trip_id,
                        "trajectory": trip.trajectory,
                        "method": method,
                        "reserve_soc_pct": float(reserve),
                        "predicted_energy_kwh": float(rec["predicted_energy_kwh"]),
                        "observed_energy_kwh": float(rec["observed_energy_kwh"]),
                        "soc_start": soc0,
                        **decision,
                    }
                )
    table = pd.DataFrame(rows)
    out = resolve_under_root(config.get("paths", {}).get("sensitivity_dir", "outputs/sensitivity"), root) / profile
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "feasibility.csv", index=False)
    summary = (
        table.groupby(["method", "reserve_soc_pct"])[["false_safe", "overly_conservative", "model_feasible", "observed_feasible"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(out / "feasibility_summary.csv", index=False)
    write_manifest(out / "feasibility_manifest.json", {"stage": "feasibility", "profile": profile, "reserves": reserves})
    print(f"Wrote feasibility sensitivity to {out}", flush=True)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Routing-oriented battery-feasibility sensitivity.")
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    run_feasibility(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
