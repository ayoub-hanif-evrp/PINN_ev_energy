#!/usr/bin/env python
"""Power-plausibility diagnostic from frozen LOTO reconstructions + physics.

Does not change the main unbounded-physics protocol.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from evaluation.power_plausibility import (  # noqa: E402
    battery_caps,
    load_npz_power,
    physics_arrays,
    row_for_power,
    summarize_method,
)
from experiment.data import load_processed  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402
from physics.vehicle_model import observed_energy_kwh  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical power plausibility diagnostic.")
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    loto_dir = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root) / profile
    out_dir = resolve_under_root("outputs/sensitivity", root) / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    trips = load_processed(config)
    params = load_vehicle_parameters(config=config)
    caps = battery_caps(params)
    rec_dir = loto_dir / "soc_reconstruction"
    pred = pd.read_csv(loto_dir / "per_trip_predictions.csv") if (loto_dir / "per_trip_predictions.csv").exists() else pd.DataFrame()
    rows: list[dict] = []
    phys_energy_rows: list[dict] = []

    for trip in trips:
        soc = trip.frame["soc"].to_numpy(dtype=float)
        e_obs = observed_energy_kwh(float(soc[0]), float(soc[-1]), params.battery_capacity_kwh)
        unbounded = physics_arrays(trip, params, apply_bounds=False)
        bounded = physics_arrays(trip, params, apply_bounds=True)
        for tag, arrs, bounds in (("physics", unbounded, False), ("physics_bounded_wheel", bounded, True)):
            rows.append(
                row_for_power(
                    trip_id=trip.trip_id,
                    trajectory=trip.trajectory,
                    method=tag,
                    seed=0,
                    p_batt=arrs["p_battery_kw"],
                    p_wheel=arrs["p_wheel_kw"],
                    p_phy=arrs["p_battery_kw"],
                    delta_p=None,
                    sat=None,
                    caps=caps,
                    apply_bounds=bounds,
                )
            )
            phys_energy_rows.append(
                {
                    "trip_id": trip.trip_id,
                    "trajectory": trip.trajectory,
                    "variant": tag,
                    "observed_energy_kwh": e_obs,
                    "predicted_energy_kwh": arrs["energy_kwh"],
                    "absolute_error_kwh": abs(arrs["energy_kwh"] - e_obs),
                    "signed_error_kwh": arrs["energy_kwh"] - e_obs,
                    "n_bound_applied": arrs["n_bound_applied"],
                }
            )

        methods = ["pinn", "pinn_fulltrip", "weak_mlp"]
        seeds = sorted(pred["seed"].unique().tolist()) if not pred.empty and "seed" in pred.columns else [0]
        for method in methods:
            for seed in seeds:
                data = load_npz_power(rec_dir, trip.trip_id, method, int(seed))
                if data is None or "p_hat" not in data:
                    continue
                sat = None
                if not pred.empty:
                    hit = pred.loc[
                        (pred["trip_id"] == trip.trip_id) & (pred["method"] == method) & (pred["seed"] == int(seed))
                    ]
                    if not hit.empty and "saturation_fraction" in hit.columns:
                        sat = float(hit["saturation_fraction"].iloc[0])
                rows.append(
                    row_for_power(
                        trip_id=trip.trip_id,
                        trajectory=trip.trajectory,
                        method=method,
                        seed=int(seed),
                        p_batt=data["p_hat"],
                        p_wheel=None,
                        p_phy=data.get("p_phy"),
                        delta_p=data.get("delta_p"),
                        sat=sat,
                        caps=caps,
                        apply_bounds=False,
                    )
                )

    per_trip = pd.DataFrame(rows)
    per_trip.to_csv(out_dir / "power_plausibility_per_trip.csv", index=False)
    summary = summarize_method(per_trip)
    summary.to_csv(out_dir / "power_plausibility_summary.csv", index=False)
    pd.DataFrame(phys_energy_rows).to_csv(out_dir / "physics_unbounded_vs_bounded.csv", index=False)
    pd.DataFrame([caps]).to_csv(out_dir / "power_capability_definitions.csv", index=False)
    print(f"Wrote power plausibility tables to {out_dir}", flush=True)
    print(summary.to_string(index=False), flush=True)
    energy = pd.DataFrame(phys_energy_rows)
    print("Physics energy MAE unbounded vs bounded wheel:", flush=True)
    print(energy.groupby("variant")["absolute_error_kwh"].mean().to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
