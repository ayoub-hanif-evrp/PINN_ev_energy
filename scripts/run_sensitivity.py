#!/usr/bin/env python
"""Assumption sensitivity: battery capacity, physics parameters, preprocessing.

Does not retune the frozen protocol from outer-test scores.
Vehicle parameters are modelling assumptions, not HELECAR measurements.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from experiment.data import load_processed  # noqa: E402
from manifest import write_manifest  # noqa: E402
from models.baselines import physics_only_predict  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402
from physics.vehicle_model import observed_energy_kwh  # noqa: E402


def _physics_table(trips, params) -> pd.DataFrame:
    rows = []
    for trip in trips:
        phys = physics_only_predict(trip, params)
        soc = trip.frame["soc"].to_numpy(dtype=float)
        e_obs = observed_energy_kwh(float(soc[0]), float(soc[-1]), params.battery_capacity_kwh)
        rows.append(
            {
                "trip_id": trip.trip_id,
                "trajectory": trip.trajectory,
                "observed_energy_kwh": e_obs,
                "physics_energy_kwh": phys["energy_kwh"],
                "signed_error_kwh": phys["energy_kwh"] - e_obs,
                "absolute_error_kwh": abs(phys["energy_kwh"] - e_obs),
            }
        )
    return pd.DataFrame(rows)


def run_sensitivity(config: dict[str, Any]) -> None:
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    out_dir = resolve_under_root(config.get("paths", {}).get("sensitivity_dir", "outputs/sensitivity"), root) / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    trips = load_processed(config)
    nominal = load_vehicle_parameters(config=config)
    nom_table = _physics_table(trips, nominal)
    nom_table.to_csv(out_dir / "physics_nominal.csv", index=False)
    nom_mae = float(nom_table["absolute_error_kwh"].mean())

    # Battery capacity: relabel observed energy. Physics P_hat is independent of E_batt.
    cap_rows = []
    loto_pred = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root) / profile / "per_trip_predictions.csv"
    pred = pd.read_csv(loto_pred) if loto_pred.exists() else pd.DataFrame()
    for cap in (5.5, 6.0, 6.5):
        alt = replace(nominal, battery_capacity_kwh=cap)
        tab = _physics_table(trips, alt)
        row = {
            "assumption": "battery_capacity_kwh",
            "value": cap,
            "physics_mae_kwh": float(tab["absolute_error_kwh"].mean()),
            "mean_observed_energy_kwh": float(tab["observed_energy_kwh"].mean()),
            "note": "E_obs scales with assumed usable capacity. Physics energy does not. PINN P_hat from the 6.0 kWh protocol is not retrained.",
        }
        if not pred.empty:
            # Scale observed labels; keep predicted energy from the frozen 6.0-trained models.
            scale = cap / 6.0
            for method in pred["method"].unique():
                sub = pred.loc[pred["method"] == method]
                e_obs = sub["observed_energy_kwh"].to_numpy(dtype=float) * scale
                e_hat = sub["predicted_energy_kwh"].to_numpy(dtype=float)
                row[f"{method}_mae_vs_relabelled_kwh"] = float(np.mean(np.abs(e_hat - e_obs)))
        cap_rows.append(row)
    pd.DataFrame(cap_rows).to_csv(out_dir / "battery_capacity.csv", index=False)

    # One-at-a-time physical parameters (physics-only energy).
    sweeps = {
        "mass_kg": (0.9 * nominal.mass_kg, nominal.mass_kg, 1.1 * nominal.mass_kg),
        "Crr": (0.008, 0.012, 0.016),
        "cda_m2": (0.50, 0.64, 0.80),
        "eta_drive": (0.75, 0.85, 0.95),
        "eta_regen": (0.20, 0.40, 0.60),
        "auxiliary_power_kw": (0.10, 0.20, 0.40),
    }
    oat = []
    for name, values in sweeps.items():
        for value in values:
            params = replace(nominal, **{name: float(value)})
            tab = _physics_table(trips, params)
            oat.append(
                {
                    "parameter": name,
                    "value": float(value),
                    "nominal": float(getattr(nominal, name)),
                    "physics_mae_kwh": float(tab["absolute_error_kwh"].mean()),
                    "physics_bias_kwh": float(tab["signed_error_kwh"].mean()),
                    "delta_mae_vs_nominal": float(tab["absolute_error_kwh"].mean()) - nom_mae,
                    "provenance": "modelling assumption; not a HELECAR-D measurement",
                }
            )
    pd.DataFrame(oat).to_csv(out_dir / "physical_parameters.csv", index=False)

    # Preprocessing: speed/altitude smoother windows. Physics energy only.
    from data.discovery import discover_dataset
    from data.loader import load_selected_trips
    from data.preprocessing import preprocess_trips

    raw = load_selected_trips(discover_dataset(config).selected)
    pre_rows = []
    for savgol in (7, 11, 15, 21):
        cfg = deepcopy(config)
        cfg.setdefault("preprocessing", {})["savgol_window"] = savgol
        trips_alt = preprocess_trips(raw, cfg)
        tab = _physics_table(trips_alt, nominal)
        pre_rows.append(
            {
                "setting": "savgol_window",
                "value": savgol,
                "physics_mae_kwh": float(tab["absolute_error_kwh"].mean()),
            }
        )
    for alt_w in (11, 21, 31):
        cfg = deepcopy(config)
        cfg.setdefault("preprocessing", {})["altitude_savgol_window"] = alt_w
        trips_alt = preprocess_trips(raw, cfg)
        tab = _physics_table(trips_alt, nominal)
        pre_rows.append(
            {
                "setting": "altitude_savgol_window",
                "value": alt_w,
                "physics_mae_kwh": float(tab["absolute_error_kwh"].mean()),
            }
        )
    for spat in (20.0, 40.0, 80.0):
        cfg = deepcopy(config)
        cfg.setdefault("preprocessing", {})["grade_spatial_window_m"] = spat
        trips_alt = preprocess_trips(raw, cfg)
        tab = _physics_table(trips_alt, nominal)
        pre_rows.append(
            {
                "setting": "grade_spatial_window_m",
                "value": spat,
                "physics_mae_kwh": float(tab["absolute_error_kwh"].mean()),
            }
        )
    pd.DataFrame(pre_rows).to_csv(out_dir / "preprocessing.csv", index=False)
    write_manifest(out_dir / "manifest.json", {"stage": "sensitivity", "profile": profile, "nominal_capacity_kwh": 6.0})
    print(f"Wrote sensitivity outputs to {out_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical/battery/preprocessing sensitivity.")
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    run_sensitivity(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
