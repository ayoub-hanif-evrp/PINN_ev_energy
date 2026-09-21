"""LOTO baselines: aggregate Wh/km and physics-only energy."""

from __future__ import annotations

from typing import Any

import numpy as np

from data.features import trip_level_features
from data.preprocessing import ProcessedTrip
from physics.parameters import VehicleParameters
from physics.vehicle_model import observed_energy_kwh, predict_trip_physics


def constant_wh_per_km(train_trips: list[ProcessedTrip], battery_capacity_kwh: float) -> float:
    """Training-only aggregate consumption: Σ E_Wh / Σ distance_km."""
    energy_wh = 0.0
    distance_km = 0.0
    for trip in train_trips:
        feats = trip_level_features(trip, battery_capacity_kwh=battery_capacity_kwh)
        energy_wh += float(feats["e_obs_kwh"]) * 1000.0
        distance_km += float(feats["distance_km"])
    if distance_km <= 1e-12:
        return 0.0
    return energy_wh / distance_km


def constant_consumption_predict(mean_wh_per_km: float, distance_km: float) -> float:
    return float(mean_wh_per_km) * float(distance_km) / 1000.0


def physics_only_predict(
    trip: ProcessedTrip,
    params: VehicleParameters,
    apply_bounds: bool = False,
    bound_sharpness: float = 8.0,
) -> dict[str, Any]:
    df = trip.frame
    result = predict_trip_physics(
        df["speed_mps"].to_numpy(dtype=float),
        df["acc_mps2"].to_numpy(dtype=float),
        df["theta_rad"].to_numpy(dtype=float),
        df["dt_s"].to_numpy(dtype=float),
        params,
        apply_bounds=apply_bounds,
        bound_sharpness=bound_sharpness,
    )
    soc = df["soc"].to_numpy(dtype=float) if "soc" in df.columns else np.array([])
    e_obs = (
        observed_energy_kwh(float(soc[0]), float(soc[-1]), params.battery_capacity_kwh)
        if soc.size
        else float("nan")
    )
    return {
        "p_battery_kw": result.p_battery_kw,
        "p_wheel_kw": result.p_wheel_kw,
        "energy_kwh": result.energy_kwh,
        "prefix_kwh": result.prefix_kwh,
        "e_obs_kwh": e_obs,
        "n_bound_applied": result.n_bound_applied,
    }
