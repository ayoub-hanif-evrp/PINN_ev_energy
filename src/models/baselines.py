"""LOTO physics-only energy baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from data.preprocessing import ProcessedTrip
from physics.parameters import VehicleParameters
from physics.vehicle_model import observed_energy_kwh, predict_trip_physics


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
