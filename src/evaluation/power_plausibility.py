"""Physical power plausibility diagnostics. Does not retune the frozen protocol.

Renault Twizy 13 kW is a WHEEL/motor bound, not a battery-power cap.
Battery traction corresponding to 13 kW wheel:
    P_batt = P_wheel / eta_drive + P_aux
Regen battery corresponding to -8 kW wheel:
    P_batt = eta_regen * P_wheel + P_aux
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physics.parameters import VehicleParameters
from physics.vehicle_model import predict_trip_physics


def battery_caps(params: VehicleParameters) -> dict[str, float]:
    wheel_hi = abs(float(params.maximum_wheel_power_kw))
    wheel_lo = -abs(float(params.maximum_regen_wheel_power_kw))
    batt_hi = wheel_hi / float(params.eta_drive) + float(params.auxiliary_power_kw)
    batt_lo = float(params.eta_regen) * wheel_lo + float(params.auxiliary_power_kw)
    return {
        "wheel_traction_kw": wheel_hi,
        "wheel_regen_kw": wheel_lo,
        "battery_traction_kw": batt_hi,
        "battery_regen_kw": batt_lo,
        "eta_drive": float(params.eta_drive),
        "eta_regen": float(params.eta_regen),
        "auxiliary_power_kw": float(params.auxiliary_power_kw),
    }


def series_stats(p: np.ndarray) -> dict[str, float]:
    x = np.asarray(p, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ("n", "min", "max", "p95_abs", "p99_abs", "mean", "mean_abs")}
    return {
        "n": float(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p95_abs": float(np.quantile(np.abs(x), 0.95)),
        "p99_abs": float(np.quantile(np.abs(x), 0.99)),
        "mean": float(np.mean(x)),
        "mean_abs": float(np.mean(np.abs(x))),
    }


def physics_arrays(trip, params: VehicleParameters, apply_bounds: bool) -> dict[str, np.ndarray | float]:
    result = predict_trip_physics(
        trip.frame["speed_mps"].to_numpy(dtype=float),
        trip.frame["acc_mps2"].to_numpy(dtype=float),
        trip.frame["theta_rad"].to_numpy(dtype=float),
        trip.frame["dt_s"].to_numpy(dtype=float),
        params,
        apply_bounds=apply_bounds,
    )
    return {
        "p_wheel_kw": result.p_wheel_kw,
        "p_battery_kw": result.p_battery_kw,
        "energy_kwh": float(result.energy_kwh),
        "n_bound_applied": int(result.n_bound_applied),
    }


def row_for_power(
    *,
    trip_id: str,
    trajectory: str,
    method: str,
    seed: int,
    p_batt: np.ndarray,
    p_wheel: np.ndarray | None,
    p_phy: np.ndarray | None,
    delta_p: np.ndarray | None,
    sat: float | None,
    caps: dict[str, float],
    apply_bounds: bool | None = None,
) -> dict[str, Any]:
    p_batt = np.asarray(p_batt, dtype=float)
    stats = series_stats(p_batt)
    n = max(int(stats["n"]), 1)
    wheel = np.asarray(p_wheel, dtype=float) if p_wheel is not None else None
    phy = np.asarray(p_phy, dtype=float) if p_phy is not None else None
    delta = np.asarray(delta_p, dtype=float) if delta_p is not None else None
    row: dict[str, Any] = {
        "trip_id": trip_id,
        "trajectory": trajectory,
        "method": method,
        "seed": int(seed),
        "apply_wheel_bounds": apply_bounds,
        "p_batt_min_kw": stats["min"],
        "p_batt_max_kw": stats["max"],
        "p_batt_p95_abs_kw": stats["p95_abs"],
        "p_batt_p99_abs_kw": stats["p99_abs"],
        "frac_batt_above_traction": float(np.mean(p_batt > caps["battery_traction_kw"])) if p_batt.size else float("nan"),
        "frac_batt_below_regen": float(np.mean(p_batt < caps["battery_regen_kw"])) if p_batt.size else float("nan"),
        "n_samples": n,
        "mean_abs_delta_p_kw": float(np.mean(np.abs(delta))) if delta is not None and delta.size else float("nan"),
        "max_abs_delta_p_kw": float(np.max(np.abs(delta))) if delta is not None and delta.size else float("nan"),
        "residual_saturation": float(sat) if sat is not None else float("nan"),
        "p_phy_max_kw": float(np.max(phy)) if phy is not None and phy.size else float("nan"),
        "p_phy_min_kw": float(np.min(phy)) if phy is not None and phy.size else float("nan"),
        "wheel_traction_kw": caps["wheel_traction_kw"],
        "wheel_regen_kw": caps["wheel_regen_kw"],
        "battery_traction_kw": caps["battery_traction_kw"],
        "battery_regen_kw": caps["battery_regen_kw"],
    }
    if wheel is not None and wheel.size:
        row["p_wheel_min_kw"] = float(np.min(wheel))
        row["p_wheel_max_kw"] = float(np.max(wheel))
        row["frac_wheel_above_traction"] = float(np.mean(wheel > caps["wheel_traction_kw"]))
        row["frac_wheel_below_regen"] = float(np.mean(wheel < caps["wheel_regen_kw"]))
    return row


def summarize_method(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = []
    for method, sub in frame.groupby("method"):
        def _mean(col: str) -> float:
            return float(sub[col].mean()) if col in sub.columns else float("nan")

        def _max(col: str) -> float:
            return float(sub[col].max()) if col in sub.columns else float("nan")

        rows.append(
            {
                "method": method,
                "peak_battery_power_kw": _max("p_batt_max_kw"),
                "min_battery_power_kw": float(sub["p_batt_min_kw"].min()) if "p_batt_min_kw" in sub.columns else float("nan"),
                "p99_abs_battery_kw": _mean("p_batt_p99_abs_kw"),
                "frac_exceeding_battery_traction": _mean("frac_batt_above_traction"),
                "frac_exceeding_battery_regen": _mean("frac_batt_below_regen"),
                "mean_abs_delta_p_kw": _mean("mean_abs_delta_p_kw"),
                "max_abs_delta_p_kw": _max("max_abs_delta_p_kw"),
                "residual_saturation": _mean("residual_saturation"),
                "max_physics_power_kw": _max("p_phy_max_kw"),
                "n_trips": int(sub["trip_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def load_npz_power(npz_dir: Path, trip_id: str, method: str, seed: int) -> dict[str, np.ndarray] | None:
    path = npz_dir / f"{trip_id}__{method}__seed{seed}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}
