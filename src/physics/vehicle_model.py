"""Longitudinal EV energy model.

Positive battery power means battery discharge.
WindSpeedAv is NEVER treated as a headwind: analysed HELECAR-D has no wind direction.
Aerodynamic drag uses vehicle speed only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.parameters import VehicleParameters
from units import WATTS_PER_KW, energy_kwh_from_power, energy_prefix_kwh, window_energy_from_prefix


def _softplus(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    hi = z > 20.0
    lo = z < -20.0
    mid = ~hi & ~lo
    out[hi] = z[hi]
    out[lo] = np.exp(z[lo])
    out[mid] = np.log1p(np.exp(z[mid]))
    return out


def soft_clamp(x: np.ndarray, lower: float, upper: float, sharpness: float = 8.0) -> np.ndarray:
    """Smooth clamp that is approximately the identity on (lower, upper).

    y = x - softplus(k(x-hi))/k + softplus(k(lo-x))/k
    Large k recovers a hard clip. Interior points are left essentially unchanged.
    """
    k = max(float(sharpness), 1e-6)
    x = np.asarray(x, dtype=float)
    if upper <= lower:
        return np.full_like(x, 0.5 * (lower + upper))
    return x - _softplus(k * (x - upper)) / k + _softplus(k * (lower - x)) / k


@dataclass
class PhysicsResult:
    p_wheel_kw: np.ndarray
    p_battery_kw: np.ndarray
    f_total_n: np.ndarray
    f_inertia_n: np.ndarray
    f_roll_n: np.ndarray
    f_grade_n: np.ndarray
    f_aero_n: np.ndarray
    energy_kwh: float
    prefix_kwh: np.ndarray
    n_bound_applied: int = 0

    def window_energy(self, start: int, end: int) -> float:
        return window_energy_from_prefix(self.prefix_kwh, start, end)


def longitudinal_forces(
    speed_mps: np.ndarray,
    acc_mps2: np.ndarray,
    theta_rad: np.ndarray,
    params: VehicleParameters,
) -> dict[str, np.ndarray]:
    v = np.asarray(speed_mps, dtype=float)
    a = np.asarray(acc_mps2, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    m = params.mass_kg
    g = params.g
    f_inertia = m * a
    f_roll = m * g * params.Crr * np.cos(theta)
    f_grade = m * g * np.sin(theta)
    f_aero = 0.5 * params.rho_air * params.cda_m2 * np.square(v)
    f_total = f_inertia + f_roll + f_grade + f_aero
    return {
        "f_inertia_n": f_inertia,
        "f_roll_n": f_roll,
        "f_grade_n": f_grade,
        "f_aero_n": f_aero,
        "f_total_n": f_total,
    }


def wheel_power_kw(
    speed_mps: np.ndarray,
    acc_mps2: np.ndarray,
    theta_rad: np.ndarray,
    params: VehicleParameters,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    forces = longitudinal_forces(speed_mps, acc_mps2, theta_rad, params)
    p_wheel_w = forces["f_total_n"] * np.asarray(speed_mps, dtype=float)
    return p_wheel_w / WATTS_PER_KW, forces


def battery_power_from_wheel(
    p_wheel_kw: np.ndarray,
    params: VehicleParameters,
    apply_bounds: bool = False,
    bound_sharpness: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    p_w = np.asarray(p_wheel_kw, dtype=float)
    n_bound = 0
    if apply_bounds:
        lo = -abs(params.maximum_regen_wheel_power_kw)
        hi = abs(params.maximum_wheel_power_kw)
        n_bound = int(np.sum((p_w < lo) | (p_w > hi)))
        p_w = soft_clamp(p_w, lo, hi, bound_sharpness)
    p_batt = np.empty_like(p_w)
    traction = p_w >= 0.0
    p_batt[traction] = p_w[traction] / params.eta_drive + params.auxiliary_power_kw
    p_batt[~traction] = params.eta_regen * p_w[~traction] + params.auxiliary_power_kw
    return p_batt, p_w, n_bound


def predict_battery_power(
    speed_mps: np.ndarray,
    acc_mps2: np.ndarray,
    theta_rad: np.ndarray,
    params: VehicleParameters,
    apply_bounds: bool = False,
    bound_sharpness: float = 8.0,
    use_wind_as_headwind: bool = False,
) -> PhysicsResult:
    if use_wind_as_headwind:
        raise ValueError(
            "WindSpeedAv must not be treated as a headwind: analysed HELECAR-D "
            "does not provide wind direction. Use vehicle speed for F_aero."
        )
    p_wheel, forces = wheel_power_kw(speed_mps, acc_mps2, theta_rad, params)
    p_batt, p_wheel_used, n_bound = battery_power_from_wheel(
        p_wheel, params, apply_bounds=apply_bounds, bound_sharpness=bound_sharpness
    )
    return PhysicsResult(
        p_wheel_kw=p_wheel_used,
        p_battery_kw=p_batt,
        f_total_n=forces["f_total_n"],
        f_inertia_n=forces["f_inertia_n"],
        f_roll_n=forces["f_roll_n"],
        f_grade_n=forces["f_grade_n"],
        f_aero_n=forces["f_aero_n"],
        energy_kwh=0.0,
        prefix_kwh=np.array([0.0]),
        n_bound_applied=n_bound,
    )


def attach_energy(result: PhysicsResult, dt_s: np.ndarray) -> PhysicsResult:
    result.energy_kwh = energy_kwh_from_power(result.p_battery_kw, dt_s)
    result.prefix_kwh = energy_prefix_kwh(result.p_battery_kw, dt_s)
    return result


def predict_trip_physics(
    speed_mps: np.ndarray,
    acc_mps2: np.ndarray,
    theta_rad: np.ndarray,
    dt_s: np.ndarray,
    params: VehicleParameters,
    apply_bounds: bool = False,
    bound_sharpness: float = 8.0,
) -> PhysicsResult:
    result = predict_battery_power(
        speed_mps,
        acc_mps2,
        theta_rad,
        params,
        apply_bounds=apply_bounds,
        bound_sharpness=bound_sharpness,
        use_wind_as_headwind=False,
    )
    return attach_energy(result, dt_s)


def observed_energy_kwh(soc_start: float, soc_end: float, battery_capacity_kwh: float) -> float:
    """SoC-derived net energy. Positive means net discharge."""
    return float(battery_capacity_kwh) * (float(soc_start) - float(soc_end)) / 100.0


def delta_soc_percent(energy_kwh: float, battery_capacity_kwh: float) -> float:
    return 100.0 * float(energy_kwh) / float(battery_capacity_kwh)
