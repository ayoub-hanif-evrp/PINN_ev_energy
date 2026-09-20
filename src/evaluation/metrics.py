"""Evaluation helpers used from Phase 4 onward.

SoC reconstruction uses ONLY the held-out trip's initial measured SoC plus
integrated predicted energy. Later SoC samples are for metrics only.
"""

from __future__ import annotations

import numpy as np

from units import energy_prefix_kwh


def reconstruct_soc(
    soc_initial: float,
    power_kw: np.ndarray,
    dt_s: np.ndarray,
    battery_capacity_kwh: float,
) -> dict[str, np.ndarray | float]:
    """
    SOC_hat[0] = SOC_observed[0]
    SOC_hat[t] = SOC_hat[0] - 100 / E_battery * cum_energy[t]

    cum_energy[t] is energy from the start through the interval ending at sample t
    (prefix[t], with prefix[0] = 0). Later measured SoC values are not used.
    """
    if battery_capacity_kwh <= 0:
        raise ValueError("battery_capacity_kwh must be positive")
    prefix = energy_prefix_kwh(power_kw, dt_s)
    soc_hat = float(soc_initial) - (100.0 / float(battery_capacity_kwh)) * prefix
    soc_end_hat = float(soc_hat[-1]) if soc_hat.size else float(soc_initial)
    return {
        "soc_hat": soc_hat,
        "soc_end_hat": soc_end_hat,
        "prefix_kwh": prefix,
        "soc_initial_used": float(soc_initial),
    }


def soc_reconstruction_metrics(
    soc_observed: np.ndarray,
    soc_hat: np.ndarray,
    q: float,
    soc_end_hat: float | None = None,
    tolerance_half_bin: bool = True,
) -> dict[str, float]:
    y = np.asarray(soc_observed, dtype=float)
    yhat = np.asarray(soc_hat, dtype=float)
    n = min(len(y), len(yhat))
    y = y[:n]
    yhat = yhat[:n]
    err = yhat - y
    end_obs = float(y[-1]) if n else np.nan
    end_hat = float(soc_end_hat) if soc_end_hat is not None else (float(yhat[-1]) if n else np.nan)
    tol = 0.5 * float(q) if tolerance_half_bin and np.isfinite(q) else np.nan
    within = np.mean(np.abs(yhat - y) <= tol) if np.isfinite(tol) else np.nan
    return {
        "soc_mae_pp": float(np.mean(np.abs(err))) if n else np.nan,
        "soc_rmse_pp": float(np.sqrt(np.mean(err**2))) if n else np.nan,
        "soc_end_error_pp": float(end_hat - end_obs) if n else np.nan,
        "soc_within_half_q_fraction": float(within),
        "soc_tolerance_pp": float(tol) if np.isfinite(tol) else np.nan,
    }
