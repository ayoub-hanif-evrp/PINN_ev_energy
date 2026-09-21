"""Reproducible trip-level paired bootstrap. Trips are the statistical unit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def trip_absolute_errors(frame: pd.DataFrame, method: str, seed_agg: str = "mean") -> pd.Series:
    """One absolute error per trip. Neural multi-seed rows are aggregated per trip."""
    sub = frame.loc[frame["method"] == method, ["trip_id", "absolute_error_kwh", "seed"]].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    if seed_agg == "mean":
        return sub.groupby("trip_id")["absolute_error_kwh"].mean()
    raise ValueError(f"Unknown seed aggregation {seed_agg}")


def mae_from_trip_errors(errors: pd.Series | np.ndarray) -> float:
    x = np.asarray(errors, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def bootstrap_mean_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 20260921,
    alpha: float = 0.05,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0.0}
    draws = rng.choice(x, size=(int(n_resamples), x.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean": float(np.mean(x)), "ci_low": float(lo), "ci_high": float(hi), "n": float(x.size)}


def paired_difference_ci(
    errors_a: pd.Series,
    errors_b: pd.Series,
    n_resamples: int = 10_000,
    seed: int = 20260921,
) -> dict[str, float]:
    """errors_a - errors_b aligned on trip_id. Negative mean => A has lower AE."""
    aligned = pd.concat([errors_a.rename("a"), errors_b.rename("b")], axis=1, join="inner").dropna()
    diff = (aligned["a"] - aligned["b"]).to_numpy(dtype=float)
    stats = bootstrap_mean_ci(diff, n_resamples=n_resamples, seed=seed)
    stats["n_trips"] = float(len(aligned))
    return stats


def per_seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, seed), sub in frame.groupby(["method", "seed"]):
        y = sub["observed_energy_kwh"].to_numpy(dtype=float)
        yhat = sub["predicted_energy_kwh"].to_numpy(dtype=float)
        err = yhat - y
        abs_obs = np.abs(y)
        mape_ok = abs_obs > 1e-12
        obs_sum = float(np.sum(abs_obs))
        denom = float(np.sum((y - np.mean(y)) ** 2))
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "n": int(len(sub)),
                "mae_kwh": float(np.mean(np.abs(err))),
                "rmse_kwh": float(np.sqrt(np.mean(err**2))),
                "mape_pct": float(np.mean(np.abs(err[mape_ok] / y[mape_ok])) * 100.0) if mape_ok.any() else float("nan"),
                "wape_pct": 100.0 * float(np.sum(np.abs(err))) / obs_sum if obs_sum > 1e-12 else float("nan"),
                "bias_kwh": float(np.mean(err)),
                "r2": 1.0 - float(np.sum(err**2) / denom) if denom > 1e-12 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def seed_mean_std(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return per_seed
    metrics = ["mae_kwh", "rmse_kwh", "mape_pct", "wape_pct", "bias_kwh", "r2"]
    rows = []
    for method, sub in per_seed.groupby("method"):
        row: dict[str, Any] = {"method": method, "n_seeds": int(sub["seed"].nunique())}
        for m in metrics:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_std"] = float(sub[m].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
