"""Conference-paper CSV tables. Four tables only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.bootstrap import bootstrap_mean_ci, trip_absolute_errors
from evaluation.metrics import summarize_energy_table
from evaluation.names import PAPER_METHODS, SCARCITY_METHODS, label


def table01_dataset(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return audit
    row = audit.iloc[0]
    n_windows = int(float(row.get("n_windows_total", 0) or 0))
    if n_windows <= 0:
        n_windows = int(
            float(row.get("windows_soc_event_0.1", 0) or 0)
            + float(row.get("windows_soc_event_0.2", 0) or 0)
            + float(row.get("windows_soc_event_0.5", 0) or 0)
            + float(row.get("windows_time_60s", 0) or 0)
            + float(row.get("windows_time_120s", 0) or 0)
            + float(row.get("windows_time_300s", 0) or 0)
            + float(row.get("windows_time_600s", 0) or 0)
            + float(row.get("windows_full_trip", 0) or 0)
        )
    dist_mean = row.get("distance_can_km_mean", np.nan)
    q = row.get("empirical_soc_q_pp", row.get("empirical_q", np.nan))
    return pd.DataFrame(
        [
            {
                "n_trips": int(row.get("n_trips", np.nan)),
                "n_T1": int(row.get("n_T1", np.nan)),
                "n_T2": int(row.get("n_T2", np.nan)),
                "n_T3": int(row.get("n_T3", np.nan)),
                "duration_s_min": float(row.get("duration_s_min", np.nan)),
                "duration_s_max": float(row.get("duration_s_max", np.nan)),
                "distance_km_min": float(row.get("distance_can_km_min", np.nan)),
                "distance_km_max": float(row.get("distance_can_km_max", np.nan)),
                "distance_km_mean": float(dist_mean) if pd.notna(dist_mean) else float("nan"),
                "empirical_soc_q_pp": float(q) if pd.notna(q) else float("nan"),
                "n_supervision_windows": n_windows,
            }
        ]
    )


def table02_main(pred: pd.DataFrame, bootstrap: pd.DataFrame | None = None) -> pd.DataFrame:
    pred = pred.loc[pred["method"].isin(PAPER_METHODS)]
    metrics = summarize_energy_table(pred)
    rows = []
    for method in PAPER_METHODS:
        if method not in metrics:
            continue
        vals = metrics[method]
        err = trip_absolute_errors(pred, method)
        ci = bootstrap_mean_ci(err.to_numpy(dtype=float))
        if bootstrap is not None and not bootstrap.empty:
            sub = bootstrap.loc[bootstrap["method"] == method]
            if "metric" in sub.columns:
                sub = sub.loc[sub["metric"] == "mae_kwh"]
            if not sub.empty:
                ci["ci_low"] = float(sub["ci_low"].iloc[0])
                ci["ci_high"] = float(sub["ci_high"].iloc[0])
        rows.append(
            {
                "Method": label(method),
                "MAE_kWh": float(err.mean()) if len(err) else vals["mae_kwh"],
                "RMSE_kWh": vals["rmse_kwh"],
                "MAPE_pct": vals["mape_pct"],
                "WAPE_pct": vals["wape_pct"],
                "Bias_kWh": vals["bias_kwh"],
                "R2": vals["r2"],
                "MAE_CI_low": ci["ci_low"],
                "MAE_CI_high": ci["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def complete_scarcity_frame(raw: pd.DataFrame, n_rep_full: int = 3) -> pd.DataFrame:
    if raw.empty or "seed" not in raw.columns:
        return raw
    keep_idx: list[int] = []
    n_trips = int(raw["trip_id"].nunique()) if "trip_id" in raw.columns else 0
    for (_, n_train, _), g in raw.groupby(["method", "n_train", "seed"], dropna=False):
        n = int(n_train)
        expected_repeats = 1 if n >= 16 or n == 0 else n_rep_full
        expected = max(n_trips, 1) * expected_repeats
        if len(g) >= expected:
            keep_idx.extend(g.index.tolist())
    return raw.loc[sorted(set(keep_idx))].copy()


def table03_scarcity(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    raw = complete_scarcity_frame(raw)
    rows = []
    n_values = sorted(int(n) for n in raw["n_train"].unique() if int(n) > 0)
    for n in n_values:
        for method in SCARCITY_METHODS:
            g = raw.loc[(raw["method"] == method) & (raw["n_train"] == n)]
            if g.empty:
                continue
            run = g.groupby(["seed", "repeat"], dropna=False)["absolute_error_kwh"].mean()
            rows.append(
                {
                    "n_train": int(n),
                    "method": label(method),
                    "MAE_kWh": float(run.mean()) if len(run) else float(g["absolute_error_kwh"].mean()),
                    "uncertainty": float(run.std(ddof=1)) if len(run) > 1 else 0.0,
                    "number_of_runs": int(len(run)),
                }
            )
    return pd.DataFrame(rows)


def table04_feasibility(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    method_col = "method" if "method" in raw.columns else "Method"
    reserve_col = "reserve_soc_pct" if "reserve_soc_pct" in raw.columns else "reserve_SoC_pct"
    false_col = "false_safe" if "false_safe" in raw.columns else "false_safe_fraction"
    cons_col = "overly_conservative" if "overly_conservative" in raw.columns else "overly_conservative_fraction"
    frame = raw.copy()
    frame["_m"] = frame[method_col].astype(str).str.lower()
    keep = frame.loc[frame["_m"].isin(["physics", "pinn"])]
    rows = []
    for (method, reserve), g in keep.groupby(["_m", reserve_col]):
        rows.append(
            {
                "reserve_soc_pct": float(reserve),
                "method": label(method),
                "false_safe_pct": 100.0 * float(g[false_col].mean()),
                "overly_conservative_pct": 100.0 * float(g[cons_col].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        order = {"Physics": 0, "PINN": 1}
        out["_o"] = out["method"].map(lambda m: order.get(m, 9))
        out = out.sort_values(["reserve_soc_pct", "_o"]).drop(columns="_o")
    return out
