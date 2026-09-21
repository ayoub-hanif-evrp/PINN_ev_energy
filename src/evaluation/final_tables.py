"""Build paper-ready CSV tables under results/tables/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.bootstrap import bootstrap_mean_ci, paired_difference_ci, trip_absolute_errors
from evaluation.names import ABLATION_ORDER, METHOD_LABELS, METHOD_ORDER, label
from evaluation.statistics import PRIMARY_PAIRS, wilcoxon_pairs
from evaluation.metrics import summarize_energy_table


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def table01_dataset(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return audit
    row = audit.iloc[0]
    out = pd.DataFrame(
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
                "distance_km_mean": float(row["distance_can_km_mean"]) if "distance_can_km_mean" in row.index and pd.notna(row.get("distance_can_km_mean")) else float("nan"),
                "soc_start_mean_pct": float(row.get("soc_start_mean", np.nan)),
                "soc_end_mean_pct": float(row.get("soc_end_mean", np.nan)),
                "empirical_soc_q_pp": float(row.get("empirical_q", np.nan)),
                "n_windows_total": int(row.get("n_windows_total", np.nan)),
                "n_windows_full_trip": int(row.get("windows_full_trip", 0)),
                "n_windows_soc_event": int(
                    float(row.get("windows_soc_event_0.1", 0))
                    + float(row.get("windows_soc_event_0.2", 0))
                    + float(row.get("windows_soc_event_0.5", 0))
                ),
                "n_windows_fixed_time": int(
                    float(row.get("windows_time_60s", 0))
                    + float(row.get("windows_time_120s", 0))
                    + float(row.get("windows_time_300s", 0))
                    + float(row.get("windows_time_600s", 0))
                ),
            }
        ]
    )
    # Prefer CAN-speed mean distance if present in a later audit.
    if "distance_can_km_mean" not in audit.columns and "distance_can_km_min" in audit.columns:
        out["distance_km_mean_note"] = "mean CAN distance not stored in this audit CSV; min/max are CAN-speed distances"
    return out


def table02_main(pred: pd.DataFrame, bootstrap: pd.DataFrame | None = None) -> pd.DataFrame:
    """MAE uses trip-level mean absolute error across seeds, not an ensemble of predictions."""
    metrics = summarize_energy_table(pred)
    rows = []
    for method in METHOD_ORDER:
        if method not in metrics:
            continue
        vals = metrics[method]
        err = trip_absolute_errors(pred, method)
        ci = bootstrap_mean_ci(err.to_numpy(dtype=float))
        if bootstrap is not None and not bootstrap.empty:
            sub = bootstrap.loc[bootstrap["method"] == method]
            if not sub.empty:
                ci["ci_low"] = float(sub["ci_low"].iloc[0])
                ci["ci_high"] = float(sub["ci_high"].iloc[0])
        rows.append(
            {
                "Method": label(method),
                "MAE_kWh": float(err.mean()) if len(err) else vals["mae_kwh"],
                "MAE_95CI_low": ci["ci_low"],
                "MAE_95CI_high": ci["ci_high"],
                "RMSE_kWh": vals["rmse_kwh"],
                "MAPE_pct": vals["mape_pct"],
                "WAPE_pct": vals["wape_pct"],
                "Bias_kWh": vals["bias_kwh"],
                "R2": vals["r2"],
            }
        )
    return pd.DataFrame(rows)


def table03_paired(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wilc = wilcoxon_pairs(pred)
    for a, b in PRIMARY_PAIRS:
        ea = trip_absolute_errors(pred, a)
        eb = trip_absolute_errors(pred, b)
        stats = paired_difference_ci(ea, eb)
        p_value = np.nan
        if not wilc.empty:
            hit = wilc.loc[(wilc["method_a"] == a) & (wilc["method_b"] == b)]
            if not hit.empty:
                p_value = float(hit["p_value"].iloc[0])
        rows.append(
            {
                "comparison": f"{label(a)} vs {label(b)}",
                "mean_paired_AE_difference_kWh": stats["mean"],
                "CI95_low": stats["ci_low"],
                "CI95_high": stats["ci_high"],
                "n_trips": int(stats.get("n_trips", stats.get("n", np.nan))),
                "wilcoxon_p_secondary": p_value,
                "note": "Negative difference means the first method has lower trip absolute error. Bootstrap intervals are descriptive; they are not called statistically significant.",
            }
        )
    return pd.DataFrame(rows)


def table04_ablation(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["physics"] + ABLATION_ORDER
    for method in order:
        err = trip_absolute_errors(pred, method)
        if err.empty:
            continue
        stats = bootstrap_mean_ci(err.to_numpy(dtype=float))
        rows.append(
            {
                "Method": label(method),
                "MAE_kWh": stats["mean"],
                "MAE_95CI_low": stats["ci_low"],
                "MAE_95CI_high": stats["ci_high"],
                "n_trips": int(stats["n"]),
            }
        )
    return pd.DataFrame(rows)


def complete_scarcity_frame(raw: pd.DataFrame, n_rep_full: int = 3) -> pd.DataFrame:
    """Drop incomplete seeds so a running experiment cannot pollute the paper table."""
    if raw.empty or "seed" not in raw.columns:
        return raw
    keep_idx: list[int] = []
    n_trips = int(raw["trip_id"].nunique()) if "trip_id" in raw.columns else 0
    for (method, n_train, seed), g in raw.groupby(["method", "n_train", "seed"], dropna=False):
        n = int(n_train)
        expected_repeats = 1 if n >= 16 or n == 0 else n_rep_full
        expected = max(n_trips, 1) * expected_repeats
        if len(g) >= expected:
            keep_idx.extend(g.index.tolist())
    return raw.loc[sorted(set(keep_idx))].copy()


def table05_scarcity(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    raw = complete_scarcity_frame(raw)
    rows = []
    methods = [m for m in ("physics", "elasticnet", "weak_mlp", "pinn") if m in set(raw["method"])]
    for method in methods:
        sub = raw.loc[raw["method"] == method]
        ns = sorted(sub["n_train"].unique())
        for n in ns:
            g = sub.loc[sub["n_train"] == n]
            run = g.groupby(["seed", "repeat"], dropna=False)["absolute_error_kwh"].mean()
            trip_mae = g.groupby("trip_id")["absolute_error_kwh"].mean()
            ci = bootstrap_mean_ci(trip_mae.to_numpy(dtype=float))
            rows.append(
                {
                    "n_train": int(n),
                    "Method": label(method),
                    "mean_MAE_kWh": float(run.mean()) if len(run) else float(g["absolute_error_kwh"].mean()),
                    "std_MAE_across_runs": float(run.std(ddof=1)) if len(run) > 1 else 0.0,
                    "trip_MAE_CI95_low": ci["ci_low"],
                    "trip_MAE_CI95_high": ci["ci_high"],
                    "n_subsets": int(g["repeat"].nunique()) if "repeat" in g.columns else 1,
                    "n_seeds": int(g["seed"].nunique()) if "seed" in g.columns else 1,
                    "n_heldout_trips": int(g["trip_id"].nunique()),
                    "n_evaluations": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def table06_cross_trajectory(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    held_col = "heldout_trajectory" if "heldout_trajectory" in raw.columns else "trajectory"
    trip_level = raw.groupby(["trip_id", "method", held_col], as_index=False).agg(
        observed_energy_kwh=("observed_energy_kwh", "mean"),
        predicted_energy_kwh=("predicted_energy_kwh", "mean"),
        absolute_error_kwh=("absolute_error_kwh", "mean"),
        signed_error_kwh=("signed_error_kwh", "mean"),
    )
    rows = []
    for held, sub in trip_level.groupby(held_col):
        for method in ("physics", "elasticnet", "weak_mlp", "pinn"):
            g = sub.loc[sub["method"] == method]
            if g.empty:
                continue
            err = g["predicted_energy_kwh"] - g["observed_energy_kwh"]
            rows.append(
                {
                    "held_out_trajectory": held,
                    "Method": label(method),
                    "n_trips": int(g["trip_id"].nunique()),
                    "MAE_kWh": float(g["absolute_error_kwh"].mean()),
                    "RMSE_kWh": float(np.sqrt(np.mean(np.square(err)))),
                    "Bias_kWh": float(g["signed_error_kwh"].mean()),
                }
            )
    return pd.DataFrame(rows)


def table07_power(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    keep = summary.loc[summary["method"].isin(["physics", "pinn"])].copy()
    if keep.empty:
        keep = summary.copy()
    keep["Method"] = keep["method"].map(label)
    cols = [
        "Method",
        "peak_battery_power_kw",
        "p99_abs_battery_kw",
        "frac_exceeding_battery_traction",
        "frac_exceeding_battery_regen",
        "mean_abs_delta_p_kw",
        "max_abs_delta_p_kw",
        "residual_saturation",
    ]
    present = [c for c in cols if c in keep.columns]
    return keep[present]


def table08_sensitivity(
    battery: pd.DataFrame,
    physical: pd.DataFrame,
    preprocessing: pd.DataFrame,
    bounds: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    if not battery.empty:
        for _, r in battery.iterrows():
            rows.append(
                {
                    "family": "battery_capacity",
                    "setting": f"{r.get('value')} kWh",
                    "physics_MAE_kWh": r.get("physics_mae_kwh"),
                    "note": r.get("note", "E_obs scales with assumed usable capacity"),
                }
            )
    if not physical.empty:
        nom = physical.loc[physical["delta_mae_vs_nominal"].abs() < 1e-12]
        for param, sub in physical.groupby("parameter"):
            lo = sub.loc[sub["value"].idxmin()]
            hi = sub.loc[sub["value"].idxmax()]
            rows.append(
                {
                    "family": "physical_parameter",
                    "setting": param,
                    "physics_MAE_kWh": float(sub.loc[sub["value"] == sub["nominal"].iloc[0], "physics_mae_kwh"].iloc[0])
                    if (sub["value"] == sub["nominal"]).any()
                    else np.nan,
                    "note": f"MAE range {float(lo['physics_mae_kwh']):.3f}–{float(hi['physics_mae_kwh']):.3f} kWh over [{lo['value']}, {hi['value']}]; modelling assumption, not a HELECAR measurement",
                }
            )
    if not preprocessing.empty:
        if "physics_mae_kwh" in preprocessing.columns:
            rows.append(
                {
                    "family": "preprocessing",
                    "setting": "speed/altitude smoother windows",
                    "physics_MAE_kWh": float(preprocessing["physics_mae_kwh"].mean()),
                    "note": f"physics MAE min={float(preprocessing['physics_mae_kwh'].min()):.3f} max={float(preprocessing['physics_mae_kwh'].max()):.3f} kWh",
                }
            )
    if not bounds.empty:
        for _, r in bounds.iterrows():
            rows.append(
                {
                    "family": "physical_power_bounds",
                    "setting": f"{r.get('variant')} / {r.get('method')}",
                    "physics_MAE_kWh": r.get("mae_kwh"),
                    "note": "Wheel-power clamp sensitivity; main protocol remains unbounded.",
                }
            )
    return pd.DataFrame(rows)


def table09_feasibility(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    if {"false_safe", "overly_conservative"}.issubset(raw.columns) and "trip_id" not in raw.columns:
        out = raw.copy()
        if "method" in out.columns:
            out["Method"] = out["method"].map(lambda m: label(m) if m in METHOD_LABELS else m)
        return out
    rows = []
    for (method, reserve), g in raw.groupby(["method", "reserve_soc_pct"]):
        rows.append(
            {
                "Method": label(method),
                "reserve_SoC_pct": float(reserve),
                "false_safe_fraction": float(g["false_safe"].mean()),
                "overly_conservative_fraction": float(g["overly_conservative"].mean()),
                "predicted_feasible_fraction": float(g["model_feasible"].mean()),
                "observed_feasible_fraction": float(g["observed_feasible"].mean()),
                "n_trips": int(g["trip_id"].nunique()) if "trip_id" in g.columns else int(len(g)),
            }
        )
    order = {name: i for i, name in enumerate(["Constant", "Physics", "PINN"])}
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["_ord"] = frame["Method"].map(lambda m: order.get(m, 99))
        frame = frame.sort_values(["reserve_SoC_pct", "_ord"]).drop(columns="_ord")
    return frame
