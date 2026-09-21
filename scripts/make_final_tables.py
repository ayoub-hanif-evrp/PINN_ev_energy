#!/usr/bin/env python
"""Write the four conference-paper CSV tables under results/tables/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from evaluation.final_tables import (  # noqa: E402
    table01_dataset,
    table02_main,
    table03_scarcity,
    table04_feasibility,
)
from paths import project_root, resolve_under_root  # noqa: E402


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fmt(x: float, nd: int = 3) -> str:
    return f"{float(x):.{nd}f}"


def write_experiment_summary(out_dir: Path, t1: pd.DataFrame, t2: pd.DataFrame, t3: pd.DataFrame, t4: pd.DataFrame) -> None:
    lines = [
        "# Experiment summary (paper-writing reference)",
        "",
        "Numbers below are taken from `results/tables/` after the final export.",
        "If a CSV changes, regenerate this file from that CSV.",
        "",
        "## Dataset",
        "",
    ]
    if not t1.empty:
        r = t1.iloc[0]
        lines += [
            f"- {int(r['n_trips'])} analysed HELECAR-D trips (T1={int(r['n_T1'])}, T2={int(r['n_T2'])}, T3={int(r['n_T3'])}).",
            f"- Duration {int(r['duration_s_min'])}–{int(r['duration_s_max'])} s.",
            f"- CAN-speed distance {_fmt(r['distance_km_min'], 2)}–{_fmt(r['distance_km_max'], 2)} km (mean {_fmt(r['distance_km_mean'], 2)} km).",
            f"- Empirical SoC resolution/quantization q ≈ {_fmt(r['empirical_soc_q_pp'], 2)} percentage points.",
            f"- Useful multi-window supervision windows: {int(r['n_supervision_windows'])}.",
            "",
        ]
    lines += [
        "## Main experiment (leave-one-trip-out)",
        "",
        "Trip-level energy metrics. MAE confidence intervals are trip-level bootstrap intervals.",
        "",
        "| Method | MAE_kWh | MAE_CI | RMSE_kWh | MAPE_pct | WAPE_pct | Bias_kWh | R2 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in t2.iterrows():
        lines.append(
            f"| {row['Method']} | {_fmt(row['MAE_kWh'])} | "
            f"[{_fmt(row['MAE_CI_low'])}, {_fmt(row['MAE_CI_high'])}] | "
            f"{_fmt(row['RMSE_kWh'])} | {_fmt(row['MAPE_pct'], 2)} | {_fmt(row['WAPE_pct'], 2)} | "
            f"{_fmt(row['Bias_kWh'])} | {_fmt(row['R2'], 3)} |"
        )
    lines += [
        "",
        "## Data scarcity",
        "",
        "Held-out energy MAE as a function of the number of training trips.",
        "Complete aggregation currently uses seed 0 and three subset repeats for n=3,5,8,12;",
        "n=16 is leave-one-trip-out with 16 training trips (one seed).",
        "",
        "| n_train | method | MAE_kWh | uncertainty | number_of_runs |",
        "|---:|---|---:|---:|---:|",
    ]
    for _, row in t3.iterrows():
        lines.append(
            f"| {int(row['n_train'])} | {row['method']} | {_fmt(row['MAE_kWh'])} | "
            f"{_fmt(row['uncertainty'])} | {int(row['number_of_runs'])} |"
        )
    lines += [
        "",
        "## Routing-oriented battery-feasibility assessment",
        "",
        "Reserve-margin decision errors. This is not an EVRP algorithm.",
        "",
        "| reserve_soc_pct | method | false_safe_pct | overly_conservative_pct |",
        "|---:|---|---:|---:|",
    ]
    for _, row in t4.iterrows():
        lines.append(
            f"| {int(row['reserve_soc_pct'])} | {row['method']} | "
            f"{_fmt(row['false_safe_pct'], 1)} | {_fmt(row['overly_conservative_pct'], 1)} |"
        )

    def _mae(name: str) -> float:
        hit = t2.loc[t2["Method"] == name]
        return float(hit["MAE_kWh"].iloc[0]) if not hit.empty else float("nan")

    physics, enet, weak, pinn = _mae("Physics"), _mae("ElasticNet"), _mae("WeakMLP"), _mae("PINN")
    lines += [
        "",
        "## Scientific interpretation",
        "",
        "What the results support:",
        "",
        f"- ElasticNet is strongest on ordinary full-data LOTO (MAE {_fmt(enet)} kWh).",
        f"- PINN improves substantially over the purely data-driven WeakMLP ({_fmt(pinn)} vs {_fmt(weak)} kWh).",
        f"- PINN improves substantially over the analytical physics model ({_fmt(pinn)} vs {_fmt(physics)} kWh).",
        "- Physics-informed learning is a useful inductive bias when instantaneous battery-power labels are unavailable, particularly when training trips are scarce.",
        "",
        "What the results do **not** support:",
        "",
        "- PINN is not globally best. ElasticNet remains better overall on full-data LOTO.",
        "- Instantaneous battery power has not been validated against ground truth; there are no direct power labels.",
        "- The routing section is only a battery-feasibility sensitivity, not a new routing algorithm.",
        "",
    ]
    if not t3.empty:
        n3 = t3.loc[t3["n_train"] == 3]
        def _n(method: str, n: int) -> float:
            hit = t3.loc[(t3["n_train"] == n) & (t3["method"] == method)]
            return float(hit["MAE_kWh"].iloc[0]) if not hit.empty else float("nan")

        lines += [
            "Data-scarcity detail:",
            "",
            f"- At 3 training trips, PINN MAE {_fmt(_n('PINN', 3))} kWh is lower than ElasticNet {_fmt(_n('ElasticNet', 3))} and WeakMLP {_fmt(_n('WeakMLP', 3))}.",
            f"- At 8 training trips, ElasticNet is unstable (MAE {_fmt(_n('ElasticNet', 8))} kWh).",
            f"- At 12 and 16 training trips, ElasticNet is again strongest ({_fmt(_n('ElasticNet', 12))} and {_fmt(_n('ElasticNet', 16))} kWh).",
            f"- PINN remains better than WeakMLP at every reported training size.",
            "- The main-table PINN MAE uses LOTO seeds 0/1/2. Scarcity n=16 uses seed 0 only, so those two PINN numbers need not match.",
            "",
        ]
    if not t4.empty:
        p20 = t4.loc[(t4["reserve_soc_pct"] == 20) & (t4["method"] == "Physics")]
        n20 = t4.loc[(t4["reserve_soc_pct"] == 20) & (t4["method"] == "PINN")]
        if not p20.empty and not n20.empty:
            lines += [
                "Feasibility detail:",
                "",
                f"- At a 20% reserve, Physics false-safe rate is {_fmt(p20['false_safe_pct'].iloc[0], 1)}% vs PINN {_fmt(n20['false_safe_pct'].iloc[0], 1)}%.",
                "- Physics is more false-safe because it systematically under-predicts trip energy.",
                "",
            ]
    lines += [
        "## Internal note (not a paper experiment)",
        "",
        "An internal full-trip-only ablation had a lower LOTO trip-energy MAE than the multi-window PINN.",
        "That finding is not used to retune window sizes or to introduce another research direction.",
        "Multi-window SoC energy supervision remains the method promised by the abstract.",
        "",
    ]
    (out_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Final conference-paper CSV tables.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    out = resolve_under_root(args.out, root) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    loto = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root) / profile
    sens = resolve_under_root(config.get("paths", {}).get("sensitivity_dir", "outputs/sensitivity"), root) / profile

    pred = _load(loto / "per_trip_predictions.csv")
    boot = _load(loto / "statistics" / "bootstrap_main_metrics.csv")
    audit = _load(resolve_under_root("outputs/data_audit", root) / "paper_dataset_summary.csv")
    trips = _load(resolve_under_root("outputs/data_audit", root) / "trip_characteristics.csv")
    if not audit.empty and not trips.empty:
        dist_col = next((c for c in ("distance_can_km", "distance_km", "s_can_km") if c in trips.columns), None)
        if dist_col is not None:
            audit = audit.copy()
            audit["distance_can_km_mean"] = float(trips[dist_col].mean())
    t1 = table01_dataset(audit)
    t1.to_csv(out / "table01_dataset.csv", index=False)

    t2 = table02_main(pred, boot if not boot.empty else None) if not pred.empty else pd.DataFrame()
    if not t2.empty:
        t2.to_csv(out / "table02_main_results.csv", index=False)

    scarcity = _load(resolve_under_root("outputs/scarcity", root) / profile / "scarcity_per_trip.csv")
    t3 = table03_scarcity(scarcity) if not scarcity.empty else pd.DataFrame()
    if not t3.empty:
        t3.to_csv(out / "table03_data_scarcity.csv", index=False)

    feas = _load(sens / "feasibility.csv")
    if feas.empty:
        feas = _load(sens / "feasibility_summary.csv")
    t4 = table04_feasibility(feas) if not feas.empty else pd.DataFrame()
    if not t4.empty:
        t4.to_csv(out / "table04_feasibility.csv", index=False)

    write_experiment_summary(resolve_under_root(args.out, root), t1, t2, t3, t4)
    print(f"Wrote tables under {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
