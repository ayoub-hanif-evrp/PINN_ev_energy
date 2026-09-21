#!/usr/bin/env python
"""Write paper-ready CSV tables under results/tables/."""

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
    table03_paired,
    table04_ablation,
    table05_scarcity,
    table06_cross_trajectory,
    table07_power,
    table08_sensitivity,
    table09_feasibility,
)
from paths import project_root, resolve_under_root  # noqa: E402


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Final paper CSV tables.")
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
    table01_dataset(audit).to_csv(out / "table01_dataset_summary.csv", index=False)
    if not pred.empty:
        table02_main(pred, boot if not boot.empty else None).to_csv(out / "table02_main_results.csv", index=False)
        table03_paired(pred).to_csv(out / "table03_paired_statistics.csv", index=False)
        table04_ablation(pred).to_csv(out / "table04_ablation.csv", index=False)

    scarcity = _load(resolve_under_root("outputs/scarcity", root) / profile / "scarcity_per_trip.csv")
    if not scarcity.empty:
        table05_scarcity(scarcity).to_csv(out / "table05_data_scarcity.csv", index=False)
    xtraj = _load(resolve_under_root("outputs/cross_trajectory", root) / profile / "cross_trajectory_per_trip.csv")
    if not xtraj.empty:
        table06_cross_trajectory(xtraj).to_csv(out / "table06_cross_trajectory.csv", index=False)

    power = _load(sens / "power_plausibility_summary.csv")
    if not power.empty:
        table07_power(power).to_csv(out / "table07_power_plausibility.csv", index=False)

    bounds = _load(sens / "power_bound_sensitivity.csv")
    table08_sensitivity(
        _load(sens / "battery_capacity.csv"),
        _load(sens / "physical_parameters.csv"),
        _load(sens / "preprocessing.csv"),
        bounds,
    ).to_csv(out / "table08_sensitivity.csv", index=False)

    feas = _load(sens / "feasibility.csv")
    if feas.empty:
        feas = _load(sens / "feasibility_summary.csv")
    if not feas.empty:
        table09_feasibility(feas).to_csv(out / "table09_feasibility.csv", index=False)

    print(f"Wrote tables under {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
