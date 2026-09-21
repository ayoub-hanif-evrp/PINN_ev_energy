#!/usr/bin/env python
"""Write CSV + LaTeX paper tables from completed experiment outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from evaluation.bootstrap import per_seed_metrics  # noqa: E402
from evaluation.tables import main_results_table, write_table  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper CSV/LaTeX tables.")
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    tables = resolve_under_root(config.get("paths", {}).get("tables_dir", "outputs/tables"), root) / profile
    loto = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root) / profile
    pred = _load(loto / "per_trip_predictions.csv")
    boot = _load(resolve_under_root("outputs/statistics", root) / "bootstrap_main_metrics.csv")
    if not pred.empty:
        write_table(main_results_table(pred, boot if not boot.empty else None), tables / "table_main_results", "Main LOTO energy results", "tab:main")
        write_table(per_seed_metrics(pred), tables / "table_main_results_by_seed", "Main results by seed", "tab:byseed")
        write_table(pred, tables / "table_per_trip_results", "Per-trip energy predictions", "tab:pertrip")
        soc = _load(loto / "soc_reconstruction_metrics.csv")
        if not soc.empty:
            write_table(soc, tables / "table_soc_reconstruction", "SoC reconstruction on held-out trips", "tab:soc")
        ablation = pred.loc[pred["method"].isin(["physics", "weak_mlp", "mlp_state", "pinn_no_dynamics", "pinn_fulltrip", "pinn"])]
        write_table(main_results_table(ablation), tables / "table_ablation", "Ablation energy results", "tab:ablation")
    audit = _load(resolve_under_root("outputs/data_audit", root) / "paper_dataset_summary.csv")
    if not audit.empty:
        write_table(audit, tables / "table_dataset_summary", "HELECAR-D analysed-trip summary", "tab:data")
    trips = _load(resolve_under_root("outputs/data_audit", root) / "trip_characteristics.csv")
    if not trips.empty:
        write_table(trips, tables / "table_trip_characteristics", "Trip characteristics", "tab:trips")
    scarcity = _load(resolve_under_root("outputs/scarcity", root) / profile / "scarcity_per_trip.csv")
    if not scarcity.empty:
        write_table(scarcity, tables / "table_data_scarcity", "Data-scarcity experiment", "tab:scarcity")
    xtraj = _load(resolve_under_root("outputs/cross_trajectory", root) / profile / "cross_trajectory_per_trip.csv")
    if not xtraj.empty:
        write_table(xtraj, tables / "table_cross_trajectory", "Cross-trajectory generalization", "tab:xtraj")
    sens = resolve_under_root("outputs/sensitivity", root) / profile
    for name, caption, label in (
        ("physical_parameters.csv", "Physical-parameter sensitivity (physics-only)", "tab:physsens"),
        ("battery_capacity.csv", "Battery-capacity label sensitivity", "tab:capsens"),
        ("feasibility_summary.csv", "Routing-oriented battery-feasibility sensitivity", "tab:feas"),
        ("preprocessing.csv", "Preprocessing smoother sensitivity", "tab:presens"),
    ):
        frame = _load(sens / name)
        if not frame.empty:
            write_table(frame, tables / f"table_{Path(name).stem}", caption, label)
    print(f"Wrote tables under {tables}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
