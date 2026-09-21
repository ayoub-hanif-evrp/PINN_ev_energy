#!/usr/bin/env python
"""Write paper-ready PNG figures under results/figures/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from experiment.data import load_processed, representative_test_ids  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402
from plotting.final_figures import (  # noqa: E402
    fig01_method_overview,
    fig02_observed_vs_predicted,
    fig03_trip_absolute_errors,
    fig04_ablation,
    fig05_data_scarcity,
    fig06_cross_trajectory,
    fig07_soc_reconstruction,
    fig08_power_plausibility,
    fig09_feasibility,
    fig10_training_diagnostics,
)


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Final paper PNG figures.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    fig_dir = resolve_under_root(args.out, root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    loto = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root) / profile
    pred = _load(loto / "per_trip_predictions.csv")

    fig01_method_overview(fig_dir / "fig01_method_overview.png")
    if not pred.empty:
        fig02_observed_vs_predicted(pred, fig_dir / "fig02_observed_vs_predicted.png")
        fig03_trip_absolute_errors(pred, fig_dir / "fig03_trip_absolute_errors.png")
        fig04_ablation(pred, fig_dir / "fig04_ablation.png")

    scarcity = _load(resolve_under_root("outputs/scarcity", root) / profile / "scarcity_per_trip.csv")
    if not scarcity.empty:
        fig05_data_scarcity(scarcity, fig_dir / "fig05_data_scarcity.png")
    xtraj = _load(resolve_under_root("outputs/cross_trajectory", root) / profile / "cross_trajectory_per_trip.csv")
    if not xtraj.empty:
        fig06_cross_trajectory(xtraj, fig_dir / "fig06_cross_trajectory.png")

    trips = load_processed(config)
    rec_dir = loto / "soc_reconstruction"
    params = load_vehicle_parameters(config=config)
    if trips:
        fig07_soc_reconstruction(trips, rec_dir, fig_dir / "fig07_soc_reconstruction.png")
        fig08_power_plausibility(trips, rec_dir, params, fig_dir / "fig08_power_plausibility.png")

    feas = _load(resolve_under_root("outputs/sensitivity", root) / profile / "feasibility.csv")
    if feas.empty:
        feas = _load(resolve_under_root("outputs/sensitivity", root) / profile / "feasibility_summary.csv")
    if not feas.empty:
        fig09_feasibility(feas, fig_dir / "fig09_feasibility.png")

    ids = representative_test_ids(trips) if trips else []
    hist = None
    for trip_id in ids:
        cand = loto / "training_histories" / f"{trip_id}__pinn__seed0__select.csv"
        if cand.exists():
            hist = cand
            break
    if hist is None:
        cands = sorted((loto / "training_histories").glob("*__pinn__seed0__select.csv"))
        hist = cands[0] if cands else None
    if hist is not None:
        fig10_training_diagnostics(hist, fig_dir / "fig10_training_diagnostics.png")

    print(f"Wrote figures under {fig_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
