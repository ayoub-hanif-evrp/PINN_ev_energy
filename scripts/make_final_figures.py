#!/usr/bin/env python
"""Write conference-paper PNG figures under results/figures/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from experiment.data import load_processed  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402
from plotting.final_figures import (  # noqa: E402
    fig01_method,
    fig02_observed_vs_predicted,
    fig03_trip_errors,
    fig04_data_scarcity,
    fig05_feasibility,
    fig06_soc_reconstruction,
    fig_error_ecdf,
    fig_mae_ci,
    fig_pinn_vs_mlp,
    fig_residuals,
    fig_training_loss,
    fig_trip_error_heatmap,
    fig_window_consistency,
)

OLD_NAMES = (
    "fig04_data_scarcity.png",
    "fig05_feasibility.png",
    "fig06_soc_reconstruction.png",
)


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Final conference-paper PNG figures.")
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

    fig01_method(fig_dir / "fig01_method.png")
    if not pred.empty:
        fig02_observed_vs_predicted(pred, fig_dir / "fig02_observed_vs_predicted.png")
        fig03_trip_errors(pred, fig_dir / "fig03_trip_errors.png")
        fig_mae_ci(pred, fig_dir / "fig04_mae_ci.png")
        fig_residuals(pred, fig_dir / "fig05_residuals.png")
        fig_error_ecdf(pred, fig_dir / "fig06_error_ecdf.png")
        fig_trip_error_heatmap(pred, fig_dir / "fig07_trip_error_heatmap.png")
        fig_pinn_vs_mlp(pred, fig_dir / "fig09_pinn_vs_mlp.png")

    scarcity = _load(resolve_under_root("outputs/scarcity", root) / profile / "scarcity_per_trip.csv")
    if not scarcity.empty and "ridge" in set(scarcity.get("method", [])):
        fig04_data_scarcity(scarcity, fig_dir / "fig08_data_scarcity.png")

    trips = load_processed(config)
    rec_dir = loto / "soc_reconstruction"
    if trips:
        fig06_soc_reconstruction(trips, rec_dir, fig_dir / "fig10_soc_reconstruction.png")

    fig_window_consistency(
        loto / "window_eval" / "nonoverlapping_test_windows.csv",
        fig_dir / "fig11_window_consistency.png",
    )

    feas = _load(resolve_under_root("outputs/sensitivity", root) / profile / "feasibility.csv")
    if feas.empty:
        feas = _load(resolve_under_root("outputs/sensitivity", root) / profile / "feasibility_summary.csv")
    if not feas.empty:
        fig05_feasibility(feas, fig_dir / "fig12_feasibility.png")

    history = loto / "training_histories" / "T1_03_29_2021__pinn__seed0__select.csv"
    fig_training_loss(history, fig_dir / "fig13_training_loss.png")

    for name in OLD_NAMES:
        stale = fig_dir / name
        if stale.exists():
            stale.unlink()

    print(f"Wrote figures under {fig_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
