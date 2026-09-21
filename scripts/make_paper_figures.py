#!/usr/bin/env python
"""Build publication figures from completed experiment outputs only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from experiment.data import load_processed, representative_test_ids  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402
from plotting.paper_figures import (  # noqa: E402
    figure_ablation,
    figure_architecture,
    figure_cross_trajectory,
    figure_delta_p,
    figure_obs_vs_pred,
    figure_per_trip_error,
    figure_representative_trip,
    figure_scarcity,
    figure_soc_reconstruction,
    figure_soc_transitions,
)


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Publication figures from real outputs.")
    parser.add_argument("--results", default="outputs/")
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root()
    results = resolve_under_root(args.results, root)
    fig_dir = resolve_under_root(config.get("paths", {}).get("figures_dir", "outputs/figures"), root) / "paper"
    fig_dir.mkdir(parents=True, exist_ok=True)

    figure_architecture(fig_dir / "fig01_architecture.png")
    trips = load_processed(config)
    if trips:
        figure_representative_trip(trips[0], fig_dir / "fig02_representative_trip.png")
        from data.quantization import combine_quantization, estimate_soc_quantization

        cq = combine_quantization([estimate_soc_quantization(t.frame["soc"]) for t in trips])
        figure_soc_transitions(cq, fig_dir / "fig03_soc_transitions.png")

    pred_path = _first_existing(results / "loto" / "paper" / "per_trip_predictions.csv", results / "loto" / "quick" / "per_trip_predictions.csv")
    if pred_path:
        pred = pd.read_csv(pred_path)
        figure_obs_vs_pred(pred, fig_dir / "fig04_obs_vs_pred.png")
        figure_per_trip_error(pred, fig_dir / "fig05_per_trip_error.png")
        figure_ablation(pred, fig_dir / "fig08_ablation.png")

        rec_dir = pred_path.parent / "soc_reconstruction"
        ids = representative_test_ids(trips) if trips else []
        by_id = {t.trip_id: t for t in trips}
        shown = 0
        for trip_id in ids:
            npz = rec_dir / f"{trip_id}__pinn__seed0.npz"
            if npz.exists() and trip_id in by_id:
                data = np.load(npz)
                figure_soc_reconstruction(by_id[trip_id], data["soc_hat"], fig_dir / f"fig06_soc_{trip_id}.png")
                figure_delta_p({k: data[k] for k in data.files}, fig_dir / f"fig09_deltaP_{trip_id}.png")
                shown += 1
        if shown == 0:
            candidates = sorted(rec_dir.glob("*__pinn__seed0.npz"))
            if candidates and trips:
                data = np.load(candidates[0])
                tid = candidates[0].name.split("__")[0]
                if tid in by_id:
                    figure_soc_reconstruction(by_id[tid], data["soc_hat"], fig_dir / "fig06_soc_reconstruction.png")
                    figure_delta_p({k: data[k] for k in data.files}, fig_dir / "fig09_deltaP.png")

    scarcity = _first_existing(results / "scarcity" / "paper" / "scarcity_per_trip.csv")
    if scarcity:
        figure_scarcity(pd.read_csv(scarcity), fig_dir / "fig07_data_scarcity.png")
    xtraj = _first_existing(results / "cross_trajectory" / "paper" / "cross_trajectory_per_trip.csv")
    if xtraj:
        figure_cross_trajectory(pd.read_csv(xtraj), fig_dir / "fig10_cross_trajectory.png")

    print(f"Wrote figures under {fig_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
