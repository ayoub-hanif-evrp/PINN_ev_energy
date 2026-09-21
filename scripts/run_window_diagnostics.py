#!/usr/bin/env python
"""Multi-window vs full-trip diagnostic. Does not retune window weights."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from data.windows import generate_windows  # noqa: E402
from evaluation.bootstrap import bootstrap_mean_ci, trip_absolute_errors  # noqa: E402
from experiment.data import load_processed  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402
from data.quantization import q_from_trips  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-window ablation diagnostic.")
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    loto_dir = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), root) / profile
    out_dir = loto_dir / "window_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    trips = load_processed(config)
    params = load_vehicle_parameters(config=config)
    q = q_from_trips(trips)
    overlapping = generate_windows(trips, params.battery_capacity_kwh, q, config=config, overlapping=True)
    nonoverlap = generate_windows(trips, params.battery_capacity_kwh, q, config=config, overlapping=False)

    def _count(windows) -> dict[str, int]:
        c = Counter(w.scale for w in windows)
        event = sum(n for s, n in c.items() if str(s).startswith("soc_event"))
        fixed = sum(n for s, n in c.items() if str(s).startswith("time_"))
        full = int(c.get("full_trip", 0))
        return {
            "n_windows_total": int(len(windows)),
            "n_event_windows": int(event),
            "n_fixed_windows": int(fixed),
            "n_full_trip_windows": int(full),
            **{f"n_{k}": int(v) for k, v in sorted(c.items())},
        }

    counts = pd.DataFrame(
        [
            {"split": "train_style_overlapping", **_count(overlapping)},
            {"split": "eval_style_nonoverlapping", **_count(nonoverlap)},
        ]
    )
    counts.to_csv(out_dir / "window_counts.csv", index=False)

    win_path = out_dir / "nonoverlapping_test_windows.csv"
    window_mae = pd.DataFrame()
    if win_path.exists():
        win = pd.read_csv(win_path)
        scale_rows = win.loc[win["level"] == "scale"] if "level" in win.columns else win
        if not scale_rows.empty:
            window_mae = (
                scale_rows.groupby(["method", "scale"], dropna=False)
                .agg(mae_kwh=("mae_kwh", "mean"), n_windows=("n_windows", "mean"), n_rows=("mae_kwh", "size"))
                .reset_index()
            )
            window_mae.to_csv(out_dir / "test_window_mae_by_scale.csv", index=False)

    val_rows = []
    for path in sorted((loto_dir / "fold_manifests").glob("*.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        for method, meta in (body.get("methods") or {}).items():
            per = meta.get("validation_mae_by_scale") or {}
            val_rows.append(
                {
                    "trip_id": body.get("test_id"),
                    "seed": body.get("seed"),
                    "method": method,
                    "validation_score": meta.get("validation_score"),
                    **{f"val_mae_{k}": v for k, v in per.items()},
                }
            )
    val = pd.DataFrame(val_rows)
    if not val.empty:
        val.to_csv(out_dir / "validation_mae_by_scale.csv", index=False)
        numeric = [c for c in val.columns if c.startswith("val_mae_")]
        if numeric:
            val.groupby("method")[numeric].mean().reset_index().to_csv(
                out_dir / "validation_mae_by_scale_mean.csv", index=False
            )

    pred_path = loto_dir / "per_trip_predictions.csv"
    trip_rows = []
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        for method in ("pinn", "pinn_fulltrip", "weak_mlp"):
            err = trip_absolute_errors(pred, method)
            stats = bootstrap_mean_ci(err.to_numpy(dtype=float))
            trip_rows.append({"method": method, "n_trips": int(len(err)), **stats})
    trip_tab = pd.DataFrame(trip_rows)
    trip_tab.to_csv(out_dir / "trip_mae_pinn_vs_fulltrip.csv", index=False)

    note = (
        "PINN_FULLTRIP uses only the full-trip energy window. "
        "PINN uses scale-balanced multi-window Smooth L1 plus dynamics/state/prior. "
        "Window weights were not retuned from held-out trip-energy MAE. "
        "A lower trip-energy MAE for PINN_FULLTRIP does not imply that multi-window "
        "supervision is unused: check test_window_mae_by_scale.csv for local consistency."
    )
    (out_dir / "window_diagnostic_note.txt").write_text(note, encoding="utf-8")
    print(counts.to_string(index=False), flush=True)
    if not trip_tab.empty:
        print(trip_tab.to_string(index=False), flush=True)
    print(note, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
