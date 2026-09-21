#!/usr/bin/env python
"""Ablations A0–A4. Reuses primary LOTO outputs when present."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from experiment.loto import run_loto  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402

ABLATION_METHODS = ["physics", "weak_mlp", "mlp_state", "pinn_no_dynamics", "pinn_fulltrip", "pinn"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Energy-model ablations (reuse LOTO when possible).")
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    loto_dir = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), project_root()) / profile
    pred_path = loto_dir / "per_trip_predictions.csv"
    if pred_path.exists():
        table = pd.read_csv(pred_path)
        have = set(table["method"].unique())
        missing = [m for m in ABLATION_METHODS if m not in have]
        if missing:
            config.setdefault("experiment", {})["methods"] = missing
            extra = run_loto(config, methods=missing)
            table = pd.concat([table, extra], ignore_index=True)
    else:
        table = run_loto(config, methods=ABLATION_METHODS)
    out = resolve_under_root(config.get("paths", {}).get("ablation_dir", "outputs/ablations"), project_root()) / profile
    out.mkdir(parents=True, exist_ok=True)
    sub = table.loc[table["method"].isin(ABLATION_METHODS)].copy()
    sub.to_csv(out / "ablation_per_trip.csv", index=False)
    print(f"Wrote ablations to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
