#!/usr/bin/env python
"""Highest-value Twizy sensitivity runs for INTIS review (Reviewer 3).

Writes only under review_results/sensitivity/. Never overwrites results/ or paper configs.
Predetermined grids; do not select a new paper default from these ranks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from experiment.loto import run_loto  # noqa: E402


def _base_paper_config() -> dict:
    cfg = load_config("configs/paper.yaml")
    cfg.setdefault("paths", {})["loto_dir"] = "review_results/sensitivity/loto"
    cfg.setdefault("experiment", {})["cache"] = True
    cfg["experiment"]["methods"] = ["pinn"]
    cfg["experiment"]["seeds"] = [0, 1, 2]
    return cfg


def run_variant(name: str, mutate) -> pd.DataFrame:
    cfg = _base_paper_config()
    mutate(cfg)
    cfg["experiment"]["profile"] = f"sens_{name}"
    out_root = ROOT / "review_results" / "sensitivity"
    out_root.mkdir(parents=True, exist_ok=True)
    table = run_loto(cfg, methods=["pinn"], seeds=[0, 1, 2])
    table = table.copy()
    table["sensitivity_variant"] = name
    table.to_csv(out_root / f"predictions_{name}.csv", index=False)
    summary = (
        table.groupby("method")["absolute_error_kwh"]
        .agg(mae="mean", n="count")
        .reset_index()
    )
    summary["sensitivity_variant"] = name
    summary.to_csv(out_root / f"summary_{name}.csv", index=False)
    print(summary.to_string(index=False))
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["residual", "lambda", "capacity", "all"], default="residual")
    args = parser.parse_args()
    frames: list[pd.DataFrame] = []

    if args.suite in {"residual", "all"}:
        for L in (2.0, 4.0, 6.0, 8.0):

            def mut_L(c, value=L):
                c.setdefault("training", {})["residual_limit_kw"] = float(value)

            frames.append(run_variant(f"L_{L:g}", mut_L))

    if args.suite in {"lambda", "all"}:
        for key, base in (("lambda_dynamics", 1.0), ("lambda_prior", 0.1)):
            for mult in (0.5, 1.0, 2.0):

                def mut_lam(c, key=key, base=base, mult=mult):
                    tr = c.setdefault("training", {})
                    tr[key] = float(base * mult)
                    if key == "lambda_prior":
                        tr["lambda_physics"] = float(base * mult)

                frames.append(run_variant(f"{key}_x{mult:g}", mut_lam))

    if args.suite in {"capacity", "all"}:
        for cap in (5.5, 6.0, 6.5):

            def mut_cap(c, cap=cap):
                vpath = ROOT / "configs" / "review" / f"_sens_vehicle_cap_{cap:g}.yaml"
                base_v = (ROOT / "configs" / "vehicle_twizy.yaml").read_text(encoding="utf-8")
                lines = []
                for line in base_v.splitlines():
                    if line.startswith("battery_capacity_kwh:"):
                        lines.append(f"battery_capacity_kwh: {cap}")
                    else:
                        lines.append(line)
                vpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
                c.setdefault("vehicle", {})["config"] = str(vpath.relative_to(ROOT)).replace("\\", "/")

            frames.append(run_variant(f"capacity_{cap:g}", mut_cap))

    if frames:
        all_tab = pd.concat(frames, ignore_index=True)
        out_dir = ROOT / "review_results" / "sensitivity"
        all_tab.to_csv(out_dir / "all_sensitivity_predictions.csv", index=False)
        mae = (
            all_tab.groupby("sensitivity_variant")["absolute_error_kwh"]
            .mean()
            .rename("mae_kwh")
            .reset_index()
        )
        mae.to_csv(out_dir / "all_sensitivity_mae.csv", index=False)
        print(mae.to_string(index=False))
        (out_dir / "meta.json").write_text(
            json.dumps({"suite": args.suite, "seeds": [0, 1, 2], "methods": ["pinn"]}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
