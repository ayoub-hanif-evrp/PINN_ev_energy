#!/usr/bin/env python
"""Final paper pipeline: audit, required experiments if missing, PNG figures, CSV tables.

Does not run smoke/quick. Does not retune from test ranks.
Existing frozen LOTO outputs are reused when present.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Final paper experiment and export pipeline.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--rerun-loto", action="store_true")
    parser.add_argument("--retrain-bounded-pinn", action="store_true")
    args = parser.parse_args()
    py = sys.executable
    loto_pred = ROOT / "outputs" / "loto" / "paper" / "per_trip_predictions.csv"

    _run([py, "scripts/audit_data.py", "--config", "configs/base.yaml"])
    if args.rerun_loto or not loto_pred.exists():
        _run([py, "scripts/run_loto.py", "--config", args.config])
    else:
        print("Reusing existing paper LOTO outputs.", flush=True)
    _run([py, "scripts/run_data_scarcity.py", "--config", args.config])
    xtraj = ROOT / "outputs" / "cross_trajectory" / "paper" / "cross_trajectory_per_trip.csv"
    if not xtraj.exists():
        _run([py, "scripts/run_cross_trajectory.py", "--config", args.config])
    sens = ROOT / "outputs" / "sensitivity" / "paper" / "physical_parameters.csv"
    if not sens.exists():
        _run([py, "scripts/run_sensitivity.py", "--config", args.config])
    feas = ROOT / "outputs" / "sensitivity" / "paper" / "feasibility.csv"
    if not feas.exists():
        _run([py, "scripts/run_feasibility_sensitivity.py", "--config", args.config])
    _run([py, "scripts/run_power_plausibility.py", "--config", args.config])
    bound_cmd = [py, "scripts/run_power_bound_sensitivity.py", "--config", args.config]
    if args.retrain_bounded_pinn:
        bound_cmd.append("--retrain-pinn")
    _run(bound_cmd)
    _run([py, "scripts/run_window_diagnostics.py", "--config", args.config])
    _run([py, "scripts/make_final_tables.py", "--config", args.config, "--out", "results"])
    _run([py, "scripts/make_final_figures.py", "--config", args.config, "--out", "results"])
    print("Final pipeline complete. Paper artifacts are in results/.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
