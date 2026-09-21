#!/usr/bin/env python
"""Conference-paper pipeline.

1. data audit
2. main LOTO if results are missing
3. data-scarcity experiment if results are missing
4. feasibility experiment if results are missing
5. final tables
6. final PNG figures

Reuses cached paper results when present. Does not run journal-scale extras.
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


def _present(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Conference-paper experiment and export pipeline.")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--rerun-loto", action="store_true")
    args = parser.parse_args()
    py = sys.executable
    loto_pred = ROOT / "outputs" / "loto" / "paper" / "per_trip_predictions.csv"
    scarcity = ROOT / "outputs" / "scarcity" / "paper" / "scarcity_per_trip.csv"
    feas = ROOT / "outputs" / "sensitivity" / "paper" / "feasibility.csv"

    _run([py, "scripts/audit_data.py", "--config", "configs/base.yaml"])
    if args.rerun_loto or not _present(loto_pred):
        _run([py, "scripts/run_loto.py", "--config", args.config])
    else:
        print("Reusing existing paper LOTO outputs.", flush=True)
    if not _present(scarcity):
        _run([py, "scripts/run_data_scarcity.py", "--config", args.config])
    else:
        print("Reusing existing paper data-scarcity outputs.", flush=True)
    if not _present(feas):
        _run([py, "scripts/run_feasibility_sensitivity.py", "--config", args.config])
    else:
        print("Reusing existing paper feasibility outputs.", flush=True)
    _run([py, "scripts/make_final_tables.py", "--config", args.config, "--out", "results"])
    _run([py, "scripts/make_final_figures.py", "--config", args.config, "--out", "results"])
    print("Final pipeline complete. Paper artifacts are in results/.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
