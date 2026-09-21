#!/usr/bin/env python
"""Ablations A0–A3 as a LOTO method subset."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config
from run_loto import run_loto


def main() -> int:
    parser = argparse.ArgumentParser(description="A0 physics / A1 weak MLP / A2 PINN full-trip / A3 full PINN")
    parser.add_argument("--config", default="configs/quick.yaml")
    args = parser.parse_args()
    config = deepcopy(load_config(args.config))
    config.setdefault("experiment", {})
    config["experiment"]["methods"] = ["physics", "weak_mlp", "pinn_fulltrip", "pinn"]
    run_loto(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
