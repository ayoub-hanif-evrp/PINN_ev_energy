#!/usr/bin/env python
"""Leave-one-trip-out energy estimation.

Paper:  python scripts/run_loto.py --config configs/paper.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from experiment.loto import run_loto  # noqa: E402
from experiment.sanity import write_sanity_report  # noqa: E402
from paths import resolve_under_root, project_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Leave-one-trip-out EV energy experiment.")
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    table = run_loto(config)
    profile = str(config.get("experiment", {}).get("profile", "base"))
    out_dir = resolve_under_root(config.get("paths", {}).get("loto_dir", "outputs/loto"), project_root()) / profile
    write_sanity_report(out_dir, table, profile=profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
