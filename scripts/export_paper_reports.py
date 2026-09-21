#!/usr/bin/env python
"""Copy lightweight paper review artifacts into reports/paper (no dataset, no checkpoints)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from paths import project_root, resolve_under_root  # noqa: E402


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_root()
    profile = str(config.get("experiment", {}).get("profile", "paper"))
    dest = root / "reports" / "paper"
    dest.mkdir(parents=True, exist_ok=True)
    loto = resolve_under_root("outputs/loto", root) / profile
    tables = resolve_under_root("outputs/tables", root) / profile
    stats = resolve_under_root("outputs/statistics", root)
    figs = resolve_under_root("outputs/figures", root) / "paper"
    _copy(root / "reports" / "protocol_frozen.md", dest / "protocol_frozen.md")
    _copy(resolve_under_root("outputs/data_audit", root) / "paper_dataset_summary.csv", dest / "dataset_summary.csv")
    _copy(loto / "summary_metrics.csv", dest / "main_results.csv")
    _copy(loto / "per_trip_predictions.csv", dest / "per_trip_results.csv")
    _copy(tables / "table_ablation.csv", dest / "ablation.csv")
    _copy(resolve_under_root("outputs/scarcity", root) / profile / "scarcity_per_trip.csv", dest / "data_scarcity.csv")
    _copy(stats / "paired_bootstrap_differences.csv", dest / "paired_bootstrap.csv")
    fig_dest = dest / "figures"
    if figs.exists():
        fig_dest.mkdir(exist_ok=True)
        for p in list(figs.glob("*.png"))[:12]:
            _copy(p, fig_dest / p.name)
            pdf = p.with_suffix(".pdf")
            if pdf.exists():
                _copy(pdf, fig_dest / pdf.name)
    readme = [
        "# Paper review artifacts",
        "",
        "Lightweight copies of experimental summaries. No HELECAR-D CSVs, checkpoints, or caches.",
        f"Profile: `{profile}`",
        "",
        "Regenerate with `make paper` after the frozen protocol run.",
        "",
    ]
    (dest / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote lightweight reports to {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
