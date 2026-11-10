#!/usr/bin/env python
"""Generate publication-quality figures for INTIS 2026 review experiments.

Writes only under review_results/figures/. Does not touch results/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from plotting.review_figures import (  # noqa: E402
    fig_ablation_paired_diff,
    fig_ablation_tesla_two_panel,
    fig_sensitivity_L,
    fig_tesla_meas_mae,
    load_and_verify_ablation_paired,
    load_and_verify_sensitivity_L,
    load_and_verify_tesla_mae,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="review_results/figures",
        help="Output directory (default: review_results/figures)",
    )
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    paired = load_and_verify_ablation_paired(
        ROOT / "review_results" / "ablations" / "table_ablation_paired_complete.csv"
    )
    tesla = load_and_verify_tesla_mae(ROOT / "review_results" / "tesla" / "loto")
    sens = load_and_verify_sensitivity_L(
        ROOT / "review_results" / "sensitivity" / "all_sensitivity_mae.csv"
    )

    paths = {}
    paths["figA"] = fig_ablation_paired_diff(paired, out / "figA_ablation_paired_diff.png")
    paths["figB"] = fig_tesla_meas_mae(tesla, out / "figB_tesla_meas_mae.png")
    paths["figC"] = fig_sensitivity_L(sens, out / "figC_sensitivity_L.png")
    paths["figAB"] = fig_ablation_tesla_two_panel(
        paired, tesla, out / "figAB_ablation_tesla_two_panel.png"
    )

    verify = {
        "ablation_paired_means_kWh": {
            row["comparator"]: {
                "mean_diff": float(row["mean_diff_kWh"]),
                "CI_low": float(row["CI_low"]),
                "CI_high": float(row["CI_high"]),
            }
            for _, row in paired.iterrows()
        },
        "tesla_mae_vs_meas_kWh": {
            f"{row['dataset']}/{row['method']}": float(row["mae_kwh"]) for _, row in tesla.iterrows()
        },
        "sensitivity_L_mae_kWh": {
            int(row["L_kW"]): float(row["mae_kwh"]) for _, row in sens.iterrows()
        },
        "outputs": {k: {fmt: str(p.relative_to(ROOT)) for fmt, p in v.items()} for k, v in paths.items()},
        "notes": [
            "Difference = Full PINN AE − comparator AE; negative means Full PINN better.",
            "Tesla MAE recomputed from trip_energy_with_meas_ref abs_err_vs_meas_kwh (seed-mean then trip-mean).",
            "Sensitivity figure uses point MAE only; no CI available in archive.",
            "SoC-state diagnostics were not used as accuracy evidence in these figures.",
        ],
    }
    meta_path = out / "figure_verification.json"
    meta_path.write_text(json.dumps(verify, indent=2), encoding="utf-8")

    print("Verified and wrote:")
    for name, mapping in paths.items():
        print(f"  {name}: {mapping['png'].relative_to(ROOT)}")
    print(f"  meta: {meta_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
