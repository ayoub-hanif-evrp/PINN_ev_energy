#!/usr/bin/env python
"""Analyze cached HELECAR-D ablation predictions without retraining.

Uses existing leave-one-trip-out rows under outputs/loto/paper/.
Does not modify results/ paper artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.bootstrap import bootstrap_mean_ci, paired_difference_ci, trip_absolute_errors  # noqa: E402
from evaluation.metrics import summarize_energy_table  # noqa: E402

LABELS = {
    "physics": "Physics Model (no residual)",
    "ridge": "Ridge Regression",
    "weak_mlp": "Data-Driven MLP",
    "pinn": "Full PINN (reference)",
    "pinn_fulltrip": "PINN full-trip supervision only",
    "pinn_no_dynamics": "PINN without conservation loss",
    "mlp_state": "MLP with depletion state (legacy)",
}


def main() -> int:
    pred = pd.read_csv(ROOT / "outputs" / "loto" / "paper" / "per_trip_predictions.csv")
    out_dir = ROOT / "review_results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in LABELS if m in set(pred["method"])]
    rows = []
    for method in methods:
        err = trip_absolute_errors(pred, method)
        ci = bootstrap_mean_ci(err.to_numpy(dtype=float))
        summary = summarize_energy_table(pred.loc[pred["method"] == method]).get(method, {})
        rows.append(
            {
                "method_id": method,
                "method": LABELS[method],
                "n_trips": int(len(err)),
                "MAE_kWh": float(err.mean()),
                "MAE_CI_low": float(ci["ci_low"]),
                "MAE_CI_high": float(ci["ci_high"]),
                "RMSE_kWh": float(summary.get("rmse_kwh", float("nan"))),
                "Bias_kWh": float(summary.get("bias_kwh", float("nan"))),
                "R2": float(summary.get("r2", float("nan"))),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_ablation_main.csv", index=False)

    pinn = trip_absolute_errors(pred, "pinn")
    paired_rows = []
    for method in methods:
        if method == "pinn":
            continue
        other = trip_absolute_errors(pred, method)
        stats = paired_difference_ci(pinn, other)
        aligned = pd.concat([pinn.rename("pinn"), other.rename("other")], axis=1, join="inner").dropna()
        diff = aligned["pinn"] - aligned["other"]
        try:
            p_value = float(wilcoxon(diff.to_numpy(dtype=float), zero_method="wilcox").pvalue)
        except ValueError:
            p_value = float("nan")
        paired_rows.append(
            {
                "comparison": f"PINN - {LABELS[method]}",
                "method_id": method,
                "n_trips": int(len(aligned)),
                "mean_absolute_error_difference_kWh": float(stats["mean"]),
                "bootstrap_CI_low": float(stats["ci_low"]),
                "bootstrap_CI_high": float(stats["ci_high"]),
                "Wilcoxon_p": p_value,
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / "table_ablation_paired_vs_pinn.csv", index=False)

    def md_table(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in frame.iterrows():
            cells = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    cells.append(f"{value:.3f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    notes = [
        "# Cached HELECAR-D ablation analysis",
        "",
        "Source: `outputs/loto/paper/per_trip_predictions.csv` (existing leave-one-trip-out cache).",
        "No model was retrained for this table.",
        "",
        "## Method definitions (from historical trainer)",
        "",
        "- `pinn`: full discrete-time PINN (multi-window + conservation + state + residual prior).",
        "- `physics`: analytical vehicle power only (no neural residual).",
        "- `pinn_fulltrip`: same PINN architecture, but window supervision restricted to `full_trip` scale.",
        "- `pinn_no_dynamics`: PINN with conservation/dynamics loss weight set to zero.",
        "- `weak_mlp`: data-driven MLP with multi-window energy supervision only.",
        "- `mlp_state`: legacy MLP with a depletion head (not a requested camera-ready ablation).",
        "",
        "## Main trip-energy MAE",
        "",
        md_table(table),
        "",
        "## Paired absolute-error difference: Full PINN minus comparator",
        "",
        "Negative values mean lower PINN absolute error.",
        "",
        md_table(paired),
        "",
        "## Scientific note",
        "",
        "If `pinn_fulltrip` has a lower point-estimate MAE than full multi-window `pinn`, that finding is reported,",
        "not concealed. Multi-window supervision remains the accepted-paper method.",
        "",
    ]
    (out_dir / "ablation_cached_analysis.md").write_text("\n".join(notes), encoding="utf-8")
    print(table.to_string(index=False))
    print()
    print(paired.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
