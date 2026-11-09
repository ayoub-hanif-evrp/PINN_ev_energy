#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.bootstrap import bootstrap_mean_ci, paired_difference_ci, trip_absolute_errors  # noqa: E402

LABELS = {
    "physics": "Physics Model",
    "weak_mlp": "Data-Driven MLP",
    "pinn": "Full PINN",
    "pinn_fulltrip": "PINN full-trip only",
    "pinn_no_dynamics": "PINN no conservation",
    "pinn_no_state": "PINN no state loss",
    "pinn_no_prior": "PINN no residual prior",
}


def main() -> int:
    paper = pd.read_csv(ROOT / "outputs" / "loto" / "paper" / "per_trip_predictions.csv")
    review = pd.read_csv(
        ROOT / "outputs" / "review" / "ablations" / "loto" / "review_ablations" / "per_trip_predictions.csv"
    )
    pred = pd.concat(
        [
            paper.loc[paper["method"].isin(["physics", "weak_mlp", "pinn", "pinn_fulltrip", "pinn_no_dynamics"])],
            review.loc[review["method"].isin(["pinn_no_state", "pinn_no_prior"])],
        ],
        ignore_index=True,
    )
    out = ROOT / "review_results" / "ablations"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, name in LABELS.items():
        err = trip_absolute_errors(pred, method)
        ci = bootstrap_mean_ci(err.to_numpy(dtype=float))
        rows.append(
            {
                "method": name,
                "method_id": method,
                "n_trips": int(len(err)),
                "MAE_kWh": float(err.mean()),
                "MAE_CI_low": float(ci["ci_low"]),
                "MAE_CI_high": float(ci["ci_high"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out / "table_ablation_complete.csv", index=False)

    pinn = trip_absolute_errors(pred, "pinn")
    paired_rows = []
    for method, name in LABELS.items():
        if method == "pinn":
            continue
        other = trip_absolute_errors(pred, method)
        stats = paired_difference_ci(pinn, other)
        aligned = pd.concat([pinn.rename("a"), other.rename("b")], axis=1, join="inner").dropna()
        diff = aligned["a"] - aligned["b"]
        p_value = float(wilcoxon(diff.to_numpy(dtype=float), zero_method="wilcox").pvalue)
        paired_rows.append(
            {
                "comparison": f"PINN - {name}",
                "mean_diff_kWh": float(stats["mean"]),
                "CI_low": float(stats["ci_low"]),
                "CI_high": float(stats["ci_high"]),
                "Wilcoxon_p": p_value,
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "table_ablation_paired_complete.csv", index=False)

    print(table.to_string(index=False))
    print()
    print(paired.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
