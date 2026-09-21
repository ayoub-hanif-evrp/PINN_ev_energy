"""Paired statistical summaries. Do not claim significance from a CI alone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.bootstrap import bootstrap_mean_ci, paired_difference_ci, per_seed_metrics, seed_mean_std, trip_absolute_errors


PRIMARY_PAIRS = (
    ("pinn", "weak_mlp"),
    ("pinn", "elasticnet"),
    ("pinn", "physics"),
)


def wilcoxon_pairs(frame: pd.DataFrame, pairs: tuple[tuple[str, str], ...] = PRIMARY_PAIRS) -> pd.DataFrame:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return pd.DataFrame()
    rows = []
    for a, b in pairs:
        ea = trip_absolute_errors(frame, a)
        eb = trip_absolute_errors(frame, b)
        aligned = pd.concat([ea.rename("a"), eb.rename("b")], axis=1, join="inner").dropna()
        if len(aligned) < 3:
            continue
        diff = aligned["a"] - aligned["b"]
        if (diff == 0).all():
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(diff.to_numpy(), alternative="two-sided", zero_method="wilcox")
        rows.append(
            {
                "method_a": a,
                "method_b": b,
                "difference": "absolute_error_a - absolute_error_b",
                "n_trips": int(len(aligned)),
                "wilcoxon_stat": float(stat),
                "p_value": float(p),
                "note": "Secondary evidence only. Not the primary inference procedure.",
            }
        )
    return pd.DataFrame(rows)


def write_statistical_outputs(
    frame: pd.DataFrame,
    out_dir: Path,
    n_resamples: int = 10_000,
    seed: int = 20260921,
    pairs: tuple[tuple[str, str], ...] = PRIMARY_PAIRS,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = per_seed_metrics(frame)
    per_seed.to_csv(out_dir / "metrics_by_seed.csv", index=False)
    seed_mean_std(per_seed).to_csv(out_dir / "metrics_by_seed_mean_std.csv", index=False)

    mae_rows = []
    methods = sorted(frame["method"].unique())
    trip_err = {m: trip_absolute_errors(frame, m) for m in methods}
    for method, errors in trip_err.items():
        stats = bootstrap_mean_ci(errors.to_numpy(dtype=float), n_resamples=n_resamples, seed=seed)
        mae_rows.append({"method": method, "metric": "mae_kwh", **stats})
    mae_df = pd.DataFrame(mae_rows)
    mae_df.to_csv(out_dir / "bootstrap_main_metrics.csv", index=False)

    diff_rows = []
    for a, b in pairs:
        if a not in trip_err or b not in trip_err:
            continue
        stats = paired_difference_ci(trip_err[a], trip_err[b], n_resamples=n_resamples, seed=seed)
        diff_rows.append(
            {
                "method_a": a,
                "method_b": b,
                "difference": "AE(a) - AE(b)",
                "sign_note": "negative => a has lower absolute error",
                **stats,
            }
        )
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(out_dir / "paired_bootstrap_differences.csv", index=False)

    wil = wilcoxon_pairs(frame, pairs)
    if not wil.empty:
        wil.to_csv(out_dir / "wilcoxon_results.csv", index=False)

    lines = [
        "# Statistical summary",
        "",
        "Trip is the primary statistical unit. Neural multi-seed absolute errors are averaged per trip before pairing.",
        "Main reported models are independent-seed fits, not an unannounced ensemble of predicted energies.",
        "",
        "## MAE 95% trip-level bootstrap intervals",
        "",
    ]
    for _, r in mae_df.iterrows():
        lines.append(
            f"- {r['method']}: mean MAE {r['mean']:.4f} kWh with a 95% bootstrap interval of "
            f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] kWh (n={int(r['n'])} trips)."
        )
    lines += ["", "## Paired absolute-error differences (trip-level bootstrap)", ""]
    for _, r in diff_df.iterrows():
        lines.append(
            f"- {r['method_a']} minus {r['method_b']}: mean paired difference {r['mean']:.4f} kWh "
            f"with a 95% bootstrap interval of [{r['ci_low']:.4f}, {r['ci_high']:.4f}] kWh. "
            "Negative values mean the first method has lower absolute error."
        )
        lo, hi = r["ci_low"], r["ci_high"]
        if lo > 0 or hi < 0:
            lines.append(
                "  The interval does not contain zero; this is a descriptive bootstrap statement, not a formal significance claim."
            )
        else:
            lines.append("  The interval contains zero.")
    if not wil.empty:
        lines += ["", "## Wilcoxon signed-rank (secondary)", ""]
        for _, r in wil.iterrows():
            lines.append(
                f"- {r['method_a']} vs {r['method_b']}: statistic={r['wilcoxon_stat']:.4g}, p={r['p_value']:.4g} (two-sided)."
            )
    (out_dir / "statistical_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"mae": mae_df, "paired": diff_df, "wilcoxon": wil, "per_seed": per_seed}
