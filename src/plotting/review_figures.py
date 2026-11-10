"""Publication figures for INTIS 2026 review experiments (separate from paper figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.names import METHOD_COLORS, METHOD_MARKERS, label
from plotting.style import DPI, apply_style

# Ablation comparator display order (paired table uses "PINN - <name>").
ABLATION_ORDER = [
    "Physics Model",
    "Data-Driven MLP",
    "PINN full-trip only",
    "PINN no conservation",
    "PINN no state loss",
    "PINN no residual prior",
]

TESLA_METHODS = ["physics", "weak_mlp", "pinn"]
TESLA_PROFILES = {
    "tesla_bev1_loto_corrected": "Model Y BEV1",
    "tesla_bev2_loto_corrected": "Model Y BEV2",
}


def save_publication(fig: plt.Figure, stem: Path) -> dict[str, Path]:
    """Export PNG (400 dpi), PDF, and SVG next to stem (suffix ignored)."""
    stem = Path(stem)
    out_dir = stem.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / stem.stem
    paths = {
        "png": base.with_suffix(".png"),
        "pdf": base.with_suffix(".pdf"),
        "svg": base.with_suffix(".svg"),
    }
    fig.savefig(paths["png"], dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _parse_comparator(comparison: str) -> str:
    prefix = "PINN - "
    if not str(comparison).startswith(prefix):
        raise ValueError(f"Unexpected paired comparison label: {comparison}")
    return str(comparison)[len(prefix) :]


def load_and_verify_ablation_paired(path: Path) -> pd.DataFrame:
    paired = pd.read_csv(path)
    required = {"comparison", "mean_diff_kWh", "CI_low", "CI_high"}
    missing = required - set(paired.columns)
    if missing:
        raise ValueError(f"Ablation paired table missing columns: {sorted(missing)}")
    paired = paired.copy()
    paired["comparator"] = paired["comparison"].map(_parse_comparator)
    # Interval must contain the mean.
    bad = ~(
        (paired["CI_low"] <= paired["mean_diff_kWh"] + 1e-12)
        & (paired["mean_diff_kWh"] <= paired["CI_high"] + 1e-12)
    )
    if bad.any():
        raise ValueError("Ablation CI does not contain mean_diff for some rows.")
    present = set(paired["comparator"])
    missing_names = [n for n in ABLATION_ORDER if n not in present]
    if missing_names:
        raise ValueError(f"Ablation paired table missing comparators: {missing_names}")
    ordered = paired.set_index("comparator").loc[ABLATION_ORDER].reset_index()
    return ordered


def load_and_verify_tesla_mae(loto_root: Path) -> pd.DataFrame:
    """Recompute V×I MAE from trip_energy_with_meas_ref and match summary_vs_meas."""
    rows = []
    for profile, display in TESLA_PROFILES.items():
        d = loto_root / profile
        summary = pd.read_csv(d / "summary_vs_meas.csv")
        detail = pd.read_csv(d / "trip_energy_with_meas_ref.csv")
        if "abs_err_vs_meas_kwh" not in detail.columns:
            raise ValueError(f"Missing abs_err_vs_meas_kwh in {profile}")
        for method in TESLA_METHODS:
            g = detail.loc[detail["method"] == method]
            # Seed-mean AE per trip, then mean over trips (paper neural aggregation).
            trip_ae = g.groupby("trip_id", sort=True)["abs_err_vs_meas_kwh"].mean()
            if len(trip_ae) != 17:
                raise ValueError(f"{profile}/{method}: expected 17 trips, got {len(trip_ae)}")
            n_seeds = int(g.groupby("trip_id")["seed"].nunique().max()) if method != "physics" else 1
            if method in {"pinn", "weak_mlp"} and n_seeds < 3:
                raise ValueError(f"{profile}/{method}: expected 3 seeds, got {n_seeds}")
            mae = float(trip_ae.mean())
            ref = float(summary.loc[summary["method"] == method, "mae_kwh"].iloc[0])
            if abs(mae - ref) > 1e-9:
                raise ValueError(
                    f"Tesla MAE mismatch {profile}/{method}: recomputed {mae:.12f} vs summary {ref:.12f}"
                )
            rows.append(
                {
                    "profile": profile,
                    "dataset": display,
                    "method": method,
                    "method_label": label(method) if method != "pinn" else "PINN",
                    "mae_kwh": mae,
                    "n_trips": 17,
                    "n_seeds": 3 if method in {"pinn", "weak_mlp"} else 1,
                }
            )
    return pd.DataFrame(rows)


def load_and_verify_sensitivity_L(mae_path: Path) -> pd.DataFrame:
    mae = pd.read_csv(mae_path)
    rows = []
    for L in (2, 4, 6, 8):
        key = f"L_{L}"
        hit = mae.loc[mae["sensitivity_variant"] == key, "mae_kwh"]
        if len(hit) != 1:
            raise ValueError(f"Expected one row for {key} in {mae_path}")
        rows.append({"L_kW": L, "mae_kwh": float(hit.iloc[0]), "is_reference": L == 4})
    frame = pd.DataFrame(rows)
    # Cross-check per-config summary files when present.
    root = mae_path.parent
    for _, row in frame.iterrows():
        summary = root / f"summary_L_{int(row['L_kW'])}.csv"
        if summary.exists():
            s = pd.read_csv(summary)
            ref = float(s.loc[s["method"] == "pinn", "mae"].iloc[0])
            if abs(ref - row["mae_kwh"]) > 1e-9:
                raise ValueError(f"Sensitivity L={row['L_kW']} mismatch vs summary file")
    return frame


def fig_ablation_paired_diff(paired: pd.DataFrame, path: Path) -> dict[str, Path]:
    """Horizontal paired AE difference: full PINN − comparator (kWh)."""
    apply_style()
    y = np.arange(len(paired))
    means = paired["mean_diff_kWh"].to_numpy(dtype=float)
    lo = paired["CI_low"].to_numpy(dtype=float)
    hi = paired["CI_high"].to_numpy(dtype=float)
    xerr = np.vstack([means - lo, hi - means])

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axvline(0.0, color="0.35", lw=1.2, zorder=1)
    ax.errorbar(
        means,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#222222",
        elinewidth=1.6,
        capsize=4.5,
        zorder=2,
    )
    for i, (mean, name) in enumerate(zip(means, paired["comparator"])):
        # Baselines use paper colors; within-PINN ablations use PINN blue.
        if name == "Physics Model":
            color, marker = METHOD_COLORS["physics"], METHOD_MARKERS["physics"]
        elif name == "Data-Driven MLP":
            color, marker = METHOD_COLORS["weak_mlp"], METHOD_MARKERS["weak_mlp"]
        else:
            color, marker = METHOD_COLORS["pinn"], "D"
        ax.plot(mean, y[i], marker=marker, color=color, ms=10, zorder=3, markeredgecolor="0.15", markeredgewidth=0.4)
        ax.annotate(
            f"{mean:+.3f}",
            (mean, y[i]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            va="bottom",
            fontsize=9.5,
            annotation_clip=False,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"vs {n}" for n in paired["comparator"]])
    ax.set_xlabel(r"Paired trip AE difference: Full PINN $-$ comparator (kWh)")
    ax.text(
        0.02,
        0.98,
        "← Full PINN lower error",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="0.25",
    )
    ax.text(
        0.98,
        0.98,
        "Comparator lower error →",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color="0.25",
    )
    ax.text(
        0.99,
        0.03,
        "whiskers: 95% paired bootstrap CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="0.25",
    )
    # Invert so first row (Physics) is at top.
    ax.set_ylim(len(paired) - 0.55, -0.75)
    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(False, axis="y")
    fig.tight_layout()
    return save_publication(fig, path)


def fig_tesla_meas_mae(tesla: pd.DataFrame, path: Path) -> dict[str, Path]:
    """Grouped MAE bars vs independent V×I energy (no CI — not in summary)."""
    apply_style()
    datasets = [TESLA_PROFILES[k] for k in TESLA_PROFILES]
    methods = TESLA_METHODS
    x = np.arange(len(datasets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for i, method in enumerate(methods):
        vals = []
        for ds in datasets:
            row = tesla.loc[(tesla["dataset"] == ds) & (tesla["method"] == method)]
            vals.append(float(row["mae_kwh"].iloc[0]))
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="0.15",
            linewidth=0.5,
            label=label(method) if method != "pinn" else "PINN",
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            ax.annotate(
                f"{val:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Trip-energy MAE vs measured V×I (kWh)")
    ax.set_ylim(0, max(tesla["mae_kwh"]) * 1.22)
    ax.legend(loc="upper right", ncol=1)
    ax.text(
        0.01,
        0.98,
        "17 evaluation trips / configuration; neural methods: seeds 0–1–2\n"
        "Feature-adapted (4 inputs, flat road); no significance claim vs V×I",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="0.25",
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    fig.tight_layout()
    return save_publication(fig, path)


def fig_sensitivity_L(frame: pd.DataFrame, path: Path) -> dict[str, Path]:
    """Twizy residual-bound sensitivity: point MAE only (no CI in archive)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    L = frame["L_kW"].to_numpy(dtype=float)
    mae = frame["mae_kwh"].to_numpy(dtype=float)
    ax.plot(L, mae, color="0.55", lw=1.4, zorder=1)
    for _, row in frame.iterrows():
        is_ref = bool(row["is_reference"])
        ax.plot(
            row["L_kW"],
            row["mae_kwh"],
            marker="o" if is_ref else "s",
            ms=11 if is_ref else 9,
            color=METHOD_COLORS["pinn"] if is_ref else "#333333",
            markeredgecolor="0.15",
            markeredgewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            f"{row['mae_kwh']:.3f}",
            (row["L_kW"], row["mae_kwh"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
        )
    # Mark accepted reference vs lower point estimate at L=6.
    ref_mae = float(frame.loc[frame["L_kW"] == 4, "mae_kwh"].iloc[0])
    l6_mae = float(frame.loc[frame["L_kW"] == 6, "mae_kwh"].iloc[0])
    ax.axvline(4, color=METHOD_COLORS["pinn"], ls="--", lw=1.2, alpha=0.7, zorder=0)
    ax.annotate(
        "Accepted reference\n(L = 4 kW)",
        xy=(4, ref_mae),
        xytext=(2.15, ref_mae + 0.045),
        fontsize=9,
        color=METHOD_COLORS["pinn"],
        arrowprops=dict(arrowstyle="->", color=METHOD_COLORS["pinn"], lw=1.0),
    )
    ax.annotate(
        "Lower point MAE\n(not adopted)",
        xy=(6, l6_mae),
        xytext=(6.7, l6_mae - 0.035),
        fontsize=9,
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0),
    )
    ax.set_xlabel("Residual bound L (kW)")
    ax.set_ylabel("Trip-energy MAE (kWh)")
    ax.set_xticks([2, 4, 6, 8])
    ax.set_ylim(min(mae) - 0.05, max(mae) + 0.06)
    ax.grid(True, axis="both", alpha=0.25)
    fig.tight_layout()
    return save_publication(fig, path)


def fig_ablation_tesla_two_panel(
    paired: pd.DataFrame,
    tesla: pd.DataFrame,
    path: Path,
) -> dict[str, Path]:
    """Compact two-panel manuscript candidate: ablation diffs + Tesla V×I MAE."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panel A — full paired set (baselines + ablations) with short labels.
    use = paired.reset_index(drop=True)
    ax = axes[0]
    y = np.arange(len(use))
    means = use["mean_diff_kWh"].to_numpy(dtype=float)
    lo = use["CI_low"].to_numpy(dtype=float)
    hi = use["CI_high"].to_numpy(dtype=float)
    xerr = np.vstack([means - lo, hi - means])
    ax.axvline(0.0, color="0.35", lw=1.1, zorder=1)
    ax.errorbar(means, y, xerr=xerr, fmt="none", ecolor="#222222", elinewidth=1.4, capsize=3.5, zorder=2)
    for i, (mean, name) in enumerate(zip(means, use["comparator"])):
        if name == "Physics Model":
            color, marker = METHOD_COLORS["physics"], METHOD_MARKERS["physics"]
        elif name == "Data-Driven MLP":
            color, marker = METHOD_COLORS["weak_mlp"], METHOD_MARKERS["weak_mlp"]
        else:
            color, marker = METHOD_COLORS["pinn"], "D"
        ax.plot(mean, y[i], marker=marker, color=color, ms=8.5, zorder=3, markeredgecolor="0.15", markeredgewidth=0.4)
    short = {
        "Physics Model": "vs Physics",
        "Data-Driven MLP": "vs MLP",
        "PINN full-trip only": "vs full-trip only",
        "PINN no conservation": "vs no conservation",
        "PINN no state loss": "vs no state",
        "PINN no residual prior": "vs no prior",
    }
    ax.set_yticks(y)
    ax.set_yticklabels([short[n] for n in use["comparator"]])
    ax.set_xlabel(r"Full PINN $-$ comparator AE (kWh)")
    ax.set_title("(a) HELECAR-D ablation (paired)", fontsize=11)
    ax.text(0.02, 0.02, "← PINN better", transform=ax.transAxes, fontsize=8.5, color="0.3")
    ax.set_ylim(len(use) - 0.5, -0.65)
    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(False, axis="y")

    # Panel B — Tesla MAE bars.
    ax = axes[1]
    datasets = [TESLA_PROFILES[k] for k in TESLA_PROFILES]
    x = np.arange(len(datasets))
    width = 0.24
    for i, method in enumerate(TESLA_METHODS):
        vals = [
            float(tesla.loc[(tesla["dataset"] == ds) & (tesla["method"] == method), "mae_kwh"].iloc[0])
            for ds in datasets
        ]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="0.15",
            linewidth=0.5,
            label=label(method) if method != "pinn" else "PINN",
        )
        for bar, val in zip(bars, vals):
            ax.annotate(
                f"{val:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("MAE vs V×I (kWh)")
    ax.set_title("(b) Tesla feature-adapted (corrected)", fontsize=11)
    ax.set_ylim(0, max(tesla["mae_kwh"]) * 1.28)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.text(
        0.02,
        0.02,
        "n=17 trips/config; seeds 0–1–2",
        transform=ax.transAxes,
        fontsize=8,
        color="0.3",
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")

    fig.tight_layout()
    return save_publication(fig, path)
