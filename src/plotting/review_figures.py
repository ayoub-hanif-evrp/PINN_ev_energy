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

ABLATION_SHORT = {
    "Physics Model": "Physics",
    "Data-Driven MLP": "MLP",
    "PINN full-trip only": "Full-trip only",
    "PINN no conservation": "No conservation",
    "PINN no state loss": "No state loss",
    "PINN no residual prior": "No residual prior",
}

TESLA_METHODS = ["physics", "weak_mlp", "pinn"]
TESLA_PROFILES = {
    "tesla_bev1_loto_corrected": "BEV1",
    "tesla_bev2_loto_corrected": "BEV2",
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


def _marker_style(name: str) -> tuple[str, str]:
    if name == "Physics Model":
        return METHOD_COLORS["physics"], METHOD_MARKERS["physics"]
    if name == "Data-Driven MLP":
        return METHOD_COLORS["weak_mlp"], METHOD_MARKERS["weak_mlp"]
    return METHOD_COLORS["pinn"], "D"


def load_and_verify_ablation_paired(path: Path) -> pd.DataFrame:
    paired = pd.read_csv(path)
    required = {"comparison", "mean_diff_kWh", "CI_low", "CI_high"}
    missing = required - set(paired.columns)
    if missing:
        raise ValueError(f"Ablation paired table missing columns: {sorted(missing)}")
    paired = paired.copy()
    paired["comparator"] = paired["comparison"].map(_parse_comparator)
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
    return paired.set_index("comparator").loc[ABLATION_ORDER].reset_index()


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

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
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
        color, marker = _marker_style(name)
        ax.plot(
            mean,
            y[i],
            marker=marker,
            color=color,
            ms=10,
            zorder=3,
            markeredgecolor="0.15",
            markeredgewidth=0.4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([ABLATION_SHORT[n] for n in paired["comparator"]])
    ax.set_xlabel("Paired MAE difference (kWh)")
    ax.set_ylim(len(paired) - 0.55, -0.55)
    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(False, axis="y")
    fig.tight_layout()
    return save_publication(fig, path)


def fig_tesla_meas_mae(tesla: pd.DataFrame, path: Path) -> dict[str, Path]:
    """Grouped MAE bars vs independent V×I energy (no CI — not in summary)."""
    apply_style()
    datasets = [TESLA_PROFILES[k] for k in TESLA_PROFILES]
    x = np.arange(len(datasets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
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
    ax.set_ylabel("Trip-energy MAE (kWh)")
    ax.set_ylim(0, float(tesla["mae_kwh"].max()) * 1.18)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    fig.tight_layout()
    return save_publication(fig, path)


def fig_sensitivity_L(frame: pd.DataFrame, path: Path) -> dict[str, Path]:
    """Twizy residual-bound sensitivity: point MAE only (no CI in archive)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    L = frame["L_kW"].to_numpy(dtype=float)
    mae = frame["mae_kwh"].to_numpy(dtype=float)
    ax.plot(L, mae, color="0.55", lw=1.4, zorder=1)
    for _, row in frame.iterrows():
        is_ref = bool(row["is_reference"])
        ax.plot(
            row["L_kW"],
            row["mae_kwh"],
            marker="o" if is_ref else "s",
            ms=11 if is_ref else 8.5,
            color=METHOD_COLORS["pinn"] if is_ref else "#333333",
            markeredgecolor="0.15",
            markeredgewidth=0.5,
            zorder=3,
            label="Reference (L = 4 kW)" if is_ref else None,
        )
        # Place labels above points; shift L=8 slightly left to avoid the edge.
        x_off = -6 if int(row["L_kW"]) == 8 else 0
        ax.annotate(
            f"{row['mae_kwh']:.3f}",
            (row["L_kW"], row["mae_kwh"]),
            textcoords="offset points",
            xytext=(x_off, 10),
            ha="center",
            va="bottom",
            fontsize=10,
            color="0.15",
            annotation_clip=False,
        )
    ax.set_xlabel("Residual bound L (kW)")
    ax.set_ylabel("Trip-energy MAE (kWh)")
    ax.set_xticks([2, 4, 6, 8])
    ax.set_ylim(float(mae.min()) - 0.04, float(mae.max()) + 0.055)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
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
    # Springer single-column ~122 mm; two-panel across ~ page width ≈ 170 mm ≈ 6.7 in
    # Use a readable double-column width for LNCS.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [1.05, 1.0]})

    use = paired.reset_index(drop=True)
    ax = axes[0]
    y = np.arange(len(use))
    means = use["mean_diff_kWh"].to_numpy(dtype=float)
    lo = use["CI_low"].to_numpy(dtype=float)
    hi = use["CI_high"].to_numpy(dtype=float)
    xerr = np.vstack([means - lo, hi - means])
    ax.axvline(0.0, color="0.35", lw=1.1, zorder=1)
    ax.errorbar(means, y, xerr=xerr, fmt="none", ecolor="#222222", elinewidth=1.3, capsize=3.2, zorder=2)
    for i, (mean, name) in enumerate(zip(means, use["comparator"])):
        color, marker = _marker_style(name)
        ax.plot(
            mean,
            y[i],
            marker=marker,
            color=color,
            ms=8,
            zorder=3,
            markeredgecolor="0.15",
            markeredgewidth=0.4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([ABLATION_SHORT[n] for n in use["comparator"]])
    ax.set_xlabel("Paired MAE difference (kWh)")
    ax.set_title("(a) HELECAR-D ablation", fontsize=10)
    ax.set_ylim(len(use) - 0.5, -0.5)
    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(False, axis="y")

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
    ax.set_ylabel("Trip-energy MAE (kWh)")
    ax.set_title("(b) Tesla Model Y validation", fontsize=10)
    ax.set_ylim(0, float(tesla["mae_kwh"].max()) * 1.22)
    ax.legend(fontsize=8, loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")

    fig.tight_layout()
    return save_publication(fig, path)
