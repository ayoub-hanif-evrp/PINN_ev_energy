"""Conference-paper figures: PNG only, 300 dpi, white background."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from data.preprocessing import ProcessedTrip, elapsed_seconds
from evaluation.bootstrap import trip_absolute_errors
from evaluation.names import (
    FEASIBILITY_METHODS,
    METHOD_COLORS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    PARITY_METHODS,
    SCARCITY_METHODS,
    label,
)
from evaluation.trip_agg import trip_level_predictions
from plotting.style import apply_style, save_png


def _c(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


def fig01_method(path: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f4f4"):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            linewidth=0.95,
            edgecolor="#222222",
            facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.4)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=0.95, color="#222222")
        )

    box(0.03, 0.70, 0.22, 0.22, "Driving / environment\nspeed, acceleration, grade,\ntemperature, humidity,\nwind, traffic, speed limit", "#eef3f8")
    box(0.32, 0.80, 0.22, 0.14, "Longitudinal physics\n$P_\\mathrm{physics}(t)$", "#f7f1e8")
    box(0.32, 0.62, 0.22, 0.14, "Neural correction\n$\\delta P=\\ell\\,\\tanh(\\mathrm{NN}(x_t))$", "#e8f0e8")
    box(0.61, 0.68, 0.35, 0.18, "$\\hat P(t)=P_\\mathrm{physics}(t)+\\delta P(t)$\nNo instantaneous battery-power labels", "#e8eef7")
    box(0.61, 0.40, 0.35, 0.18, "Trapezoidal energy integration\n$\\Delta E_i=0.5(\\hat P_i+\\hat P_{i+1})\\,\\Delta t_i/3600$", "#f4f4f4")
    box(0.03, 0.08, 0.45, 0.22, "Multi-window SoC energy supervision\n$E_\\mathrm{obs}(a,b)=E_\\mathrm{batt}(\\mathrm{SOC}[a]-\\mathrm{SOC}[b])/100$\nvs $E_\\mathrm{pred}(a,b)=\\int_a^b \\hat P\\,dt$", "#f4f4f4")
    box(0.52, 0.08, 0.44, 0.22, "Battery-energy conservation\n$r_i=\\hat D[i+1]-\\hat D[i]-100\\,\\Delta E_i/E_\\mathrm{batt}$\nDiscrete-time PINN (not a PDE PINN)", "#f8ecec")

    arrow(0.25, 0.88, 0.32, 0.88)
    arrow(0.25, 0.76, 0.32, 0.70)
    arrow(0.54, 0.87, 0.61, 0.80)
    arrow(0.54, 0.69, 0.61, 0.76)
    arrow(0.78, 0.68, 0.78, 0.58)
    arrow(0.70, 0.40, 0.48, 0.30)
    arrow(0.78, 0.40, 0.74, 0.30)
    save_png(fig, path)


def fig02_observed_vs_predicted(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    trip = trip_level_predictions(pred)
    methods = [m for m in PARITY_METHODS if m in set(trip["method"])]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.2), sharex=True, sharey=True)
    lo = min(trip["observed_energy_kwh"].min(), trip["predicted_energy_kwh"].min())
    hi = max(trip["observed_energy_kwh"].max(), trip["predicted_energy_kwh"].max())
    pad = 0.08 * (hi - lo)
    lim = (lo - pad, hi + pad)
    for ax, method in zip(axes.ravel(), methods):
        g = trip.loc[trip["method"] == method]
        ax.plot(lim, lim, color="0.55", lw=1.0, zorder=0)
        ax.scatter(
            g["observed_energy_kwh"],
            g["predicted_energy_kwh"],
            s=36,
            c=_c(method),
            marker=METHOD_MARKERS.get(method, "o"),
            edgecolors="0.15",
            linewidths=0.4,
            zorder=2,
        )
        err = trip_absolute_errors(pred, method)
        mae = float(err.mean())
        ax.text(0.04, 0.96, f"{label(method)}\nMAE {mae:.3f} kWh", transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
    axes[1, 0].set_xlabel("Observed trip energy (kWh)")
    axes[1, 1].set_xlabel("Observed trip energy (kWh)")
    axes[0, 0].set_ylabel("Predicted trip energy (kWh)")
    axes[1, 0].set_ylabel("Predicted trip energy (kWh)")
    fig.tight_layout()
    save_png(fig, path)


def fig03_trip_errors(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    methods = [m for m in PARITY_METHODS if m in set(pred["method"])]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    rng = np.random.default_rng(0)
    for i, method in enumerate(methods):
        err = trip_absolute_errors(pred, method).sort_index()
        x = np.full(len(err), i, dtype=float) + rng.uniform(-0.08, 0.08, len(err))
        ax.scatter(
            x,
            err.to_numpy(),
            s=28,
            c=_c(method),
            marker=METHOD_MARKERS.get(method, "o"),
            zorder=3,
            edgecolors="0.15",
            linewidths=0.35,
        )
        q = err.quantile([0.25, 0.5, 0.75])
        ax.hlines(q[0.5], i - 0.18, i + 0.18, color="0.1", lw=1.4, zorder=4)
        ax.vlines(i, q[0.25], q[0.75], color="0.1", lw=1.1, zorder=4)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([label(m) for m in methods])
    ax.set_ylabel("Absolute held-out trip-energy error (kWh)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_png(fig, path)


def fig04_data_scarcity(raw: pd.DataFrame, path: Path) -> None:
    from evaluation.final_tables import complete_scarcity_frame

    raw = complete_scarcity_frame(raw)
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    if "physics" in set(raw["method"]):
        phys = float(raw.loc[raw["method"] == "physics", "absolute_error_kwh"].mean())
        ax.axhline(phys, color=_c("physics"), ls="--", lw=1.1, label="Physics (fixed)")
    for method in SCARCITY_METHODS:
        g = raw.loc[raw["method"] == method]
        if g.empty:
            continue
        run = g.groupby(["n_train", "seed", "repeat"], dropna=False)["absolute_error_kwh"].mean().reset_index()
        stats = run.groupby("n_train")["absolute_error_kwh"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            stats["n_train"],
            stats["mean"],
            yerr=stats["std"].fillna(0.0),
            color=_c(method),
            marker=METHOD_MARKERS.get(method, "o"),
            ls=METHOD_LINESTYLES.get(method, "-"),
            capsize=3,
            label=label(method),
        )
    ax.set_xlabel("Number of training trips")
    ax.set_ylabel("Held-out energy MAE (kWh)")
    ax.set_xticks([3, 5, 8, 12, 16])
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    fig.tight_layout()
    save_png(fig, path)


def fig05_feasibility(table: pd.DataFrame, path: Path) -> None:
    apply_style()
    if table.empty:
        return
    method_col = "method" if "method" in table.columns else "Method"
    reserve_col = "reserve_soc_pct" if "reserve_soc_pct" in table.columns else "reserve_SoC_pct"
    false_col = "false_safe" if "false_safe" in table.columns else "false_safe_fraction"
    cons_col = "overly_conservative" if "overly_conservative" in table.columns else "overly_conservative_fraction"

    def _key(val: str) -> str:
        val = str(val).lower()
        return {"physics": "physics", "pinn": "pinn"}.get(val, val)

    table = table.copy()
    table["_m"] = table[method_col].map(_key)
    methods = [m for m in FEASIBILITY_METHODS if m in set(table["_m"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True)
    for method in methods:
        g = table.loc[table["_m"] == method].groupby(reserve_col, as_index=False)[[false_col, cons_col]].mean()
        axes[0].plot(
            g[reserve_col],
            100.0 * g[false_col],
            marker=METHOD_MARKERS.get(method, "o"),
            color=_c(method),
            ls=METHOD_LINESTYLES.get(method, "-"),
            label=label(method),
        )
        axes[1].plot(
            g[reserve_col],
            100.0 * g[cons_col],
            marker=METHOD_MARKERS.get(method, "o"),
            color=_c(method),
            ls=METHOD_LINESTYLES.get(method, "-"),
            label=label(method),
        )
    axes[0].set_ylabel("False-safe rate (%)")
    axes[1].set_ylabel("Overly conservative rate (%)")
    for ax in axes:
        ax.set_xlabel("Battery reserve (%)")
        ax.set_xticks([5, 10, 15, 20])
        ax.set_ylim(0, None)
    axes[0].legend(loc="best")
    fig.tight_layout()
    save_png(fig, path)


def fig06_soc_reconstruction(trips: list[ProcessedTrip], rec_dir: Path, path: Path) -> None:
    apply_style()
    from experiment.data import representative_test_ids

    ids = representative_test_ids(trips)
    by_id = {t.trip_id: t for t in trips}
    chosen = [i for i in ids if i in by_id and (rec_dir / f"{i}__pinn__seed0.npz").exists()]
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 1, figsize=(7.6, 2.35 * len(chosen)), sharex=False)
    if len(chosen) == 1:
        axes = [axes]
    for ax, trip_id in zip(axes, chosen):
        trip = by_id[trip_id]
        data = np.load(rec_dir / f"{trip_id}__pinn__seed0.npz")
        tmin = elapsed_seconds(trip.frame["dt_s"].to_numpy(dtype=float)) / 60.0
        ax.plot(tmin, trip.frame["soc"], color="0.15", lw=1.1, label="Observed quantized SoC")
        ax.plot(tmin, data["soc_hat"], color=_c("pinn"), lw=1.1, ls="--", label="SoC from integrated PINN power")
        ax.set_ylabel("SoC (%)")
        ax.text(0.01, 0.06, f"{trip_id} ({trip.trajectory})", transform=ax.transAxes, fontsize=8)
        if ax is axes[0]:
            ax.legend(loc="upper right")
        ax.set_xlabel("Time (min)")
    fig.tight_layout()
    save_png(fig, path)
