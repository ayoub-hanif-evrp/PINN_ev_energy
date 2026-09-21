"""Paper figures: PNG only, 300 dpi, white background."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from evaluation.bootstrap import bootstrap_mean_ci, trip_absolute_errors
from evaluation.names import (
    ABLATION_ORDER,
    FEASIBILITY_METHODS,
    METHOD_COLORS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    PARITY_METHODS,
    SCARCITY_METHODS,
    label,
)
from evaluation.power_plausibility import battery_caps
from evaluation.trip_agg import trip_level_predictions
from plotting.style import apply_style, save_png
from data.preprocessing import ProcessedTrip, elapsed_seconds


def _c(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


def fig01_method_overview(path: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f4f4"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02", linewidth=0.9, edgecolor="#222222", facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=0.9, color="#222222"))

    box(0.02, 0.72, 0.18, 0.18, "Driving / environment\ntelemetry\n(no measured SoC)", "#eef3f8")
    box(0.27, 0.78, 0.18, 0.14, "Longitudinal model\n$P_\\mathrm{physics}(t)$", "#f7f1e8")
    box(0.27, 0.58, 0.18, 0.14, "Residual NN\n$\\delta P(t)=\\ell\\,\\tanh(\\cdot)$", "#e8f0e8")
    box(0.52, 0.66, 0.20, 0.16, "$\\hat P(t)=P_\\mathrm{physics}+\\delta P$\nno $P_\\mathrm{batt}$ labels", "#e8eef7")
    box(0.78, 0.74, 0.20, 0.16, "Trapezoidal\nintegration\n$E[a:b]=\\mathrm{prefix}[b]-\\mathrm{prefix}[a]$", "#f4f4f4")
    box(0.78, 0.50, 0.20, 0.16, "Multi-scale windows\nvs $E_\\mathrm{obs}=E_\\mathrm{batt}\\Delta\\mathrm{SOC}/100$", "#f4f4f4")
    box(0.27, 0.18, 0.22, 0.22, "Causal state head\n$\\hat D(t)=D_\\mathrm{raw}(t)-D_\\mathrm{raw}(0)$\nprogress: time, distance", "#f8ecec")
    box(0.56, 0.18, 0.22, 0.22, "Discrete conservation\n$r_i=\\hat D_{i+1}-\\hat D_i$\n$-100\\,\\Delta E_i/E_\\mathrm{batt}$", "#f8ecec")
    box(0.02, 0.22, 0.18, 0.16, "SoC is a label\nnot a power-branch\ninput", "#f3f3f3")
    box(0.78, 0.18, 0.20, 0.16, "Trip energy\n$\\int \\hat P\\,dt$\nprimary inference", "#e8eef7")

    arrow(0.20, 0.84, 0.27, 0.84)
    arrow(0.20, 0.78, 0.27, 0.66)
    arrow(0.45, 0.85, 0.52, 0.76)
    arrow(0.45, 0.65, 0.52, 0.72)
    arrow(0.72, 0.74, 0.78, 0.82)
    arrow(0.88, 0.74, 0.88, 0.66)
    arrow(0.20, 0.30, 0.27, 0.30)
    arrow(0.49, 0.29, 0.56, 0.29)
    arrow(0.72, 0.74, 0.67, 0.40)
    arrow(0.78, 0.29, 0.78, 0.29)
    arrow(0.72, 0.26, 0.78, 0.26)
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


def fig03_trip_absolute_errors(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    methods = [m for m in PARITY_METHODS if m in set(pred["method"])]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    rng = np.random.default_rng(0)
    for i, method in enumerate(methods):
        err = trip_absolute_errors(pred, method).sort_index()
        x = np.full(len(err), i, dtype=float) + rng.uniform(-0.08, 0.08, len(err))
        ax.scatter(x, err.to_numpy(), s=28, c=_c(method), marker=METHOD_MARKERS.get(method, "o"), zorder=3, edgecolors="0.15", linewidths=0.35)
        q = err.quantile([0.25, 0.5, 0.75])
        ax.hlines(q[0.5], i - 0.18, i + 0.18, color="0.1", lw=1.4, zorder=4)
        ax.vlines(i, q[0.25], q[0.75], color="0.1", lw=1.1, zorder=4)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([label(m) for m in methods])
    ax.set_ylabel("Trip absolute energy error (kWh)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_png(fig, path)


def fig04_ablation(pred: pd.DataFrame, path: Path) -> None:
    apply_style()
    methods = [m for m in (["physics"] + ABLATION_ORDER) if m in set(pred["method"])]
    means, lo, hi = [], [], []
    for method in methods:
        stats = bootstrap_mean_ci(trip_absolute_errors(pred, method).to_numpy(dtype=float))
        means.append(stats["mean"])
        lo.append(stats["ci_low"])
        hi.append(stats["ci_high"])
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    xerr = np.vstack([np.array(means) - np.array(lo), np.array(hi) - np.array(means)])
    ax.errorbar(means, y, xerr=xerr, fmt="none", ecolor="#222222", capsize=3, zorder=2)
    for i, method in enumerate(methods):
        ax.plot(means[i], y[i], marker=METHOD_MARKERS.get(method, "o"), color=_c(method), ms=8)
    ax.set_yticks(y)
    ax.set_yticklabels([label(m) for m in methods])
    ax.set_xlabel("Trip-level MAE (kWh) with 95% bootstrap interval")
    ax.invert_yaxis()
    fig.tight_layout()
    save_png(fig, path)


def fig05_data_scarcity(raw: pd.DataFrame, path: Path) -> None:
    from evaluation.final_tables import complete_scarcity_frame

    raw = complete_scarcity_frame(raw)
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    if "physics" in set(raw["method"]):
        phys = raw.loc[raw["method"] == "physics", "absolute_error_kwh"].mean()
        ax.axhline(phys, color=_c("physics"), ls="--", lw=1.1, label="Physics (no fitting)")
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
    ax.set_ylabel("Held-out trip energy MAE (kWh)")
    ax.set_xticks([3, 5, 8, 12, 16])
    ax.set_yscale("log")
    ax.legend(loc="best")
    fig.tight_layout()
    save_png(fig, path)


def fig06_cross_trajectory(raw: pd.DataFrame, path: Path) -> None:
    apply_style()
    held_col = "heldout_trajectory" if "heldout_trajectory" in raw.columns else "trajectory"
    methods = [m for m in PARITY_METHODS if m in set(raw["method"])]
    held = [h for h in ("T1", "T2", "T3") if h in set(raw[held_col])]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.08
    for i, method in enumerate(methods):
        xs, ys = [], []
        for j, h in enumerate(held):
            g = raw.loc[(raw["method"] == method) & (raw[held_col] == h)]
            trip = g.groupby("trip_id")["absolute_error_kwh"].mean()
            xs.append(j + (i - 1.5) * 0.18)
            ys.append(float(trip.mean()) if len(trip) else np.nan)
            ax.scatter(
                np.full(len(trip), j + (i - 1.5) * 0.18) + np.linspace(-width, width, len(trip)),
                trip.to_numpy(),
                s=18,
                color=_c(method),
                alpha=0.55,
                zorder=2,
            )
        ax.plot(xs, ys, marker=METHOD_MARKERS.get(method, "o"), color=_c(method), ls=METHOD_LINESTYLES.get(method, "-"), label=label(method), zorder=3)
    ax.set_xticks(range(len(held)))
    ax.set_xticklabels([f"Hold out {h}" for h in held])
    ax.set_ylabel("Trip energy MAE (kWh)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    fig.tight_layout()
    save_png(fig, path)


def fig07_soc_reconstruction(trips: list[ProcessedTrip], rec_dir: Path, path: Path) -> None:
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
        ax.plot(tmin, trip.frame["soc"], color="0.15", lw=1.1, label="Observed SoC")
        ax.plot(tmin, data["soc_hat"], color=_c("pinn"), lw=1.1, ls="--", label="PINN reconstructed SoC")
        phys = rec_dir / f"{trip_id}__physics__seed0.npz"
        # Physics reconstruction from p_phy in the PINN file if present.
        if "p_phy" in data.files:
            from evaluation.metrics import reconstruct_soc
            from physics.parameters import load_vehicle_parameters
            # Capacity 6 kWh is the protocol value; reconstruction is diagnostic.
            rec = reconstruct_soc(float(trip.frame["soc"].iloc[0]), data["p_phy"], trip.frame["dt_s"].to_numpy(dtype=float), 6.0)
            ax.plot(tmin, rec["soc_hat"], color=_c("physics"), lw=0.9, ls=":", label="Physics reconstructed SoC")
        ax.set_ylabel("SoC (%)")
        ax.text(0.01, 0.06, f"{trip_id} ({trip.trajectory})", transform=ax.transAxes, fontsize=8)
        if ax is axes[0]:
            ax.legend(loc="upper right", ncols=1)
        ax.set_xlabel("Time (min)")
    fig.tight_layout()
    save_png(fig, path)


def fig08_power_plausibility(trips: list[ProcessedTrip], rec_dir: Path, params, path: Path) -> None:
    apply_style()
    from experiment.data import representative_test_ids
    from evaluation.power_plausibility import physics_arrays

    ids = representative_test_ids(trips)
    by_id = {t.trip_id: t for t in trips}
    chosen = [i for i in ids if i in by_id and (rec_dir / f"{i}__pinn__seed0.npz").exists()]
    if not chosen:
        return
    caps = battery_caps(params)
    fig, axes = plt.subplots(len(chosen), 2, figsize=(9.6, 2.5 * len(chosen)), sharex=False)
    if len(chosen) == 1:
        axes = np.array([axes])
    for row, trip_id in enumerate(chosen):
        trip = by_id[trip_id]
        data = np.load(rec_dir / f"{trip_id}__pinn__seed0.npz")
        tmin = elapsed_seconds(trip.frame["dt_s"].to_numpy(dtype=float)) / 60.0
        p_hat = np.asarray(data["p_hat"], dtype=float)
        p_phy = np.asarray(data["p_phy"], dtype=float) if "p_phy" in data.files else physics_arrays(trip, params, False)["p_battery_kw"]
        ax = axes[row, 0]
        ax.plot(tmin, p_phy, color=_c("physics"), lw=0.7, label="Physics $P(t)$")
        ax.plot(tmin, p_hat, color=_c("pinn"), lw=0.7, label="PINN $\\hat P(t)$")
        ax.axhline(caps["battery_traction_kw"], color="0.2", ls="--", lw=0.8)
        ax.axhline(caps["battery_regen_kw"], color="0.2", ls=":", lw=0.8)
        ax.set_ylabel("Battery power (kW)")
        ax.text(0.01, 0.92, trip_id, transform=ax.transAxes, fontsize=7.5)
        if row == 0:
            ax.legend(loc="upper right", fontsize=7.5)
        ax.set_xlabel("Time (min)")
        axh = axes[row, 1]
        lo = min(float(np.min(p_phy)), float(np.min(p_hat)), caps["battery_regen_kw"] - 1)
        hi = max(float(np.max(p_phy)), float(np.max(p_hat)), caps["battery_traction_kw"] + 1)
        bins = np.linspace(lo, hi, 40)
        axh.hist(p_phy, bins=bins, color=_c("physics"), alpha=0.55, label="Physics")
        axh.hist(p_hat, bins=bins, color=_c("pinn"), alpha=0.45, label="PINN")
        axh.axvline(caps["battery_traction_kw"], color="0.2", ls="--", lw=0.8)
        axh.axvline(caps["battery_regen_kw"], color="0.2", ls=":", lw=0.8)
        axh.set_xlabel("Battery power (kW)")
        axh.set_ylabel("Samples")
        if row == 0:
            axh.legend(fontsize=7.5)
    fig.tight_layout()
    save_png(fig, path)


def fig09_feasibility(table: pd.DataFrame, path: Path) -> None:
    apply_style()
    if table.empty:
        return
    method_col = "method" if "method" in table.columns else "Method"
    reserve_col = "reserve_soc_pct" if "reserve_soc_pct" in table.columns else "reserve_SoC_pct"
    false_col = "false_safe" if "false_safe" in table.columns else "false_safe_fraction"
    cons_col = "overly_conservative" if "overly_conservative" in table.columns else "overly_conservative_fraction"
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True)
    methods = [m for m in FEASIBILITY_METHODS if m in set(table[method_col].str.lower() if table[method_col].dtype == object else table[method_col])]
    # Normalize method names to keys.
    def _key(val: str) -> str:
        val = str(val).lower()
        return {"constant": "constant", "physics": "physics", "pinn": "pinn"}.get(val, val)

    table = table.copy()
    table["_m"] = table[method_col].map(_key)
    methods = [m for m in FEASIBILITY_METHODS if m in set(table["_m"])]
    for method in methods:
        g = table.loc[table["_m"] == method].groupby(reserve_col, as_index=False)[[false_col, cons_col]].mean()
        axes[0].plot(g[reserve_col], 100.0 * g[false_col], marker=METHOD_MARKERS.get(method, "o"), color=_c(method), ls=METHOD_LINESTYLES.get(method, "-"), label=label(method))
        axes[1].plot(g[reserve_col], 100.0 * g[cons_col], marker=METHOD_MARKERS.get(method, "o"), color=_c(method), ls=METHOD_LINESTYLES.get(method, "-"), label=label(method))
    axes[0].set_ylabel("False-safe rate (%)")
    axes[1].set_ylabel("Overly conservative rate (%)")
    for ax in axes:
        ax.set_xlabel("Reserve SoC (%)")
        ax.set_xticks([5, 10, 15, 20])
        ax.set_ylim(0, None)
    axes[0].legend(loc="best")
    fig.tight_layout()
    save_png(fig, path)


def fig10_training_diagnostics(history_csv: Path, path: Path, title: str = "") -> None:
    if not history_csv.exists():
        return
    apply_style()
    hist = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    if "l_window" in hist.columns:
        axes[0].plot(hist["epoch"], hist["l_window"], color="#0072b2", label="window")
    if "l_dynamics" in hist.columns:
        axes[0].plot(hist["epoch"], hist["l_dynamics"], color="#d55e00", label="dynamics")
    if "l_state" in hist.columns:
        axes[0].plot(hist["epoch"], hist["l_state"], color="#009e73", label="state")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss term")
    axes[0].set_yscale("log")
    axes[0].legend()
    if "val_window_mae_kwh" in hist.columns and hist["val_window_mae_kwh"].notna().any():
        axes[1].plot(hist["epoch"], hist["val_window_mae_kwh"], color="#222222")
        axes[1].set_ylabel("Validation window MAE (kWh)")
    elif "train_loss" in hist.columns:
        axes[1].plot(hist["epoch"], hist["train_loss"], color="#222222")
        axes[1].set_ylabel("Training loss")
    axes[1].set_xlabel("Epoch")
    fig.tight_layout()
    save_png(fig, path)
