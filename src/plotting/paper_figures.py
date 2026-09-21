"""Publication figures from completed experiment CSVs/NPZs only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.preprocessing import ProcessedTrip, elapsed_seconds


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def figure_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    boxes = [
        (0.02, "Telemetry\n(no SoC)"),
        (0.20, "Analytical\nphysics P_phy"),
        (0.38, "Residual NN\nδP = ℓ tanh(·)"),
        (0.56, "P_hat =\nP_phy + δP"),
        (0.72, "Trapezoidal\nintegration"),
        (0.88, "SoC / window\nconstraints"),
    ]
    for x, text in boxes:
        ax.add_patch(plt.Rectangle((x, 0.35), 0.14, 0.4, fill=False, lw=1.2))
        ax.text(x + 0.07, 0.55, text, ha="center", va="center", fontsize=8)
        if x < 0.8:
            ax.annotate("", xy=(x + 0.16, 0.55), xytext=(x + 0.14, 0.55), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Discrete-time PINN energy model (not an autodiff PDE PINN)")
    _save(fig, path)


def figure_representative_trip(trip: ProcessedTrip, path: Path) -> None:
    df = trip.frame
    t = elapsed_seconds(df["dt_s"].to_numpy(dtype=float))
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, df["speed_mps"] * 3.6, lw=0.8, color="#1f4e79")
    axes[0].set_ylabel("Speed (km/h)")
    axes[1].plot(t, df["alt_m"], lw=0.8, color="#b35c00", label="altitude")
    axg = axes[1].twinx()
    axg.plot(t, df["grade"] * 100.0, lw=0.7, color="#6b4c9a", alpha=0.8, label="grade")
    axes[1].set_ylabel("Altitude (m)")
    axg.set_ylabel("Grade (%)")
    axes[2].plot(t, df["soc"], lw=0.8, color="#2a7f62")
    axes[2].set_ylabel("Observed SoC (%)")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title(f"Representative trip {trip.trip_id} ({trip.trajectory})")
    fig.tight_layout()
    _save(fig, path)


def figure_soc_transitions(combined: dict, path: Path) -> None:
    values = combined.get("transition_values") or []
    counts = combined.get("transition_counts") or []
    fig, ax = plt.subplots(figsize=(8, 4))
    if values:
        ax.bar([str(v) for v in values[:25]], counts[:25], color="#3b6d99")
        ax.set_xlabel("|ΔSoC| (percentage points)")
        ax.set_ylabel("Count")
        ax.set_title(f"SoC transition / quantization distribution (empirical q={combined.get('q')})")
        ax.tick_params(axis="x", rotation=45)
    _save(fig, path)


def figure_obs_vs_pred(table: pd.DataFrame, path: Path, methods: list[str] | None = None) -> None:
    methods = methods or ["constant", "physics", "elasticnet", "weak_mlp", "pinn"]
    sub = table.loc[table["method"].isin(methods)]
    if sub.empty:
        return
    agg = sub.groupby(["trip_id", "method"], as_index=False)[["observed_energy_kwh", "predicted_energy_kwh"]].mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    lo = min(agg["observed_energy_kwh"].min(), agg["predicted_energy_kwh"].min())
    hi = max(agg["observed_energy_kwh"].max(), agg["predicted_energy_kwh"].max())
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=1)
    for method, g in agg.groupby("method"):
        ax.scatter(g["observed_energy_kwh"], g["predicted_energy_kwh"], label=method, s=28, alpha=0.85)
    ax.set_xlabel("Observed SoC-derived trip energy (kWh)")
    ax.set_ylabel("Predicted trip energy (kWh)")
    ax.set_title("Held-out total trip energy")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    _save(fig, path)


def figure_per_trip_error(table: pd.DataFrame, path: Path, methods: list[str] | None = None) -> None:
    methods = methods or ["constant", "physics", "elasticnet", "weak_mlp", "pinn"]
    sub = table.loc[table["method"].isin(methods)]
    if sub.empty:
        return
    agg = sub.groupby(["trip_id", "method"], as_index=False)["absolute_error_kwh"].mean()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    trips = sorted(agg["trip_id"].unique())
    x = np.arange(len(trips))
    width = 0.15
    for i, method in enumerate(methods):
        g = agg.loc[agg["method"] == method].set_index("trip_id").reindex(trips)
        ax.bar(x + i * width, g["absolute_error_kwh"].to_numpy(), width=width, label=method)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(trips, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Absolute energy error (kWh)")
    ax.legend()
    ax.set_title("Per-trip absolute energy error")
    fig.tight_layout()
    _save(fig, path)


def figure_soc_reconstruction(trip: ProcessedTrip, soc_hat: np.ndarray, path: Path) -> None:
    t = elapsed_seconds(trip.frame["dt_s"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, trip.frame["soc"], lw=0.9, label="observed quantized/noisy SoC")
    ax.plot(t, soc_hat, lw=0.9, label="reconstructed SoC from ∫P_hat (initial SoC only)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SoC (%)")
    ax.set_title(f"{trip.trip_id}: SoC reconstruction on an unseen trip")
    ax.legend()
    _save(fig, path)


def figure_scarcity(table: pd.DataFrame, path: Path) -> None:
    if table.empty or "n_train" not in table.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, g in table.groupby("method"):
        if method == "physics":
            mae = g["absolute_error_kwh"].mean()
            axes[0].axhline(mae, ls="--", label="physics (reference)")
            continue
        stats = g.groupby("n_train")["absolute_error_kwh"].agg(["mean", "std"]).reset_index()
        axes[0].errorbar(stats["n_train"], stats["mean"], yerr=stats["std"], marker="o", label=method)
    axes[0].set_xlabel("Number of training trips")
    axes[0].set_ylabel("MAE (kWh)")
    axes[0].legend()
    axes[0].set_title("Data-scarcity learning curve")
    if {"pinn", "weak_mlp"}.issubset(set(table["method"])):
        a = table.loc[table["method"] == "pinn"].groupby(["trip_id", "n_train", "repeat"])["absolute_error_kwh"].mean()
        b = table.loc[table["method"] == "weak_mlp"].groupby(["trip_id", "n_train", "repeat"])["absolute_error_kwh"].mean()
        diff = (a - b).rename("diff").reset_index()
        stats = diff.groupby("n_train")["diff"].agg(["mean", "std"]).reset_index()
        axes[1].errorbar(stats["n_train"], stats["mean"], yerr=stats["std"], marker="o", color="#8c1d18")
        axes[1].axhline(0, color="0.5", lw=1)
        axes[1].set_xlabel("Number of training trips")
        axes[1].set_ylabel("PINN AE − WeakMLP AE (kWh)")
        axes[1].set_title("Negative = PINN lower error")
    fig.tight_layout()
    _save(fig, path)


def figure_ablation(table: pd.DataFrame, path: Path) -> None:
    order = ["physics", "weak_mlp", "mlp_state", "pinn_no_dynamics", "pinn_fulltrip", "pinn"]
    sub = table.loc[table["method"].isin(order)]
    if sub.empty:
        return
    mae = sub.groupby("method")["absolute_error_kwh"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(mae.index.astype(str), mae.to_numpy(), color="#3b6d99")
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("Ablation comparison")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def figure_delta_p(arrays: dict[str, np.ndarray], path: Path) -> None:
    delta = arrays.get("delta_p")
    p_phy = arrays.get("p_phy")
    p_hat = arrays.get("p_hat")
    if delta is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(delta, bins=40, color="#8c1d18")
    axes[0].set_xlabel("PINN δP (kW)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Neural correction distribution")
    if p_phy is not None and p_hat is not None:
        t = np.arange(len(p_hat))
        axes[1].plot(t, p_phy, lw=0.7, label="analytical power estimate")
        axes[1].plot(t, p_hat, lw=0.7, label="PINN reconstructed latent power")
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel("kW")
        axes[1].legend(fontsize=8)
        axes[1].set_title("Not measured battery power")
    fig.tight_layout()
    _save(fig, path)


def figure_cross_trajectory(table: pd.DataFrame, path: Path) -> None:
    if table.empty or "heldout_trajectory" not in table.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    methods = ["physics", "elasticnet", "weak_mlp", "pinn"]
    held = sorted(table["heldout_trajectory"].unique())
    x = np.arange(len(held))
    width = 0.18
    for i, method in enumerate(methods):
        means = [
            table.loc[(table["method"] == method) & (table["heldout_trajectory"] == h), "absolute_error_kwh"].mean()
            for h in held
        ]
        ax.bar(x + i * width, means, width=width, label=method)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"hold out {h}" for h in held])
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("Cross-trajectory generalization")
    ax.legend()
    _save(fig, path)
