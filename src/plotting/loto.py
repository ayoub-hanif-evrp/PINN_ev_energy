"""LOTO diagnostic plots. Power curves are estimated, not ground truth."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.preprocessing import elapsed_seconds, ProcessedTrip


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["method"], summary["mae_kwh"], color="#3b6d99")
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("Quick LOTO trip-energy MAE (debug run, not paper numbers)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def plot_trip_diagnostics(
    trip: ProcessedTrip,
    arrays: dict[str, np.ndarray],
    soc_hat_pinn: np.ndarray,
    path: Path,
) -> None:
    df = trip.frame
    t = elapsed_seconds(df["dt_s"].to_numpy(dtype=float))
    fig, axes = plt.subplots(7, 1, figsize=(11, 14), sharex=True)
    axes[0].plot(t, df["speed_mps"] * 3.6, lw=0.8)
    axes[0].set_ylabel("Speed (km/h)")
    axes[1].plot(t, df["grade"] * 100.0, lw=0.8, color="#6b4c9a")
    axes[1].set_ylabel("Grade (%)")
    if "p_physics" in arrays:
        axes[2].plot(t, arrays["p_physics"], lw=0.8, label="physics-only (estimated)")
    if "p_mlp" in arrays:
        axes[2].plot(t, arrays["p_mlp"], lw=0.8, label="weak MLP (estimated)")
    if "p_pinn" in arrays:
        axes[2].plot(t, arrays["p_pinn"], lw=0.8, label="PINN P_hat (estimated)")
    axes[2].set_ylabel("P_batt (kW)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("Reconstructed latent battery-power profiles — not ground-truth power")
    if "delta_pinn" in arrays:
        axes[3].plot(t, arrays["delta_pinn"], lw=0.8, color="#8c1d18")
    axes[3].set_ylabel("PINN δP (kW)")
    if "soc" in df.columns:
        axes[4].plot(t, df["soc"], lw=0.8, label="observed quantized/noisy SoC")
    axes[4].plot(t, soc_hat_pinn, lw=0.8, label="PINN reconstructed SoC from ∫P_hat")
    axes[4].set_ylabel("SoC (%)")
    axes[4].legend(loc="upper right", fontsize=8)
    if "d_hat" in arrays and np.isfinite(arrays["d_hat"]).any():
        axes[5].plot(t, arrays["d_hat"], lw=0.8, color="#2a7f62")
    axes[5].set_ylabel("PINN D_hat (pp)")
    axes[6].plot(t, arrays.get("p_physics", np.zeros(len(t))), lw=0.6, alpha=0.7, label="P_phy")
    axes[6].plot(t, arrays.get("p_pinn", np.zeros(len(t))), lw=0.6, label="P_hat")
    axes[6].set_ylabel("kW")
    axes[6].set_xlabel("Time (s)")
    axes[0].set_title(f"{trip.trip_id}: diagnostic overlay (held-out prediction)")
    fig.tight_layout()
    _save(fig, path)
