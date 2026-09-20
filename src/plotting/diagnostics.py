"""Diagnostic plots for Phases 1–4. No fabricated values."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.preprocessing import ProcessedTrip, smooth_series


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_soc_transition_histogram(combined: dict, path: Path) -> None:
    values = combined.get("transition_values") or []
    counts = combined.get("transition_counts") or []
    fig, ax = plt.subplots(figsize=(7, 4))
    if values:
        ax.bar([str(v) for v in values[:30]], counts[:30], color="#3b6d99")
        ax.set_xlabel("Nonzero |ΔSoC| (percentage points)")
        ax.set_ylabel("Count")
        q = combined.get("q")
        ax.set_title(f"SoC transition distribution (estimated q = {q})")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No SoC transitions available", ha="center", va="center")
        ax.set_axis_off()
    _save(fig, path)


def plot_trip_overview(trips: list[ProcessedTrip], path: Path, max_trips: int = 6) -> None:
    subset = trips[:max_trips]
    if not subset:
        return
    fig, axes = plt.subplots(len(subset), 3, figsize=(12, 2.4 * len(subset)), squeeze=False)
    for i, trip in enumerate(subset):
        df = trip.frame
        t = np.cumsum(df["dt_s"].to_numpy(dtype=float))
        axes[i, 0].plot(t, df["speed_mps"] * 3.6, lw=0.8)
        axes[i, 0].set_ylabel("Speed (km/h)")
        axes[i, 1].plot(t, df["alt_m"], lw=0.8, color="#b35c00")
        axes[i, 1].set_ylabel("Altitude (m)")
        if "soc" in df.columns:
            axes[i, 2].plot(t, df["soc"], lw=0.8, color="#2a7f62")
        axes[i, 2].set_ylabel("SoC (%)")
        axes[i, 0].set_title(f"{trip.trip_id} ({trip.trajectory})")
        for ax in axes[i]:
            ax.set_xlabel("Time (s)")
    fig.tight_layout()
    _save(fig, path)


def plot_speed_acceleration(trip: ProcessedTrip, path: Path, sensitivity_windows: Iterable[int] | None = None) -> None:
    df = trip.frame
    t = np.cumsum(df["dt_s"].to_numpy(dtype=float))
    v = df["speed_mps"].to_numpy(dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, v * 3.6, lw=0.7, label="raw CAN speed", alpha=0.7)
    axes[0].plot(t, df["speed_mps_smooth"] * 3.6, lw=1.0, label="smoothed")
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"{trip.trip_id}: speed smoothing and acceleration")
    axes[1].plot(t, df["acc_mps2"], lw=0.8)
    axes[1].set_ylabel("Acceleration (m/s²)")
    if sensitivity_windows:
        for w in sensitivity_windows:
            vs = smooth_series(v, int(w), 2)
            axes[2].plot(t, vs * 3.6, lw=0.8, label=f"window={w}")
        axes[2].set_ylabel("Smoothed speed (km/h)")
        axes[2].legend(loc="upper right", ncol=2)
    else:
        axes[2].plot(t, df["grade"] * 100.0, lw=0.8, color="#6b4c9a")
        axes[2].set_ylabel("Grade (%)")
    axes[2].set_xlabel("Time (s)")
    fig.tight_layout()
    _save(fig, path)


def plot_grade(trip: ProcessedTrip, path: Path) -> None:
    df = trip.frame
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    axes[0].plot(df["s_m"] / 1000.0, df["alt_m"], lw=0.8)
    axes[0].set_xlabel("Distance (km)")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].set_title(f"{trip.trip_id}: altitude and grade")
    axes[1].plot(df["s_m"] / 1000.0, df["grade"] * 100.0, lw=0.8, color="#6b4c9a")
    axes[1].set_xlabel("Distance (km)")
    axes[1].set_ylabel("Grade (%)")
    axes[2].plot(df["s_m"] / 1000.0, np.rad2deg(df["theta_rad"]), lw=0.8, color="#8c1d18")
    axes[2].set_xlabel("Distance (km)")
    axes[2].set_ylabel("Slope angle (deg)")
    fig.tight_layout()
    _save(fig, path)


def plot_method_diagram(path: Path) -> None:
    """Schematic of the weakly supervised PINN. Contains no numerical results."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Weakly supervised physics-informed EV energy estimation (schematic)")
    boxes = [
        (0.3, 4.0, "Driving telemetry\n(v, a, θ, env.)"),
        (3.0, 4.6, "Longitudinal\nphysics P_phy(t)"),
        (3.0, 3.2, "Small NN\ncorrection δP"),
        (5.8, 4.0, "Latent power\nP̂ = P_phy + δP"),
        (8.3, 4.0, "Integration\nÊ[a:b]"),
        (8.3, 1.8, "SoC-derived\nE_obs[a:b]"),
        (5.8, 1.8, "Multi-scale\nenergy constraint"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 2.2, 1.1, fill=True, facecolor="#eef3f8", edgecolor="#1f3b57"))
        ax.text(x + 1.1, y + 0.55, text, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(3.0, 5.15), xytext=(2.5, 4.55), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(3.0, 3.75), xytext=(2.5, 4.35), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5.8, 4.55), xytext=(5.2, 5.15), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5.8, 4.35), xytext=(5.2, 3.75), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(8.3, 4.55), xytext=(8.0, 4.55), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(9.4, 4.0), xytext=(9.4, 2.9), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(8.0, 2.35), xytext=(8.3, 2.35), arrowprops=dict(arrowstyle="->"))
    ax.text(5.5, 0.6, "No instantaneous battery-power labels. Wind is not used as headwind.", fontsize=9)
    _save(fig, path)


def plot_physics_vs_soc_energy(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(table["e_obs_kwh"], table["e_physics_kwh"], c="#1f3b57")
    lims = [
        min(table["e_obs_kwh"].min(), table["e_physics_kwh"].min()),
        max(table["e_obs_kwh"].max(), table["e_physics_kwh"].max()),
    ]
    ax.plot(lims, lims, "k--", lw=0.8)
    ax.set_xlabel("SoC-derived trip energy (kWh)")
    ax.set_ylabel("Physics-only trip energy (kWh)")
    ax.set_title("Diagnostic only — not a paper result table")
    ax.set_aspect("equal", adjustable="box")
    _save(fig, path)
