"""Non-overlapping test-window energy evaluation (evaluation only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data.preprocessing import ProcessedTrip
from data.windows import generate_trip_windows
from units import energy_prefix_kwh, window_energy_from_prefix


def nonoverlapping_window_metrics(
    trip: ProcessedTrip,
    p_hat: np.ndarray,
    battery_capacity_kwh: float,
    q: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fixed window scales from config. Windows are non-overlapping. Trip remains the unit."""
    windows = generate_trip_windows(
        trip, battery_capacity_kwh, q, config=config, overlapping=False
    )
    dt = trip.frame["dt_s"].to_numpy(dtype=float)
    prefix = energy_prefix_kwh(np.asarray(p_hat, dtype=float), dt)
    rows: list[dict[str, Any]] = []
    by_scale: dict[str, list[tuple[float, float]]] = {}
    for w in windows:
        e_pred = window_energy_from_prefix(prefix, w.start, w.end)
        by_scale.setdefault(w.scale, []).append((e_pred, w.e_obs_kwh))
        rows.append(
            {
                "scale": w.scale,
                "start": w.start,
                "end": w.end,
                "e_pred_kwh": float(e_pred),
                "e_obs_kwh": float(w.e_obs_kwh),
                "absolute_error_kwh": float(abs(e_pred - w.e_obs_kwh)),
            }
        )
    summary = []
    for scale, pairs in sorted(by_scale.items()):
        err = np.array([abs(a - b) for a, b in pairs], dtype=float)
        se = np.array([a - b for a, b in pairs], dtype=float)
        summary.append(
            {
                "scale": scale,
                "n_windows": int(len(pairs)),
                "mae_kwh": float(np.mean(err)),
                "rmse_kwh": float(np.sqrt(np.mean(se**2))),
            }
        )
    # Keep both per-window and per-scale summaries via a tag.
    out = [{"level": "scale", **s} for s in summary]
    out.extend({"level": "window", **r} for r in rows)
    return out


def summarize_window_eval(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    scale_rows = frame.loc[frame["level"] == "scale"] if "level" in frame.columns else frame
    if scale_rows.empty:
        return scale_rows
    return (
        scale_rows.groupby(["method", "scale"], dropna=False)[["mae_kwh", "rmse_kwh", "n_windows"]]
        .mean()
        .reset_index()
    )
