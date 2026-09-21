"""Trip-level aggregation. Seeds are averaged; 1 Hz samples are never the unit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import summarize_energy_table


def trip_level_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per (trip, method): neural seeds averaged. Statistical unit = trip."""
    if pred.empty:
        return pred
    keys = ["trip_id", "method"]
    extra = [c for c in ("trajectory",) if c in pred.columns]
    num_cols = [
        c
        for c in (
            "distance_km",
            "duration_s",
            "observed_energy_kwh",
            "predicted_energy_kwh",
        )
        if c in pred.columns
    ]
    grouped = pred.groupby(keys + extra, as_index=False)[num_cols].mean()
    grouped["signed_error_kwh"] = grouped["predicted_energy_kwh"] - grouped["observed_energy_kwh"]
    grouped["absolute_error_kwh"] = grouped["signed_error_kwh"].abs()
    obs = grouped["observed_energy_kwh"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        grouped["percentage_error"] = np.where(
            np.abs(obs) > 1e-12,
            100.0 * grouped["signed_error_kwh"].to_numpy(dtype=float) / obs,
            np.nan,
        )
    return grouped


def trip_metrics(pred: pd.DataFrame) -> dict[str, dict[str, float]]:
    return summarize_energy_table(trip_level_predictions(pred))
