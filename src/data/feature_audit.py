"""Feature-unit and missingness checks. Imputation stats come from training trips."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data.preprocessing import ProcessedTrip
from data.schema import FEATURE_UNITS, MAIN_MODEL_FEATURES, PROGRESS_FEATURES


def feature_distribution_table(trips: list[ProcessedTrip], columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    cols = list(columns or (MAIN_MODEL_FEATURES + PROGRESS_FEATURES))
    rows = []
    for name in cols:
        values = []
        n_missing = 0
        n_total = 0
        for trip in trips:
            if name not in trip.frame.columns:
                n_total += trip.n_rows
                n_missing += trip.n_rows
                continue
            x = pd.to_numeric(trip.frame[name], errors="coerce").to_numpy(dtype=float)
            n_total += x.size
            n_missing += int(np.isnan(x).sum())
            values.append(x[np.isfinite(x)])
        stacked = np.concatenate(values) if values else np.array([])
        miss = n_missing / n_total if n_total else 1.0
        rows.append(
            {
                "feature": name,
                "unit": FEATURE_UNITS.get(name, ""),
                "n": int(stacked.size),
                "n_unique": int(np.unique(np.round(stacked, 6)).size) if stacked.size else 0,
                "min": float(np.min(stacked)) if stacked.size else np.nan,
                "max": float(np.max(stacked)) if stacked.size else np.nan,
                "mean": float(np.mean(stacked)) if stacked.size else np.nan,
                "std": float(np.std(stacked, ddof=0)) if stacked.size else np.nan,
                "missing_fraction": float(miss),
            }
        )
    return pd.DataFrame(rows)


def assert_feature_units_and_coverage(table: pd.DataFrame, max_missing: float = 0.5) -> None:
    if table.empty:
        raise AssertionError("No feature statistics to check.")
    wind = table.loc[table["feature"] == "wind_speed_mps"]
    if not wind.empty:
        mx = float(wind["max"].iloc[0])
        # Analysed WindSpeedAv is m/s. Typical values are well below 50.
        if np.isfinite(mx) and mx > 80:
            raise AssertionError(
                f"wind_speed_mps max={mx} looks like km/h. Do not convert WindSpeedAv as km/h."
            )
    bad = table.loc[table["missing_fraction"] > max_missing, "feature"].tolist()
    if bad:
        raise AssertionError(f"Features exceed {max_missing:.0%} missing on the checked trips: {bad}")


def summarize_units() -> dict[str, str]:
    return dict(FEATURE_UNITS)
