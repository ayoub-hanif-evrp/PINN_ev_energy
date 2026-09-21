"""Fit feature scalers on training trips only."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.preprocessing import ProcessedTrip
from data.schema import MAIN_MODEL_FEATURES


@dataclass
class TripStandardScaler:
    columns: tuple[str, ...] = MAIN_MODEL_FEATURES
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    fitted_trip_ids: list[str] = field(default_factory=list)

    def fit(self, trips: list[ProcessedTrip]) -> "TripStandardScaler":
        frames = []
        ids = []
        for trip in trips:
            missing = [c for c in self.columns if c not in trip.frame.columns]
            if missing:
                raise KeyError(f"{trip.trip_id}: missing {missing}")
            frames.append(trip.frame.loc[:, list(self.columns)])
            ids.append(trip.trip_id)
        stacked = pd.concat(frames, axis=0, ignore_index=True)
        self.mean_ = stacked.mean(axis=0).to_numpy(dtype=float)
        std = stacked.std(axis=0, ddof=0).to_numpy(dtype=float)
        std = np.where(std < 1e-12, 1.0, std)
        self.std_ = std
        self.fitted_trip_ids = ids
        return self

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fit.")
        x = frame.loc[:, list(self.columns)].to_numpy(dtype=float)
        scaled = (x - self.mean_) / self.std_
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def transform_trip(self, trip: ProcessedTrip) -> np.ndarray:
        return self.transform_frame(trip.frame)

    def assert_not_fitted_on(self, trip_id: str) -> None:
        if trip_id in self.fitted_trip_ids:
            raise AssertionError(f"Leakage: scaler was fit including test trip {trip_id}")
