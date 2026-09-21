"""Elevation-gain features must not sum raw GPS altitude noise."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from data.features import elevation_gain_loss_from_grade, trip_level_features
from data.loader import TripRecord
from data.preprocessing import preprocess_trip
from data.schema import ColumnMap
from paths import project_root


def _synthetic_record(altitude: np.ndarray) -> TripRecord:
    n = len(altitude)
    # ~10 m/s along a meridian so Haversine distance is well defined.
    lat = 34.0 + np.arange(n) * (10.0 / 111_000.0)
    lon = np.full(n, -6.8)
    frame = pd.DataFrame(
        {
            "date": ["01/01/2021"] * n,
            "time": [f"10:00:{i:02d}" if i < 60 else f"10:01:{i - 60:02d}" for i in range(n)],
            "soc": np.linspace(80.0, 79.0, n),
            "speed_kmh": np.full(n, 36.0),
            "mode": ["D"] * n,
            "lat": lat,
            "lon": lon,
            "alt_m": altitude,
            "gps_speed_kmh": np.full(n, 36.0),
            "temperature_c": np.full(n, 20.0),
            "humidity_pct": np.full(n, 50.0),
            "weather": ["Clear"] * n,
            "wind_speed_mps": np.full(n, 1.0),
            "traffic": np.zeros(n),
            "speed_limit_kmh": np.full(n, 40.0),
        }
    )
    return TripRecord("elev_synth", "T1", Path("elev_synth.csv"), "analysed", frame, ColumnMap({}, {}), [])


def test_noisy_altitude_does_not_inflate_cumulative_gain():
    n = 120
    true_alt = np.linspace(100.0, 130.0, n)  # 30 m true ascent
    rng = np.random.default_rng(0)
    noisy_alt = true_alt + rng.normal(0.0, 8.0, size=n)
    raw_gain = float(np.sum(np.clip(np.diff(noisy_alt, prepend=noisy_alt[0]), 0.0, None)))
    assert raw_gain > 80.0  # successive raw diffs explode

    config = load_config(project_root() / "configs" / "base.yaml")
    trip = preprocess_trip(_synthetic_record(noisy_alt), config)
    assert "alt_smooth_m" in trip.frame.columns
    feats = trip_level_features(trip)
    gain = feats["cumulative_elevation_gain_m"]
    assert gain < 0.5 * raw_gain
    assert gain < 80.0
    # True climb is 30 m; smoothed/grade-based gain should stay in that ballpark.
    assert 5.0 < gain < 70.0


def test_elevation_from_grade_is_grade_times_ds():
    grade = np.array([0.0, 0.1, -0.05, 0.0])
    ds = np.array([0.0, 10.0, 10.0, 10.0])
    gain, loss = elevation_gain_loss_from_grade(grade, ds)
    assert abs(gain - 1.0) < 1e-12
    assert abs(loss - 0.5) < 1e-12
