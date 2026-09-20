"""Unit conversion tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocessing import haversine_m
from units import (
    EARTH_RADIUS_M,
    KMH_TO_MPS,
    energy_kwh_from_power,
    kmh_to_mps,
    mps_to_kmh,
)


def test_kmh_to_mps_and_back():
    v = np.array([0.0, 36.0, 72.0, 108.0])
    mps = kmh_to_mps(v)
    np.testing.assert_allclose(mps, [0.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(mps_to_kmh(mps), v)
    assert abs(KMH_TO_MPS - 1.0 / 3.6) < 1e-15


def test_haversine_one_degree_at_equator():
    d = float(haversine_m(0.0, 0.0, 0.0, 1.0))
    expected = 2 * EARTH_RADIUS_M * np.arcsin(np.sin(np.deg2rad(0.5)))
    # 1 deg longitude at equator: R * deg2rad(1)
    expected2 = EARTH_RADIUS_M * np.deg2rad(1.0)
    assert abs(d - expected2) / expected2 < 1e-6
    assert abs(d - expected) / expected < 1e-6


def test_dt_in_seconds_used_for_energy():
    p = np.ones(11)
    dt = np.zeros(11)
    dt[:-1] = 2.0  # 10 intervals of 2 s
    # 1 kW * 20 s = 20/3600 kWh
    assert abs(energy_kwh_from_power(p, dt) - 20.0 / 3600.0) < 1e-12


def test_epoch_seconds_is_unit_safe():
    from data.preprocessing import compute_dt_seconds, epoch_seconds

    ts = pd.date_range("2021-03-29 11:13:45", periods=5, freq="s")
    epoch = epoch_seconds(ts)
    np.testing.assert_allclose(np.diff(epoch), 1.0)
    dt, stats = compute_dt_seconds(ts.to_numpy())
    np.testing.assert_allclose(dt[:-1], 1.0)
    assert dt[-1] == 0.0
    assert stats["n_intervals"] == 4
    assert stats["n_nonpositive_dt"] == 0
