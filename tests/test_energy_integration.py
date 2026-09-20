"""Energy integration prefix vs direct summation."""

from __future__ import annotations

import numpy as np

from units import energy_kwh_from_power, energy_prefix_kwh, window_energy_from_prefix


def test_prefix_window_equals_direct_sum():
    rng = np.random.default_rng(0)
    p = rng.normal(size=200)
    dt = rng.uniform(0.8, 1.2, size=200)
    prefix = energy_prefix_kwh(p, dt)
    for start, end in [(0, 10), (3, 3), (50, 180), (0, 199)]:
        direct = energy_kwh_from_power(p[start : end + 1], dt[start : end + 1])
        via = window_energy_from_prefix(prefix, start, end)
        assert abs(direct - via) < 1e-12


def test_full_series_prefix_matches_total_energy():
    p = np.linspace(-2.0, 5.0, 500)
    dt = np.full(500, 0.5)
    prefix = energy_prefix_kwh(p, dt)
    assert abs(prefix[-1] - energy_kwh_from_power(p, dt)) < 1e-12
    assert prefix[0] == 0.0
