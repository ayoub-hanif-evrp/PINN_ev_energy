"""Energy integration: trapezoid over n-1 intervals, E(a,b)=prefix[b]-prefix[a]."""

from __future__ import annotations

import numpy as np

from units import energy_kwh_from_power, energy_prefix_kwh, window_energy_from_prefix


def test_one_kw_over_3600_intervals_is_one_kwh():
    # 3601 samples spanning 3600 one-second intervals.
    p = np.ones(3601)
    dt = np.zeros(3601)
    dt[:-1] = 1.0
    assert abs(energy_kwh_from_power(p, dt) - 1.0) < 1e-12


def test_prefix_window_equals_trapezoid_sum():
    rng = np.random.default_rng(0)
    p = rng.normal(size=200)
    dt = np.zeros(200)
    dt[:-1] = rng.uniform(0.8, 1.2, size=199)
    prefix = energy_prefix_kwh(p, dt)
    assert prefix[0] == 0.0
    assert prefix.shape == (200,)
    for start, end in [(0, 10), (3, 3), (50, 180), (0, 199)]:
        if end == start:
            direct = 0.0
        else:
            mid = 0.5 * (p[start:end] + p[start + 1 : end + 1])
            direct = float(np.sum(mid * dt[start:end] / 3600.0))
        via = window_energy_from_prefix(prefix, start, end)
        assert abs(direct - via) < 1e-12


def test_full_series_prefix_matches_total_energy():
    p = np.linspace(-2.0, 5.0, 500)
    dt = np.zeros(500)
    dt[:-1] = 0.5
    prefix = energy_prefix_kwh(p, dt)
    assert abs(prefix[-1] - energy_kwh_from_power(p, dt)) < 1e-12
    assert prefix[0] == 0.0
    assert prefix.shape == (500,)
