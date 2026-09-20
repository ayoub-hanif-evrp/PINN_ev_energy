"""Physics-model sanity tests."""

from __future__ import annotations

import numpy as np
import pytest

from physics.vehicle_model import (
    battery_power_from_wheel,
    predict_battery_power,
    wheel_power_kw,
)
from units import energy_kwh_from_power


def test_one_kw_for_3600_seconds_is_one_kwh():
    p = np.ones(3600)
    dt = np.ones(3600)
    assert abs(energy_kwh_from_power(p, dt) - 1.0) < 1e-12


def test_uphill_grade_increases_wheel_power(vehicle_params):
    v = np.array([10.0, 10.0])
    a = np.array([0.0, 0.0])
    flat, _ = wheel_power_kw(v, a, np.array([0.0, 0.0]), vehicle_params)
    up, _ = wheel_power_kw(v, a, np.array([0.1, 0.1]), vehicle_params)
    assert np.all(up > flat)


def test_positive_acceleration_increases_power(vehicle_params):
    v = np.array([10.0, 10.0])
    theta = np.array([0.0, 0.0])
    coast, _ = wheel_power_kw(v, np.array([0.0, 0.0]), theta, vehicle_params)
    accel, _ = wheel_power_kw(v, np.array([1.0, 1.0]), theta, vehicle_params)
    assert np.all(accel > coast)


def test_negative_wheel_power_can_regenerate(vehicle_params):
    p_batt, _ = battery_power_from_wheel(np.array([-4.0]), vehicle_params)
    assert p_batt[0] < vehicle_params.auxiliary_power_kw
    expected = vehicle_params.eta_regen * (-4.0) + vehicle_params.auxiliary_power_kw
    assert abs(p_batt[0] - expected) < 1e-12


def test_zero_speed_mechanical_power_is_zero_except_aux(vehicle_params):
    v = np.array([0.0])
    a = np.array([0.5])
    theta = np.array([0.2])
    p_wheel, forces = wheel_power_kw(v, a, theta, vehicle_params)
    assert abs(p_wheel[0]) < 1e-12
    result = predict_battery_power(v, a, theta, vehicle_params, apply_bounds=False)
    assert abs(result.p_battery_kw[0] - vehicle_params.auxiliary_power_kw) < 1e-12
    assert abs(forces["f_total_n"][0]) > 0.0  # forces exist, power is F*v


def test_wind_as_headwind_is_rejected(vehicle_params):
    with pytest.raises(ValueError, match="headwind"):
        predict_battery_power(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            vehicle_params,
            use_wind_as_headwind=True,
        )
