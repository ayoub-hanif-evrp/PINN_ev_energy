"""Physics-model sanity tests."""

from __future__ import annotations

import numpy as np
import pytest

from physics.vehicle_model import (
    battery_power_from_wheel,
    predict_battery_power,
    soft_clamp,
    wheel_power_kw,
)
from units import energy_kwh_from_power


def test_one_kw_for_3600_seconds_is_one_kwh():
    p = np.ones(3601)
    dt = np.zeros(3601)
    dt[:-1] = 1.0
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
    p_batt, _, _ = battery_power_from_wheel(np.array([-4.0]), vehicle_params)
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
    assert abs(forces["f_total_n"][0]) > 0.0


def test_wind_as_headwind_is_rejected(vehicle_params):
    with pytest.raises(ValueError, match="headwind"):
        predict_battery_power(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            vehicle_params,
            use_wind_as_headwind=True,
        )


def test_soft_clamp_is_identity_inside_range():
    x = np.array([-4.0, 0.0, 5.0, 12.0])
    y = soft_clamp(x, -8.0, 13.0, sharpness=8.0)
    np.testing.assert_allclose(y, x, atol=1e-3)


def test_battery_power_can_exceed_motor_mechanical_limit(vehicle_params):
    # Interior wheel power is unchanged; losses make battery power exceed 13 kW.
    p_batt, p_w, n_bound = battery_power_from_wheel(
        np.array([12.0]), vehicle_params, apply_bounds=True
    )
    assert n_bound == 0
    assert p_batt[0] > 13.0
    expected = 12.0 / vehicle_params.eta_drive + vehicle_params.auxiliary_power_kw
    assert abs(p_batt[0] - expected) < 1e-3
    assert abs(p_w[0] - 12.0) < 1e-3
