"""Model-level tests for Phase 1–4 (physics path and later stubs)."""

from __future__ import annotations

import numpy as np
import pytest

from models.pinn import PINN
from models.weak_mlp import WeakMLP
from phase_status import PhaseNotImplementedError
from physics.parameters import load_vehicle_parameters
from physics.vehicle_model import predict_trip_physics


def test_vehicle_yaml_loads():
    params = load_vehicle_parameters("configs/vehicle_twizy.yaml")
    assert params.battery_capacity_kwh == 6.0
    assert params.mass_kg == 549.0
    assert abs(params.cda_m2 - params.cd * params.frontal_area_m2) < 1e-12


def test_trip_physics_shapes(vehicle_params):
    n = 50
    v = np.linspace(0.0, 20.0, n)
    a = np.gradient(v)
    theta = np.zeros(n)
    dt = np.ones(n)
    result = predict_trip_physics(v, a, theta, dt, vehicle_params)
    assert result.p_battery_kw.shape == (n,)
    assert result.prefix_kwh.shape == (n + 1,)
    assert np.isfinite(result.energy_kwh)


def test_pinn_and_mlp_are_not_yet_implemented():
    with pytest.raises(PhaseNotImplementedError):
        PINN()
    with pytest.raises(PhaseNotImplementedError):
        WeakMLP()
