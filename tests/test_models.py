"""Model-level tests for Phase 1–4 (physics path and later stubs)."""

from __future__ import annotations

import numpy as np
import torch

from models.pinn import PINN
from models.weak_mlp import WeakMLP
from physics.parameters import load_vehicle_parameters
from physics.vehicle_model import predict_trip_physics


def test_vehicle_yaml_loads():
    params = load_vehicle_parameters("configs/vehicle_twizy.yaml")
    assert params.battery_capacity_kwh == 6.0
    assert params.mass_kg == 549.0
    assert abs(params.cda_m2 - 0.64) < 1e-12
    assert params.cd is None
    assert params.frontal_area_m2 is None
    assert params.maximum_wheel_power_kw == 13.0


def test_trip_physics_shapes(vehicle_params):
    n = 50
    v = np.linspace(0.0, 20.0, n)
    a = np.gradient(v)
    theta = np.zeros(n)
    dt = np.zeros(n)
    dt[:-1] = 1.0
    result = predict_trip_physics(v, a, theta, dt, vehicle_params)
    assert result.p_battery_kw.shape == (n,)
    assert result.prefix_kwh.shape == (n,)
    assert result.prefix_kwh[0] == 0.0
    assert np.isfinite(result.energy_kwh)


def test_mlp_and_pinn_construct():
    mlp = WeakMLP()
    pinn = PINN()
    x = np.zeros((8, 8), dtype=np.float32)
    p = mlp(torch.tensor(x))
    assert p.shape == (8,)
    ph, dlt, dh = pinn(torch.tensor(x), torch.zeros(8), torch.zeros(8, 2))
    assert ph.shape == dlt.shape == dh.shape == (8,)
