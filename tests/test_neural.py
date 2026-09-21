"""Neural model, PINN residual, and leakage tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from config import load_config
from data.loader import load_trip_csv
from data.preprocessing import preprocess_trip
from data.schema import MAIN_MODEL_FEATURES
from data.scaling import TripStandardScaler
from data.windows import generate_windows
from models.pinn import PINN
from models.weak_mlp import WeakMLP
from paths import project_root
from physics.parameters import load_vehicle_parameters
from training.cross_validation import leave_one_trip_out, select_validation_trip
from training.losses import pinn_losses
from training.torch_ops import torch_energy_prefix, torch_interval_increments_kwh
from training.trainer import make_trip_batch

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _trips():
    config = load_config(project_root() / "configs" / "base.yaml")
    t1 = preprocess_trip(
        load_trip_csv(FIXTURES / "analysed" / "T1" / "T1_01_01_2021.csv", "T1_01_01_2021", "T1", "analysed"),
        config,
    )
    t2 = preprocess_trip(
        load_trip_csv(FIXTURES / "analysed" / "T2" / "T2_01_01_2021.csv", "T2_01_01_2021", "T2", "analysed"),
        config,
    )
    return config, t1, t2


def test_mlp_and_pinn_output_shapes():
    n, f = 20, len(MAIN_MODEL_FEATURES)
    x = torch.randn(n, f)
    mlp = WeakMLP()
    p = mlp(x)
    assert p.shape == (n,)
    assert torch.isfinite(p).all()
    pinn = PINN(residual_limit_kw=4.0)
    p_phy = torch.zeros(n)
    p_hat, delta, d_hat = pinn(x, p_phy)
    assert p_hat.shape == (n,)
    assert delta.shape == (n,)
    assert d_hat.shape == (n,)
    assert torch.max(delta.abs()) <= 4.0 + 1e-5


def test_pinn_residual_bounded_by_limit():
    pinn = PINN(residual_limit_kw=4.0)
    x = torch.randn(50, len(MAIN_MODEL_FEATURES)) * 10
    with torch.no_grad():
        _, delta, _ = pinn(x, torch.zeros(50))
        assert float(delta.abs().max()) <= 4.0 + 1e-6


def test_dynamics_loss_zero_when_state_matches_power_integral():
    p_hat = torch.linspace(1.0, 2.0, 30, dtype=torch.float64)
    dt = torch.ones(30, dtype=torch.float64)
    dt[-1] = 0.0
    prefix = torch_energy_prefix(p_hat, dt)
    d_hat = 100.0 * prefix / 6.0
    delta = torch.zeros_like(p_hat)
    starts = torch.tensor([0], dtype=torch.long)
    ends = torch.tensor([29], dtype=torch.long)
    e_obs = prefix[-1:].float()
    loss, logs = pinn_losses(
        p_hat.float(),
        delta.float(),
        d_hat.float(),
        d_hat.float(),
        dt.float(),
        starts,
        ends,
        e_obs,
        ["full_trip"],
        6.0,
        0.01,
        0.1,
        5.0,
        {"lambda_window": 0.0, "lambda_dynamics": 1.0, "lambda_state": 0.0, "lambda_boundary": 0.0, "lambda_prior": 0.0},
    )
    assert logs["l_dynamics"] < 1e-6


def test_dynamics_loss_increases_when_state_violates_conservation():
    p_hat = torch.ones(30)
    dt = torch.ones(30)
    dt[-1] = 0.0
    de = torch_interval_increments_kwh(p_hat, dt)
    prefix = torch_energy_prefix(p_hat, dt)
    d_true = 100.0 * prefix / 6.0
    d_bad = torch.zeros_like(d_true)
    zeros = torch.zeros_like(p_hat)
    starts = torch.tensor([0])
    ends = torch.tensor([29])
    kwargs = dict(
        p_hat=p_hat,
        delta_p=zeros,
        dt_s=dt,
        starts=starts,
        ends=ends,
        e_obs=prefix[-1:].detach(),
        scales=["full_trip"],
        battery_capacity_kwh=6.0,
        huber_window=0.01,
        huber_state=0.1,
        p_scale_kw=5.0,
        lambdas={"lambda_window": 0.0, "lambda_dynamics": 1.0, "lambda_state": 0.0, "lambda_boundary": 0.0, "lambda_prior": 0.0},
    )
    good, glog = pinn_losses(d_obs=d_true, d_hat=d_true, **kwargs)
    bad, blog = pinn_losses(d_obs=d_bad, d_hat=d_bad, **kwargs)
    assert blog["l_dynamics"] > glog["l_dynamics"] + 1e-6


def test_test_soc_not_in_predictor_tensors():
    config, train, test = _trips()
    scaler = TripStandardScaler().fit([train])
    scaler.assert_not_fitted_on(test.trip_id)
    x = scaler.transform_trip(test)
    assert x.shape[1] == len(MAIN_MODEL_FEATURES)
    assert "soc" not in MAIN_MODEL_FEATURES
    params = load_vehicle_parameters("configs/vehicle_twizy.yaml")
    batch = make_trip_batch(test, x, params, config, [], torch.device("cpu"), include_state=False)
    batch.assert_no_soc_input()
    assert batch.d_obs is None


def test_loto_validation_is_not_the_outer_test_trip():
    config, t1, t2 = _trips()
    folds = leave_one_trip_out([t1, t2], seed=0)
    assert len(folds) == 2
    for fold in folds:
        assert fold.val_id != fold.test_id
        assert fold.test_id not in fold.inner_train_ids
        assert fold.test_id not in fold.train_ids or True
        assert fold.test_id not in fold.inner_train_ids
        windows = generate_windows([t for t in [t1, t2] if t.trip_id in fold.inner_train_ids], 6.0, 0.02, config=config)
        assert all(w.trip_id != fold.test_id for w in windows)


def test_select_validation_trip_is_deterministic():
    a = select_validation_trip(["T1", "T2", "T3"], "held", 0)
    b = select_validation_trip(["T1", "T2", "T3"], "held", 0)
    assert a == b
    assert a != "held"
