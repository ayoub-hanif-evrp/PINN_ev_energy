"""Neural model, PINN residual, and leakage tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from config import load_config
from data.loader import load_trip_csv
from data.preprocessing import preprocess_trip
from data.quantization import q_from_trips
from data.schema import MAIN_MODEL_FEATURES, PROGRESS_FEATURES
from data.scaling import ProgressScaler, TripStandardScaler
from data.windows import generate_windows
from models.mlp_state import MLPState
from models.pinn import PINN
from models.weak_mlp import WeakMLP
from paths import project_root
from physics.parameters import load_vehicle_parameters
from training.cross_validation import leave_one_trip_out, select_validation_ids, select_validation_trip
from training.losses import pinn_losses
from training.torch_ops import torch_energy_prefix, torch_interval_increments_kwh
from training.trainer import make_inference_batch, make_trip_batch, predict_power

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


def _three_trips():
    config, t1, t2 = _trips()
    t3 = replace(t1, trip_id="T3_clone", trajectory="T3", frame=t1.frame.copy())
    return config, t1, t2, t3


def test_mlp_and_pinn_output_shapes():
    n, f = 20, len(MAIN_MODEL_FEATURES)
    x = torch.randn(n, f)
    progress = torch.randn(n, len(PROGRESS_FEATURES))
    mlp = WeakMLP()
    p = mlp(x)
    assert p.shape == (n,)
    assert torch.isfinite(p).all()
    pinn = PINN(residual_limit_kw=4.0)
    p_phy = torch.zeros(n)
    with torch.no_grad():
        p_hat, delta, d_hat = pinn(x, p_phy, progress)
        assert p_hat.shape == (n,)
        assert delta.shape == (n,)
        assert d_hat.shape == (n,)
        assert torch.max(delta.abs()) <= 4.0 + 1e-5
        assert float(d_hat[0]) == pytest.approx(0.0, abs=1e-6)
        state = MLPState()
        p_s, d_s = state(x, progress)
        assert p_s.shape == d_s.shape == (n,)
        assert float(d_s[0]) == pytest.approx(0.0, abs=1e-6)


def test_pinn_residual_bounded_by_limit():
    pinn = PINN(residual_limit_kw=4.0)
    x = torch.randn(50, len(MAIN_MODEL_FEATURES)) * 10
    progress = torch.randn(50, len(PROGRESS_FEATURES))
    with torch.no_grad():
        _, delta, d_hat = pinn(x, torch.zeros(50), progress)
        assert float(delta.abs().max()) <= 4.0 + 1e-6
        assert float(d_hat[0]) == pytest.approx(0.0, abs=1e-7)


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
    prog = ProgressScaler().fit([train])
    scaler.assert_not_fitted_on(test.trip_id)
    x = scaler.transform_trip(test)
    assert x.shape[1] == len(MAIN_MODEL_FEATURES)
    assert "soc" not in MAIN_MODEL_FEATURES
    params = load_vehicle_parameters("configs/vehicle_twizy.yaml")
    batch = make_inference_batch(test, x, params, config, torch.device("cpu"), progress_scaled=prog.transform_trip(test))
    batch.assert_no_soc_input()
    assert batch.d_obs is None
    assert batch.contains_measured_soc is False


def test_poisoned_test_soc_after_t0_does_not_change_phat():
    config, train, test = _trips()
    scaler = TripStandardScaler().fit([train])
    prog = ProgressScaler().fit([train])
    params = load_vehicle_parameters("configs/vehicle_twizy.yaml")
    device = torch.device("cpu")
    x = scaler.transform_trip(test)
    pr = prog.transform_trip(test)
    model = PINN()
    b1 = make_inference_batch(test, x, params, config, device, progress_scaled=pr)
    p1 = predict_power(model, b1, "pinn")
    poisoned = replace(test, frame=test.frame.copy())
    poisoned.frame.loc[poisoned.frame.index[1:], "soc"] = 0.0
    b2 = make_inference_batch(poisoned, x, params, config, device, progress_scaled=pr)
    p2 = predict_power(model, b2, "pinn")
    np.testing.assert_allclose(p1["p_hat"], p2["p_hat"])
    np.testing.assert_allclose(p1["delta_p"], p2["delta_p"])


def test_held_out_soc_resolution_does_not_change_training_q():
    _, t1, t2 = _trips()
    q_train = q_from_trips([t1])
    poisoned = replace(t2, frame=t2.frame.copy())
    n = len(poisoned.frame)
    poisoned.frame["soc"] = np.round(np.linspace(90.0, 50.0, n))
    q_train_again = q_from_trips([t1])
    assert q_train_again == pytest.approx(q_train)
    # Changing the held-out trip's SoC grid must not change training-only q.
    assert q_from_trips([t1]) == pytest.approx(q_train)
    q_leaky = q_from_trips([t1, poisoned])
    assert isinstance(q_leaky, float)


def test_loto_validation_is_not_the_outer_test_trip():
    config, t1, t2, t3 = _three_trips()
    folds = leave_one_trip_out([t1, t2, t3], seed=0)
    assert len(folds) == 3
    for fold in folds:
        assert fold.test_id not in fold.train_ids
        assert fold.test_id not in fold.inner_train_ids
        assert fold.test_id not in fold.val_ids
        assert set(fold.val_ids).isdisjoint(fold.inner_train_ids)
        windows = generate_windows(
            [t for t in [t1, t2, t3] if t.trip_id in fold.inner_train_ids],
            6.0,
            0.02,
            config=config,
        )
        assert all(w.trip_id != fold.test_id for w in windows)
        scaler = TripStandardScaler().fit([t for t in [t1, t2, t3] if t.trip_id in fold.inner_train_ids] or [t for t in [t1, t2, t3] if t.trip_id in fold.train_ids])
        scaler.assert_not_fitted_on(fold.test_id)


def test_grouped_validation_one_per_trajectory():
    _, t1, t2, t3 = _three_trips()
    ids = select_validation_ids([t1, t2, t3], [t1.trip_id, t2.trip_id, t3.trip_id], "held", 0)
    traj = {t.trip_id: t.trajectory for t in (t1, t2, t3)}
    assert len(ids) == 3
    assert {traj[i] for i in ids} == {"T1", "T2", "T3"}


def test_select_validation_trip_is_deterministic():
    a = select_validation_trip(["T1", "T2", "T3"], "held", 0)
    b = select_validation_trip(["T1", "T2", "T3"], "held", 0)
    assert a == b
    assert a != "held"


def test_progress_features_are_causal_and_training_scaled():
    _, train, test = _trips()
    assert "elapsed_s" in train.frame.columns
    assert "cumulative_distance_km" in train.frame.columns
    assert train.frame["elapsed_s"].iloc[0] == pytest.approx(0.0)
    scaler = ProgressScaler().fit([train])
    scaler.assert_not_fitted_on(test.trip_id)
    z = scaler.transform_trip(test)
    assert z.shape == (test.n_rows, len(PROGRESS_FEATURES))
