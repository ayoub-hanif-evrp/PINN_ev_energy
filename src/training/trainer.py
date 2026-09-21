"""Neural trainers for WeakMLP and the discrete-time PINN.

Scaler fitting, early stopping, and lambda selection use training/inner-validation
trips only. The outer test trip is never an input to those procedures.
Measured SoC is not a network input. Test-trip SoC[0] is used only after
prediction when reconstructing a diagnostic SoC trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from data.preprocessing import ProcessedTrip
from data.schema import EXCLUDED_MAIN_PREDICTORS, MAIN_MODEL_FEATURES, SOC_LABEL_COLUMNS
from data.windows import EnergyWindow, huber_delta_kwh
from models.pinn import PINN
from models.weak_mlp import WeakMLP
from physics.parameters import VehicleParameters
from physics.vehicle_model import predict_trip_physics
from training.early_stopping import EarlyStopping
from training.losses import mlp_energy_loss, pinn_losses


def _assert_features_safe(names: tuple[str, ...] | list[str]) -> None:
    forbidden = set(EXCLUDED_MAIN_PREDICTORS) | set(SOC_LABEL_COLUMNS)
    leaked = [n for n in names if n in forbidden or str(n).lower().startswith("soc")]
    if leaked:
        raise AssertionError(f"SoC/geo/time leakage in neural features: {leaked}")


@dataclass
class TripBatch:
    trip_id: str
    x: torch.Tensor
    dt: torch.Tensor
    p_phy: torch.Tensor
    d_obs: torch.Tensor | None
    starts: torch.Tensor
    ends: torch.Tensor
    e_obs: torch.Tensor
    scales: list[str]
    soc_initial: float
    feature_names: tuple[str, ...] = MAIN_MODEL_FEATURES

    def assert_no_soc_input(self) -> None:
        _assert_features_safe(self.feature_names)


def physics_power(trip: ProcessedTrip, params: VehicleParameters, config: dict[str, Any]) -> np.ndarray:
    phys = config.get("physics", {})
    result = predict_trip_physics(
        trip.frame["speed_mps"].to_numpy(dtype=float),
        trip.frame["acc_mps2"].to_numpy(dtype=float),
        trip.frame["theta_rad"].to_numpy(dtype=float),
        trip.frame["dt_s"].to_numpy(dtype=float),
        params,
        apply_bounds=bool(phys.get("apply_power_bounds", False)),
        bound_sharpness=float(phys.get("bound_sharpness", 8.0)),
    )
    return result.p_battery_kw


def make_trip_batch(
    trip: ProcessedTrip,
    x_scaled: np.ndarray,
    params: VehicleParameters,
    config: dict[str, Any],
    windows: list[EnergyWindow],
    device: torch.device,
    include_state: bool,
    window_scales: set[str] | None = None,
) -> TripBatch:
    _assert_features_safe(MAIN_MODEL_FEATURES)
    dt = torch.tensor(trip.frame["dt_s"].to_numpy(dtype=float), dtype=torch.float32, device=device)
    x = torch.tensor(np.nan_to_num(x_scaled, nan=0.0), dtype=torch.float32, device=device)
    p_phy = torch.tensor(physics_power(trip, params, config), dtype=torch.float32, device=device)
    soc_initial = 0.0
    d_obs = None
    if "soc" in trip.frame.columns:
        soc_series = trip.frame["soc"].to_numpy(dtype=float)
        soc_initial = float(soc_series[0]) if soc_series.size else 0.0
        if include_state:
            d_obs = torch.tensor(soc_series[0] - soc_series, dtype=torch.float32, device=device)
    kept = [w for w in windows if w.trip_id == trip.trip_id]
    if window_scales is not None:
        kept = [w for w in kept if w.scale in window_scales]
    if kept:
        starts = torch.tensor([w.start for w in kept], dtype=torch.long, device=device)
        ends = torch.tensor([w.end for w in kept], dtype=torch.long, device=device)
        e_obs = torch.tensor([w.e_obs_kwh for w in kept], dtype=torch.float32, device=device)
        scales = [w.scale for w in kept]
    else:
        starts = torch.zeros(0, dtype=torch.long, device=device)
        ends = torch.zeros(0, dtype=torch.long, device=device)
        e_obs = torch.zeros(0, dtype=torch.float32, device=device)
        scales = []
    return TripBatch(
        trip_id=trip.trip_id,
        x=x,
        dt=dt,
        p_phy=p_phy,
        d_obs=d_obs,
        starts=starts,
        ends=ends,
        e_obs=e_obs,
        scales=scales,
        soc_initial=soc_initial,
    )


def _lambdas(config: dict[str, Any], lambda_prior: float | None = None) -> dict[str, float]:
    tr = config.get("training", {})
    return {
        "lambda_window": float(tr.get("lambda_window", tr.get("lambda_energy", 1.0))),
        "lambda_dynamics": float(tr.get("lambda_dynamics", 1.0)),
        "lambda_state": float(tr.get("lambda_state", 0.1)),
        "lambda_boundary": float(tr.get("lambda_boundary", 1.0)),
        "lambda_prior": float(lambda_prior if lambda_prior is not None else tr.get("lambda_prior", tr.get("lambda_physics", 0.1))),
    }


def _huber_window(config: dict[str, Any], q: float, battery_capacity_kwh: float) -> float:
    wcfg = config.get("windows", {})
    return huber_delta_kwh(
        battery_capacity_kwh,
        q,
        mode=str(wcfg.get("huber_delta_mode", "quantization")),
        fixed=wcfg.get("huber_delta_kwh"),
    )


@torch.no_grad()
def _eval_window_mae(model: nn.Module, batches: list[TripBatch], kind: str) -> float:
    if not batches:
        return float("inf")
    errs: list[float] = []
    for batch in batches:
        if batch.starts.numel() == 0:
            continue
        if kind == "mlp":
            p_hat = model(batch.x)
        else:
            p_hat, _, _ = model(batch.x, batch.p_phy)
        from training.torch_ops import torch_energy_prefix, torch_window_energy

        prefix = torch_energy_prefix(p_hat, batch.dt)
        e_pred = torch_window_energy(prefix, batch.starts, batch.ends)
        errs.append(float((e_pred - batch.e_obs).abs().mean()))
    if not errs:
        return float("inf")
    return float(np.mean(errs))


def train_neural_model(
    kind: Literal["mlp", "pinn"],
    train_batches: list[TripBatch],
    val_batches: list[TripBatch],
    config: dict[str, Any],
    battery_capacity_kwh: float,
    q: float,
    seed: int,
    lambda_prior: float | None = None,
) -> dict[str, Any]:
    for b in train_batches:
        b.assert_no_soc_input()
    tr = config.get("training", {})
    hidden = tuple(int(x) for x in tr.get("hidden_layers", [64, 64, 64]))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if kind == "mlp":
        model: nn.Module = WeakMLP(
            hidden_layers=hidden,
            activation=str(tr.get("activation", "tanh")),
            power_center_kw=float(tr.get("mlp_power_center_kw", 5.0)),
            power_half_range_kw=float(tr.get("mlp_power_half_range_kw", 30.0)),
        )
    else:
        model = PINN(
            hidden_layers=hidden,
            activation=str(tr.get("activation", "tanh")),
            residual_limit_kw=float(tr.get("residual_limit_kw", 4.0)),
        )
    device = torch.device(str(config.get("experiment", {}).get("device", "cpu")))
    model.to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(tr.get("learning_rate", 1e-3)),
        weight_decay=float(tr.get("weight_decay", 1e-5)),
    )
    stopper = EarlyStopping(patience=int(tr.get("early_stopping_patience", 40)))
    lambdas = _lambdas(config, lambda_prior)
    h_win = _huber_window(config, q, battery_capacity_kwh)
    h_state = float(tr.get("huber_state_delta_pp", 0.1))
    p_scale = float(tr.get("p_scale_kw", config.get("physics", {}).get("p_scale_kw", 5.0)))
    history: list[dict[str, float]] = []
    max_epochs = int(tr.get("max_epochs", 200))
    for epoch in range(max_epochs):
        model.train()
        epoch_losses = []
        for batch in train_batches:
            opt.zero_grad()
            if kind == "mlp":
                p_hat = model(batch.x)
                loss = mlp_energy_loss(p_hat, batch.dt, batch.starts, batch.ends, batch.e_obs, batch.scales, h_win)
                logs = {"l_window": float(loss.detach())}
            else:
                p_hat, delta, d_hat = model(batch.x, batch.p_phy)
                if batch.d_obs is None:
                    raise RuntimeError("PINN training requires training-trip SoC depletion labels.")
                loss, logs = pinn_losses(
                    p_hat,
                    delta,
                    d_hat,
                    batch.d_obs,
                    batch.dt,
                    batch.starts,
                    batch.ends,
                    batch.e_obs,
                    batch.scales,
                    battery_capacity_kwh,
                    h_win,
                    h_state,
                    p_scale,
                    lambdas,
                )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))
        val_mae = _eval_window_mae(model, val_batches, kind)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(epoch_losses) if epoch_losses else float("nan")),
            "val_window_mae_kwh": float(val_mae),
        }
        history.append(row)
        if stopper.step(epoch, val_mae, model):
            break
    stopper.restore(model)
    model.eval()
    return {
        "model": model,
        "history": history,
        "best_epoch": stopper.best_epoch,
        "best_val_mae": stopper.best,
        "lambdas": lambdas,
        "kind": kind,
    }


@torch.no_grad()
def predict_power(model: nn.Module, batch: TripBatch, kind: str) -> dict[str, np.ndarray]:
    batch.assert_no_soc_input()
    model.eval()
    if kind == "mlp":
        p_hat = model(batch.x)
        sat = model.saturation_fraction(batch.x) if hasattr(model, "saturation_fraction") else float("nan")
        return {
            "p_hat": p_hat.cpu().numpy(),
            "p_phy": batch.p_phy.cpu().numpy(),
            "delta_p": np.zeros(p_hat.shape[0], dtype=float),
            "d_hat": np.full(p_hat.shape[0], np.nan),
            "saturation_frac": float(sat),
        }
    p_hat, delta, d_hat = model(batch.x, batch.p_phy)
    sat = model.saturation_fraction(batch.x) if hasattr(model, "saturation_fraction") else float("nan")
    return {
        "p_hat": p_hat.cpu().numpy(),
        "p_phy": batch.p_phy.cpu().numpy(),
        "delta_p": delta.cpu().numpy(),
        "d_hat": d_hat.cpu().numpy(),
        "saturation_frac": float(sat),
    }
