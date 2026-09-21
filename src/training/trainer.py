"""Neural trainers for WeakMLP, MLP_STATE, and the discrete-time PINN.

Scaler fitting, early stopping, q, and lambda selection use training/inner
validation trips only. The outer test trip is never an input to those procedures.
Measured SoC is not a network input. Test-trip SoC[0] is used only after
prediction when reconstructing a diagnostic SoC trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from data.preprocessing import ProcessedTrip
from data.schema import EXCLUDED_MAIN_PREDICTORS, MAIN_MODEL_FEATURES, PROGRESS_FEATURES, SOC_LABEL_COLUMNS
from data.windows import EnergyWindow, huber_delta_kwh
from models.mlp_state import MLPState
from models.pinn import PINN
from models.weak_mlp import WeakMLP
from physics.parameters import VehicleParameters
from physics.vehicle_model import predict_trip_physics
from training.early_stopping import EarlyStopping
from training.losses import mlp_energy_loss, mlp_state_losses, pinn_losses, scale_balanced_window_mae
from training.torch_ops import torch_energy_prefix, torch_window_energy

NeuralKind = Literal["mlp", "mlp_state", "pinn", "pinn_no_dynamics"]
NEURAL_KINDS = ("mlp", "mlp_state", "pinn", "pinn_no_dynamics")


def _assert_features_safe(names: tuple[str, ...] | list[str]) -> None:
    forbidden = set(EXCLUDED_MAIN_PREDICTORS) | set(SOC_LABEL_COLUMNS)
    leaked = [n for n in names if n in forbidden or str(n).lower().startswith("soc")]
    if leaked:
        raise AssertionError(f"SoC/geo/time leakage in neural features: {leaked}")


def method_kind(method: str) -> NeuralKind:
    if method in {"weak_mlp", "mlp"}:
        return "mlp"
    if method == "mlp_state":
        return "mlp_state"
    if method == "pinn_no_dynamics":
        return "pinn_no_dynamics"
    if method in {"pinn", "pinn_fulltrip"}:
        return "pinn"
    raise ValueError(f"Unknown neural method {method}")


def uses_progress(kind: str) -> bool:
    return kind in {"mlp_state", "pinn", "pinn_no_dynamics"}


def uses_physics(kind: str) -> bool:
    return kind in {"pinn", "pinn_no_dynamics"}


def uses_state(kind: str) -> bool:
    return kind in {"mlp_state", "pinn", "pinn_no_dynamics"}


def window_scale_filter(method: str) -> set[str] | None:
    if method == "pinn_fulltrip":
        return {"full_trip"}
    return None


@dataclass
class TripBatch:
    trip_id: str
    x: torch.Tensor
    dt: torch.Tensor
    p_phy: torch.Tensor
    progress: torch.Tensor
    d_obs: torch.Tensor | None
    starts: torch.Tensor
    ends: torch.Tensor
    e_obs: torch.Tensor
    scales: list[str]
    feature_names: tuple[str, ...] = MAIN_MODEL_FEATURES
    progress_names: tuple[str, ...] = PROGRESS_FEATURES
    contains_measured_soc: bool = False

    def assert_no_soc_input(self) -> None:
        _assert_features_safe(self.feature_names)
        _assert_features_safe(self.progress_names)


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


def _window_tensors(
    trip: ProcessedTrip,
    windows: list[EnergyWindow],
    device: torch.device,
    window_scales: set[str] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
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
    return starts, ends, e_obs, scales


def make_trip_batch(
    trip: ProcessedTrip,
    x_scaled: np.ndarray,
    params: VehicleParameters,
    config: dict[str, Any],
    windows: list[EnergyWindow],
    device: torch.device,
    include_state: bool,
    window_scales: set[str] | None = None,
    progress_scaled: np.ndarray | None = None,
) -> TripBatch:
    """Training/validation batch. Measured SoC is labels only when include_state."""
    _assert_features_safe(MAIN_MODEL_FEATURES)
    dt = torch.tensor(trip.frame["dt_s"].to_numpy(dtype=float), dtype=torch.float32, device=device)
    x = torch.tensor(np.nan_to_num(x_scaled, nan=0.0), dtype=torch.float32, device=device)
    p_phy = torch.tensor(physics_power(trip, params, config), dtype=torch.float32, device=device)
    if progress_scaled is None:
        progress = torch.zeros((x.shape[0], len(PROGRESS_FEATURES)), dtype=torch.float32, device=device)
    else:
        progress = torch.tensor(np.nan_to_num(progress_scaled, nan=0.0), dtype=torch.float32, device=device)
    d_obs = None
    contains_soc = False
    if include_state:
        if "soc" not in trip.frame.columns:
            raise RuntimeError(f"{trip.trip_id}: state supervision requested but SoC is missing.")
        soc_series = trip.frame["soc"].to_numpy(dtype=float)
        d_obs = torch.tensor(soc_series[0] - soc_series, dtype=torch.float32, device=device)
        contains_soc = True
    starts, ends, e_obs, scales = _window_tensors(trip, windows, device, window_scales)
    return TripBatch(
        trip_id=trip.trip_id,
        x=x,
        dt=dt,
        p_phy=p_phy,
        progress=progress,
        d_obs=d_obs,
        starts=starts,
        ends=ends,
        e_obs=e_obs,
        scales=scales,
        contains_measured_soc=contains_soc,
    )


def make_inference_batch(
    trip: ProcessedTrip,
    x_scaled: np.ndarray,
    params: VehicleParameters,
    config: dict[str, Any],
    device: torch.device,
    progress_scaled: np.ndarray | None = None,
) -> TripBatch:
    """Test inference: features, dt, P_physics, progress. No measured SoC."""
    _assert_features_safe(MAIN_MODEL_FEATURES)
    dt = torch.tensor(trip.frame["dt_s"].to_numpy(dtype=float), dtype=torch.float32, device=device)
    x = torch.tensor(np.nan_to_num(x_scaled, nan=0.0), dtype=torch.float32, device=device)
    p_phy = torch.tensor(physics_power(trip, params, config), dtype=torch.float32, device=device)
    if progress_scaled is None:
        progress = torch.zeros((x.shape[0], len(PROGRESS_FEATURES)), dtype=torch.float32, device=device)
    else:
        progress = torch.tensor(np.nan_to_num(progress_scaled, nan=0.0), dtype=torch.float32, device=device)
    empty = torch.zeros(0, dtype=torch.long, device=device)
    return TripBatch(
        trip_id=trip.trip_id,
        x=x,
        dt=dt,
        p_phy=p_phy,
        progress=progress,
        d_obs=None,
        starts=empty,
        ends=empty,
        e_obs=torch.zeros(0, dtype=torch.float32, device=device),
        scales=[],
        contains_measured_soc=False,
    )


def _lambdas(config: dict[str, Any], kind: str, lambda_prior: float | None = None) -> dict[str, float]:
    tr = config.get("training", {})
    lam = {
        "lambda_window": float(tr.get("lambda_window", tr.get("lambda_energy", 1.0))),
        "lambda_dynamics": float(tr.get("lambda_dynamics", 1.0)),
        "lambda_state": float(tr.get("lambda_state", 0.1)),
        "lambda_boundary": float(tr.get("lambda_boundary", 0.0)),
        "lambda_prior": float(lambda_prior if lambda_prior is not None else tr.get("lambda_prior", tr.get("lambda_physics", 0.1))),
    }
    if kind == "pinn_no_dynamics":
        lam["lambda_dynamics"] = 0.0
    if kind == "mlp_state":
        lam["lambda_dynamics"] = 0.0
        lam["lambda_prior"] = 0.0
        lam["lambda_boundary"] = 0.0
    if kind == "mlp":
        lam["lambda_dynamics"] = 0.0
        lam["lambda_state"] = 0.0
        lam["lambda_prior"] = 0.0
        lam["lambda_boundary"] = 0.0
    return lam


def _huber_window(config: dict[str, Any], q: float, battery_capacity_kwh: float) -> float:
    wcfg = config.get("windows", {})
    return huber_delta_kwh(
        battery_capacity_kwh,
        q,
        mode=str(wcfg.get("huber_delta_mode", "quantization")),
        fixed=wcfg.get("huber_delta_kwh"),
    )


def build_model(kind: str, config: dict[str, Any], seed: int) -> nn.Module:
    tr = config.get("training", {})
    hidden = tuple(int(x) for x in tr.get("hidden_layers", [64, 64, 64]))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if kind == "mlp":
        return WeakMLP(
            hidden_layers=hidden,
            activation=str(tr.get("activation", "tanh")),
            power_center_kw=float(tr.get("mlp_power_center_kw", 5.0)),
            power_half_range_kw=float(tr.get("mlp_power_half_range_kw", 30.0)),
        )
    if kind == "mlp_state":
        return MLPState(
            hidden_layers=hidden,
            activation=str(tr.get("activation", "tanh")),
            power_center_kw=float(tr.get("mlp_power_center_kw", 5.0)),
            power_half_range_kw=float(tr.get("mlp_power_half_range_kw", 30.0)),
        )
    return PINN(
        hidden_layers=hidden,
        activation=str(tr.get("activation", "tanh")),
        residual_limit_kw=float(tr.get("residual_limit_kw", 4.0)),
    )


def _forward(model: nn.Module, batch: TripBatch, kind: str) -> dict[str, torch.Tensor]:
    if kind == "mlp":
        p_hat = model(batch.x)
        return {"p_hat": p_hat, "delta": torch.zeros_like(p_hat), "d_hat": torch.full_like(p_hat, float("nan"))}
    if kind == "mlp_state":
        p_hat, d_hat = model(batch.x, batch.progress)
        return {"p_hat": p_hat, "delta": torch.zeros_like(p_hat), "d_hat": d_hat}
    p_hat, delta, d_hat = model(batch.x, batch.p_phy, batch.progress)
    return {"p_hat": p_hat, "delta": delta, "d_hat": d_hat}


def _batch_loss(
    out: dict[str, torch.Tensor],
    batch: TripBatch,
    kind: str,
    battery_capacity_kwh: float,
    h_win: float,
    h_state: float,
    p_scale: float,
    lambdas: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    if kind == "mlp":
        loss = mlp_energy_loss(out["p_hat"], batch.dt, batch.starts, batch.ends, batch.e_obs, batch.scales, h_win)
        logs = {"l_window": float(loss.detach()), "w_window": float(lambdas["lambda_window"]) * float(loss.detach())}
        return loss, logs
    if batch.d_obs is None:
        raise RuntimeError("State-supervised training requires training-trip SoC depletion labels.")
    if kind == "mlp_state":
        return mlp_state_losses(
            out["p_hat"],
            out["d_hat"],
            batch.d_obs,
            batch.dt,
            batch.starts,
            batch.ends,
            batch.e_obs,
            batch.scales,
            h_win,
            h_state,
            lambdas,
        )
    return pinn_losses(
        out["p_hat"],
        out["delta"],
        out["d_hat"],
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


@torch.no_grad()
def eval_scale_balanced_mae(model: nn.Module, batches: list[TripBatch], kind: str) -> tuple[float, dict[str, float]]:
    if not batches:
        return float("inf"), {}
    preds: list[torch.Tensor] = []
    obs: list[torch.Tensor] = []
    scales: list[str] = []
    model.eval()
    for batch in batches:
        if batch.starts.numel() == 0:
            continue
        out = _forward(model, batch, kind)
        prefix = torch_energy_prefix(out["p_hat"], batch.dt)
        e_pred = torch_window_energy(prefix, batch.starts, batch.ends)
        preds.append(e_pred)
        obs.append(batch.e_obs)
        scales.extend(batch.scales)
    if not preds:
        return float("inf"), {}
    return scale_balanced_window_mae(torch.cat(preds), torch.cat(obs), scales)


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.data.norm(2) ** 2)
    return float(total**0.5)


def train_neural_model(
    kind: NeuralKind,
    train_batches: list[TripBatch],
    val_batches: list[TripBatch],
    config: dict[str, Any],
    battery_capacity_kwh: float,
    q: float,
    seed: int,
    lambda_prior: float | None = None,
    max_epochs: int | None = None,
    early_stopping: bool = True,
) -> dict[str, Any]:
    for b in train_batches:
        b.assert_no_soc_input()
    tr = config.get("training", {})
    model = build_model(kind, config, seed)
    device = next(iter(train_batches)).x.device if train_batches else torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(tr.get("learning_rate", 1e-3)),
        weight_decay=float(tr.get("weight_decay", 1e-5)),
    )
    stopper = EarlyStopping(patience=int(tr.get("early_stopping_patience", 40)))
    lambdas = _lambdas(config, kind, lambda_prior)
    h_win = _huber_window(config, q, battery_capacity_kwh)
    h_state = float(tr.get("huber_state_delta_pp", 0.1))
    p_scale = float(tr.get("p_scale_kw", config.get("physics", {}).get("p_scale_kw", 5.0)))
    history: list[dict[str, float]] = []
    last_finite_loss = float("nan")
    n_epochs = int(max_epochs if max_epochs is not None else tr.get("max_epochs", 200))
    for epoch in range(n_epochs):
        model.train()
        epoch_logs: list[dict[str, float]] = []
        epoch_losses: list[float] = []
        grad_norms: list[float] = []
        for batch in train_batches:
            opt.zero_grad()
            out = _forward(model, batch, kind)
            loss, logs = _batch_loss(out, batch, kind, battery_capacity_kwh, h_win, h_state, p_scale, lambdas)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            grad_norms.append(_grad_norm(model))
            opt.step()
            epoch_losses.append(float(loss.detach()))
            last_finite_loss = float(loss.detach())
            epoch_logs.append(logs)
        val_mae, per_scale = eval_scale_balanced_mae(model, val_batches, kind) if val_batches else (float("nan"), {})
        merged = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(epoch_losses) if epoch_losses else float("nan")),
            "val_window_mae_kwh": float(val_mae),
            "grad_norm": float(np.mean(grad_norms) if grad_norms else float("nan")),
        }
        if epoch_logs:
            keys = set().union(*(d.keys() for d in epoch_logs))
            for key in keys:
                vals = [d[key] for d in epoch_logs if key in d and np.isfinite(d[key])]
                if vals:
                    merged[key] = float(np.mean(vals))
        for scale, mae in per_scale.items():
            merged[f"val_mae_{scale}"] = mae
        history.append(merged)
        if early_stopping and val_batches:
            if stopper.step(epoch, val_mae, model):
                break
        else:
            stopper.step(epoch, merged["train_loss"] if np.isfinite(merged["train_loss"]) else float("inf"), model)
    if early_stopping and val_batches:
        stopper.restore(model)
        selected_epoch = stopper.best_epoch
        best_val = stopper.best
    else:
        selected_epoch = n_epochs - 1 if n_epochs else 0
        best_val = float("nan")
    model.eval()
    return {
        "model": model,
        "history": history,
        "best_epoch": int(selected_epoch),
        "best_val_mae": best_val,
        "lambdas": lambdas,
        "kind": kind,
        "last_finite_loss": last_finite_loss,
        "n_epochs_run": len(history),
    }


@torch.no_grad()
def predict_power(model: nn.Module, batch: TripBatch, kind: str) -> dict[str, np.ndarray]:
    batch.assert_no_soc_input()
    if batch.contains_measured_soc:
        raise AssertionError("Inference batch must not carry measured SoC labels.")
    model.eval()
    out = _forward(model, batch, kind)
    sat = model.saturation_fraction(batch.x) if hasattr(model, "saturation_fraction") else float("nan")
    d_hat = out["d_hat"]
    return {
        "p_hat": out["p_hat"].cpu().numpy(),
        "p_phy": batch.p_phy.cpu().numpy(),
        "delta_p": out["delta"].cpu().numpy(),
        "d_hat": d_hat.cpu().numpy() if torch.isfinite(d_hat).any() else np.full(out["p_hat"].shape[0], np.nan),
        "saturation_frac": float(sat),
    }
