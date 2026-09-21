"""Training losses for the weakly supervised MLP and discrete-time PINN."""

from __future__ import annotations

from collections import defaultdict

import torch

from training.torch_ops import torch_energy_prefix, torch_interval_increments_kwh, torch_window_energy


def huber(residual: torch.Tensor, delta: float) -> torch.Tensor:
    delta_t = max(float(delta), 1e-12)
    abs_r = residual.abs()
    quad = 0.5 * residual * residual / delta_t
    lin = abs_r - 0.5 * delta_t
    return torch.where(abs_r <= delta_t, quad, lin)


def scale_balanced_window_loss(
    e_pred: torch.Tensor,
    e_obs: torch.Tensor,
    scales: list[str],
    delta: float,
) -> torch.Tensor:
    """Mean of per-scale Huber means so frequent scales cannot dominate."""
    if e_pred.numel() == 0:
        return e_pred.sum() * 0.0
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, scale in enumerate(scales):
        grouped[str(scale)].append(i)
    means = []
    for idxs in grouped.values():
        idx = torch.tensor(idxs, device=e_pred.device, dtype=torch.long)
        means.append(huber(e_pred[idx] - e_obs[idx], delta).mean())
    return torch.stack(means).mean()


def scale_balanced_window_mae(
    e_pred: torch.Tensor,
    e_obs: torch.Tensor,
    scales: list[str],
) -> tuple[float, dict[str, float]]:
    """Validation score: mean over scales of MAE_s. pinn_fulltrip has one scale."""
    if e_pred.numel() == 0:
        return float("inf"), {}
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, scale in enumerate(scales):
        grouped[str(scale)].append(i)
    per: dict[str, float] = {}
    for scale, idxs in grouped.items():
        idx = torch.tensor(idxs, device=e_pred.device, dtype=torch.long)
        per[scale] = float((e_pred[idx] - e_obs[idx]).abs().mean())
    return float(sum(per.values()) / len(per)), per


def window_energy_loss(
    power_kw: torch.Tensor,
    dt_s: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    e_obs: torch.Tensor,
    scales: list[str],
    delta: float,
) -> torch.Tensor:
    prefix = torch_energy_prefix(power_kw, dt_s)
    e_pred = torch_window_energy(prefix, starts, ends)
    return scale_balanced_window_loss(e_pred, e_obs, scales, delta)


def mlp_energy_loss(
    power_kw: torch.Tensor,
    dt_s: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    e_obs: torch.Tensor,
    scales: list[str],
    delta: float,
) -> torch.Tensor:
    if starts.numel() == 0:
        return power_kw.sum() * 0.0
    return window_energy_loss(power_kw, dt_s, starts, ends, e_obs, scales, delta)


def _weighted_logs(raw: dict[str, torch.Tensor], lambdas: dict[str, float], total: torch.Tensor) -> dict[str, float]:
    logs = {"loss": float(total.detach())}
    for key, term in raw.items():
        logs[f"l_{key}"] = float(term.detach())
        lam_name = f"lambda_{key}"
        if lam_name in lambdas:
            logs[f"w_{key}"] = float(lambdas[lam_name]) * float(term.detach())
    return logs


def mlp_state_losses(
    p_hat: torch.Tensor,
    d_hat: torch.Tensor,
    d_obs: torch.Tensor,
    dt_s: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    e_obs: torch.Tensor,
    scales: list[str],
    huber_window: float,
    huber_state: float,
    lambdas: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    l_window = mlp_energy_loss(p_hat, dt_s, starts, ends, e_obs, scales, huber_window)
    l_state = huber(d_hat - d_obs, huber_state).mean()
    total = float(lambdas["lambda_window"]) * l_window + float(lambdas["lambda_state"]) * l_state
    return total, _weighted_logs({"window": l_window, "state": l_state}, lambdas, total)


def pinn_losses(
    p_hat: torch.Tensor,
    delta_p: torch.Tensor,
    d_hat: torch.Tensor,
    d_obs: torch.Tensor,
    dt_s: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    e_obs: torch.Tensor,
    scales: list[str],
    battery_capacity_kwh: float,
    huber_window: float,
    huber_state: float,
    p_scale_kw: float,
    lambdas: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Discrete-time physics-informed residual.

    Interval energy:
        ΔE_i = 0.5 (P̂_i + P̂_{i+1}) Δt_i / 3600
    Conservation residual:
        r_i = D̂_{i+1} - D̂_i - 100 ΔE_i / E_battery
    where D̂ is predicted SoC depletion with D̂(0)=0 by construction.

    L_PINN = λ_window L_window + λ_dynamics L_dynamics
             + λ_state L_state + λ_prior L_prior

    L_boundary is logged as a diagnostic (should be ~0) and is not a
    meaningful optimisation term when the initial condition is hard-coded.
    """
    l_window = mlp_energy_loss(p_hat, dt_s, starts, ends, e_obs, scales, huber_window)
    de = torch_interval_increments_kwh(p_hat, dt_s)
    if p_hat.numel() >= 2:
        r = d_hat[1:] - d_hat[:-1] - (100.0 / float(battery_capacity_kwh)) * de[:-1]
        l_dyn = (r * r).mean()
        dyn_rmse = torch.sqrt((r * r).mean())
        dyn_mae = r.abs().mean()
    else:
        l_dyn = p_hat.sum() * 0.0
        dyn_rmse = l_dyn
        dyn_mae = l_dyn
    l_state = huber(d_hat - d_obs, huber_state).mean()
    l_boundary = d_hat[0] * d_hat[0]
    scale = max(float(p_scale_kw), 1e-6)
    l_prior = ((delta_p / scale) ** 2).mean()
    total = (
        float(lambdas["lambda_window"]) * l_window
        + float(lambdas["lambda_dynamics"]) * l_dyn
        + float(lambdas["lambda_state"]) * l_state
        + float(lambdas.get("lambda_boundary", 0.0)) * l_boundary
        + float(lambdas["lambda_prior"]) * l_prior
    )
    logs = _weighted_logs(
        {
            "window": l_window,
            "dynamics": l_dyn,
            "state": l_state,
            "prior": l_prior,
            "boundary": l_boundary,
        },
        lambdas,
        total,
    )
    logs["dynamics_residual_rmse"] = float(dyn_rmse.detach())
    logs["dynamics_residual_mae"] = float(dyn_mae.detach())
    return total, logs
