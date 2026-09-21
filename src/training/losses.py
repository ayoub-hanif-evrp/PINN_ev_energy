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
    where D̂ is predicted SoC depletion (SOC_start - SOC(t)), independent of
    an autodiff ∂SOC/∂t that would freeze time-varying covariates.

    L = λ_window L_window + λ_dynamics L_dynamics + λ_state L_state
        + λ_boundary L_boundary + λ_prior L_prior
    """
    l_window = mlp_energy_loss(p_hat, dt_s, starts, ends, e_obs, scales, huber_window)
    de = torch_interval_increments_kwh(p_hat, dt_s)
    if p_hat.numel() >= 2:
        r = d_hat[1:] - d_hat[:-1] - (100.0 / float(battery_capacity_kwh)) * de[:-1]
        l_dyn = (r * r).mean()
    else:
        l_dyn = p_hat.sum() * 0.0
    l_state = huber(d_hat - d_obs, huber_state).mean()
    l_boundary = d_hat[0] * d_hat[0]
    scale = max(float(p_scale_kw), 1e-6)
    l_prior = ((delta_p / scale) ** 2).mean()
    total = (
        float(lambdas["lambda_window"]) * l_window
        + float(lambdas["lambda_dynamics"]) * l_dyn
        + float(lambdas["lambda_state"]) * l_state
        + float(lambdas["lambda_boundary"]) * l_boundary
        + float(lambdas["lambda_prior"]) * l_prior
    )
    logs = {
        "loss": float(total.detach()),
        "l_window": float(l_window.detach()),
        "l_dynamics": float(l_dyn.detach()),
        "l_state": float(l_state.detach()),
        "l_boundary": float(l_boundary.detach()),
        "l_prior": float(l_prior.detach()),
    }
    return total, logs
