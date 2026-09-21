"""Discrete-time physics-informed residual network for EV energy estimation.

This is NOT an autodiff-in-time PDE PINN. We do not form SOC = NN(t, v, a, θ, ...)
and take ∂SOC/∂t while holding time-varying covariates constant. That partial
is not the total derivative along the measured 1 Hz trajectory.

Architecture:

    x_t  = instantaneous telemetry (no SoC, no LAT/LON, no clock time)
    h_t  = shared 64-64-64 tanh torso(x_t)
    δP   = residual_limit * tanh(head_delta(h_t))
    P_hat = P_physics + δP

    progress_t = (elapsed_time_s, cumulative_distance_km)  # causal, trip-start → t
    D_raw(t)   = state_head(concat(h_t, progress_t))
    D_hat(t)   = D_raw(t) - D_raw(0)     # D_hat(0) = 0 by construction

Discrete battery-energy conservation residual on interval i → i+1:

    ΔE_i = 0.5 (P_hat[i] + P_hat[i+1]) Δt_i / 3600
    r_i  = D_hat[i+1] - D_hat[i] - 100 ΔE_i / E_battery

Primary energy at inference is the trapezoidal integral of P_hat.
D_hat is used for the conservation residual, state supervision, and SoC
diagnostics. Regenerative braking means D_hat is not required to be monotonic.
"""

from __future__ import annotations

import torch
from torch import nn

from data.schema import MAIN_MODEL_FEATURES, PROGRESS_FEATURES
from models.backbone import MLPTorso


class PINN(nn.Module):
    """Shared telemetry torso, physics residual power head, causal state head."""

    def __init__(
        self,
        in_dim: int = len(MAIN_MODEL_FEATURES),
        progress_dim: int = len(PROGRESS_FEATURES),
        hidden_layers: tuple[int, ...] | list[int] = (64, 64, 64),
        activation: str = "tanh",
        residual_limit_kw: float = 4.0,
    ):
        super().__init__()
        self.torso = MLPTorso(in_dim, hidden_layers, activation=activation)
        self.head_delta = nn.Linear(self.torso.out_dim, 1)
        self.state_in = nn.Linear(self.torso.out_dim + int(progress_dim), self.torso.out_dim)
        self.state_out = nn.Linear(self.torso.out_dim, 1)
        self.residual_limit_kw = float(residual_limit_kw)
        nn.init.zeros_(self.head_delta.weight)
        nn.init.zeros_(self.head_delta.bias)
        nn.init.xavier_uniform_(self.state_in.weight, gain=0.1)
        nn.init.zeros_(self.state_in.bias)
        nn.init.xavier_uniform_(self.state_out.weight, gain=0.1)
        nn.init.zeros_(self.state_out.bias)

    def _state_raw(self, h: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        z = torch.cat([h, progress], dim=-1)
        return self.state_out(torch.tanh(self.state_in(z))).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        p_phy: torch.Tensor,
        progress: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.torso(x)
        delta = self.residual_limit_kw * torch.tanh(self.head_delta(h).squeeze(-1))
        d_raw = self._state_raw(h, progress)
        d_hat = d_raw - d_raw[0]
        p_hat = p_phy + delta
        return p_hat, delta, d_hat

    def saturation_fraction(self, x: torch.Tensor, thresh: float = 0.95) -> float:
        with torch.no_grad():
            h = self.torso(x)
            z = torch.tanh(self.head_delta(h).squeeze(-1)).abs()
            return float((z > thresh).float().mean())
