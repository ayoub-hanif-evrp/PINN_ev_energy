"""Discrete-time physics-informed residual network for EV energy estimation.

This is NOT an autodiff-in-time PINN. We do not form SOC = NN(t, v, a, θ, ...)
and take ∂SOC/∂t while holding time-varying covariates constant. That partial
is not the total derivative along the measured 1 Hz trajectory.

Instead:

    P_phy(t)  = analytical longitudinal battery-power model
    δP(t)     = residual_limit * tanh(NN_δ(x_t))
    P_hat(t)  = P_phy(t) + δP(t)
    D_hat(t)  = NN_D(x_t)   # predicted SoC depletion, SOC_start - SOC(t)

Discrete battery-energy conservation residual on interval i → i+1:

    ΔE_i = 0.5 (P_hat[i] + P_hat[i+1]) Δt_i / 3600
    r_i  = D_hat[i+1] - D_hat[i] - 100 ΔE_i / E_battery

Primary energy at inference is the trapezoidal integral of P_hat.
D_hat is used for the conservation residual, state supervision, and SoC
diagnostics. It is not claimed to be continuous ground-truth SoC.
"""

from __future__ import annotations

import torch
from torch import nn

from data.schema import MAIN_MODEL_FEATURES
from models.backbone import MLPTorso


class PINN(nn.Module):
    """Shared 8→64→64→64 tanh torso with power-residual and depletion heads."""

    def __init__(
        self,
        in_dim: int = len(MAIN_MODEL_FEATURES),
        hidden_layers: tuple[int, ...] | list[int] = (64, 64, 64),
        activation: str = "tanh",
        residual_limit_kw: float = 4.0,
    ):
        super().__init__()
        self.torso = MLPTorso(in_dim, hidden_layers, activation=activation)
        self.head_delta = nn.Linear(self.torso.out_dim, 1)
        self.head_depletion = nn.Linear(self.torso.out_dim, 1)
        self.residual_limit_kw = float(residual_limit_kw)
        nn.init.zeros_(self.head_delta.weight)
        nn.init.zeros_(self.head_delta.bias)
        nn.init.zeros_(self.head_depletion.weight)
        nn.init.zeros_(self.head_depletion.bias)

    def forward(
        self,
        x: torch.Tensor,
        p_phy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.torso(x)
        delta = self.residual_limit_kw * torch.tanh(self.head_delta(h).squeeze(-1))
        d_hat = self.head_depletion(h).squeeze(-1)
        p_hat = p_phy + delta
        return p_hat, delta, d_hat

    def saturation_fraction(self, x: torch.Tensor, thresh: float = 0.95) -> float:
        with torch.no_grad():
            h = self.torso(x)
            z = torch.tanh(self.head_delta(h).squeeze(-1)).abs()
            return float((z > thresh).float().mean())
