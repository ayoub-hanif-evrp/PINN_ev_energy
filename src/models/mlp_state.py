"""Ablation: window + state-level SoC supervision, no analytical physics.

MLP_STATE uses the same telemetry torso, the same causal progress features for
its depletion head, the same window energy labels, and the same D_obs
supervision as the PINN. It does NOT add P_physics, a deltaP prior, or a
discrete conservation residual. Primary energy still comes from a learned
power branch.
"""

from __future__ import annotations

import torch
from torch import nn

from data.schema import MAIN_MODEL_FEATURES, PROGRESS_FEATURES
from models.backbone import MLPTorso


class MLPState(nn.Module):
    def __init__(
        self,
        in_dim: int = len(MAIN_MODEL_FEATURES),
        progress_dim: int = len(PROGRESS_FEATURES),
        hidden_layers: tuple[int, ...] | list[int] = (64, 64, 64),
        activation: str = "tanh",
        power_center_kw: float = 5.0,
        power_half_range_kw: float = 30.0,
    ):
        super().__init__()
        self.torso = MLPTorso(in_dim, hidden_layers, activation=activation)
        self.head_power = nn.Linear(self.torso.out_dim, 1)
        self.state_in = nn.Linear(self.torso.out_dim + int(progress_dim), self.torso.out_dim)
        self.state_out = nn.Linear(self.torso.out_dim, 1)
        self.power_center_kw = float(power_center_kw)
        self.power_half_range_kw = float(power_half_range_kw)
        nn.init.zeros_(self.head_power.weight)
        nn.init.zeros_(self.head_power.bias)
        nn.init.xavier_uniform_(self.state_in.weight, gain=0.1)
        nn.init.zeros_(self.state_in.bias)
        nn.init.xavier_uniform_(self.state_out.weight, gain=0.1)
        nn.init.zeros_(self.state_out.bias)

    def forward(self, x: torch.Tensor, progress: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.torso(x)
        raw = self.head_power(h).squeeze(-1)
        p_hat = self.power_center_kw + self.power_half_range_kw * torch.tanh(raw)
        d_raw = self.state_out(torch.tanh(self.state_in(torch.cat([h, progress], dim=-1)))).squeeze(-1)
        d_hat = d_raw - d_raw[0]
        return p_hat, d_hat

    def saturation_fraction(self, x: torch.Tensor, thresh: float = 0.95) -> float:
        with torch.no_grad():
            raw = self.head_power(self.torso(x)).squeeze(-1)
            z = torch.tanh(raw).abs()
            return float((z > thresh).float().mean())
