"""Weakly supervised MLP: latent battery power from aggregate SoC-window energy.

The network never sees instantaneous battery-power labels. Training uses
trapezoidal integrals of P_hat versus SoC-derived window energy.
"""

from __future__ import annotations

import torch
from torch import nn

from data.schema import MAIN_MODEL_FEATURES
from models.backbone import MLPTorso


class WeakMLP(nn.Module):
    """8 → 64 tanh → 64 tanh → 64 tanh → 1, then a broad tanh power range.

    P_hat = center + half_range * tanh(raw)

    Bounds are physically broad so the data-only MLP is not handicapped versus
    the PINN. They are not tuned from outer-test performance.
    """

    def __init__(
        self,
        in_dim: int = len(MAIN_MODEL_FEATURES),
        hidden_layers: tuple[int, ...] | list[int] = (64, 64, 64),
        activation: str = "tanh",
        power_center_kw: float = 5.0,
        power_half_range_kw: float = 30.0,
    ):
        super().__init__()
        self.torso = MLPTorso(in_dim, hidden_layers, activation=activation)
        self.head = nn.Linear(self.torso.out_dim, 1)
        self.power_center_kw = float(power_center_kw)
        self.power_half_range_kw = float(power_half_range_kw)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.torso(x)).squeeze(-1)
        return self.power_center_kw + self.power_half_range_kw * torch.tanh(raw)

    def saturation_fraction(self, x: torch.Tensor, thresh: float = 0.95) -> float:
        with torch.no_grad():
            raw = self.head(self.torso(x)).squeeze(-1)
            z = torch.tanh(raw).abs()
            return float((z > thresh).float().mean())
