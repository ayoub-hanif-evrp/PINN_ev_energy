"""Shared tanh MLP torso used by both WeakMLP and the discrete-time PINN."""

from __future__ import annotations

import torch
from torch import nn


class MLPTorso(nn.Module):
    def __init__(self, in_dim: int, hidden_layers: tuple[int, ...] | list[int], activation: str = "tanh"):
        super().__init__()
        if activation != "tanh":
            raise ValueError("Fair MLP/PINN comparison uses tanh; other activations are not enabled.")
        layers: list[nn.Module] = []
        last = int(in_dim)
        for width in hidden_layers:
            layers.append(nn.Linear(last, int(width)))
            layers.append(nn.Tanh())
            last = int(width)
        self.net = nn.Sequential(*layers)
        self.out_dim = last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
