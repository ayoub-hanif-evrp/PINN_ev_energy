"""Torch trapezoidal integration matches NumPy and is differentiable."""

from __future__ import annotations

import numpy as np
import torch

from training.torch_ops import torch_energy_prefix
from units import energy_prefix_kwh


def test_torch_prefix_matches_numpy():
    rng = np.random.default_rng(1)
    p = rng.normal(size=80)
    dt = np.zeros(80)
    dt[:-1] = rng.uniform(0.8, 1.2, size=79)
    numpy_prefix = energy_prefix_kwh(p, dt)
    torch_prefix = torch_energy_prefix(torch.tensor(p, dtype=torch.float64), torch.tensor(dt, dtype=torch.float64))
    np.testing.assert_allclose(torch_prefix.numpy(), numpy_prefix, atol=1e-12)


def test_torch_prefix_has_gradient():
    p = torch.randn(40, requires_grad=True)
    dt = torch.ones(40)
    dt[-1] = 0.0
    prefix = torch_energy_prefix(p, dt)
    prefix[-1].backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0
