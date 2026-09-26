"""Canonical method names, display labels, and plot styling."""

from __future__ import annotations

METHOD_ORDER = [
    "physics",
    "ridge",
    "weak_mlp",
    "pinn",
]

METHOD_LABELS = {
    "physics": "Physics Model",
    "ridge": "Ridge Regression",
    "weak_mlp": "Data-Driven MLP",
    "pinn": "PINN",
}

METHOD_COLORS = {
    "physics": "#7a7a7a",
    "ridge": "#000000",
    "weak_mlp": "#d55e00",
    "pinn": "#0072b2",
}

METHOD_MARKERS = {
    "physics": "s",
    "ridge": "o",
    "weak_mlp": "^",
    "pinn": "o",
}

METHOD_LINESTYLES = {
    "physics": "--",
    "ridge": "-",
    "weak_mlp": "-.",
    "pinn": "-",
}

PAPER_METHODS = ["physics", "ridge", "weak_mlp", "pinn"]
PARITY_METHODS = ["physics", "ridge", "weak_mlp", "pinn"]
SCARCITY_METHODS = ["ridge", "weak_mlp", "pinn"]
FEASIBILITY_METHODS = ["physics", "pinn"]


def label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method))
