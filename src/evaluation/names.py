"""Canonical method names, display labels, and plot styling."""

from __future__ import annotations

METHOD_ORDER = [
    "constant",
    "physics",
    "elasticnet",
    "weak_mlp",
    "mlp_state",
    "pinn_no_dynamics",
    "pinn_fulltrip",
    "pinn",
]

METHOD_LABELS = {
    "constant": "Constant",
    "physics": "Physics Model",
    "elasticnet": "Regularized Regression",
    "weak_mlp": "Data-Driven MLP",
    "mlp_state": "MLP_STATE",
    "pinn_no_dynamics": "PINN_NO_DYNAMICS",
    "pinn_fulltrip": "PINN_FULLTRIP",
    "pinn": "PINN",
}

# Colorblind-friendly, remaining distinguishable in grayscale via markers/linestyles.
METHOD_COLORS = {
    "constant": "#4d4d4d",
    "physics": "#7a7a7a",
    "elasticnet": "#000000",
    "weak_mlp": "#d55e00",
    "mlp_state": "#cc79a7",
    "pinn_no_dynamics": "#009e73",
    "pinn_fulltrip": "#0072b2",
    "pinn": "#0072b2",
}

METHOD_MARKERS = {
    "constant": "D",
    "physics": "s",
    "elasticnet": "o",
    "weak_mlp": "^",
    "mlp_state": "v",
    "pinn_no_dynamics": "P",
    "pinn_fulltrip": "X",
    "pinn": "o",
}

METHOD_LINESTYLES = {
    "constant": ":",
    "physics": "--",
    "elasticnet": "-",
    "weak_mlp": "-.",
    "mlp_state": ":",
    "pinn_no_dynamics": "--",
    "pinn_fulltrip": "-.",
    "pinn": "-",
}

PAPER_METHODS = ["physics", "elasticnet", "weak_mlp", "pinn"]
ABLATION_ORDER = ["weak_mlp", "mlp_state", "pinn_no_dynamics", "pinn_fulltrip", "pinn"]
PARITY_METHODS = ["physics", "elasticnet", "weak_mlp", "pinn"]
SCARCITY_METHODS = ["elasticnet", "weak_mlp", "pinn"]
FEASIBILITY_METHODS = ["physics", "pinn"]


def label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method))
