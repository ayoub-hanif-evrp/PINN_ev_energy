"""Pytest path and synthetic HELECAR-like fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config
from physics.parameters import VehicleParameters


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def project_config():
    return load_config(ROOT / "configs" / "base.yaml")


@pytest.fixture
def vehicle_params() -> VehicleParameters:
    return VehicleParameters(
        battery_capacity_kwh=6.0,
        mass_kg=549.0,
        Crr=0.012,
        cd=0.64,
        frontal_area_m2=1.20,
        rho_air=1.225,
        eta_drive=0.85,
        eta_regen=0.40,
        auxiliary_power_kw=0.20,
        maximum_traction_power_kw=13.0,
        maximum_regen_power_kw=8.0,
        g=9.80665,
    )
