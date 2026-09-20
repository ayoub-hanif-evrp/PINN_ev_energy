"""Twizy physical parameter container. Values come from YAML, not code constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from paths import project_root, resolve_under_root
from units import STANDARD_GRAVITY


@dataclass(frozen=True)
class VehicleParameters:
    battery_capacity_kwh: float
    mass_kg: float
    Crr: float
    cd: float
    frontal_area_m2: float
    rho_air: float
    eta_drive: float
    eta_regen: float
    auxiliary_power_kw: float
    maximum_traction_power_kw: float
    maximum_regen_power_kw: float
    g: float = STANDARD_GRAVITY

    @property
    def cda_m2(self) -> float:
        return self.cd * self.frontal_area_m2

    def as_dict(self) -> dict[str, float]:
        return {
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "mass_kg": self.mass_kg,
            "Crr": self.Crr,
            "cd": self.cd,
            "frontal_area_m2": self.frontal_area_m2,
            "cda_m2": self.cda_m2,
            "rho_air": self.rho_air,
            "eta_drive": self.eta_drive,
            "eta_regen": self.eta_regen,
            "auxiliary_power_kw": self.auxiliary_power_kw,
            "maximum_traction_power_kw": self.maximum_traction_power_kw,
            "maximum_regen_power_kw": self.maximum_regen_power_kw,
            "g": self.g,
        }


def _require(data: dict[str, Any], key: str) -> float:
    if key not in data:
        raise KeyError(f"Missing vehicle parameter '{key}' in YAML.")
    return float(data[key])


def load_vehicle_parameters(path: str | Path | None = None, config: dict[str, Any] | None = None) -> VehicleParameters:
    if path is None:
        rel = (config or {}).get("vehicle", {}).get("config", "configs/vehicle_twizy.yaml")
        path = resolve_under_root(rel, project_root())
    else:
        path = resolve_under_root(path, project_root())
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return VehicleParameters(
        battery_capacity_kwh=_require(data, "battery_capacity_kwh"),
        mass_kg=_require(data, "mass_kg"),
        Crr=_require(data, "Crr"),
        cd=_require(data, "cd"),
        frontal_area_m2=_require(data, "frontal_area_m2"),
        rho_air=_require(data, "rho_air"),
        eta_drive=_require(data, "eta_drive"),
        eta_regen=_require(data, "eta_regen"),
        auxiliary_power_kw=_require(data, "auxiliary_power_kw"),
        maximum_traction_power_kw=_require(data, "maximum_traction_power_kw"),
        maximum_regen_power_kw=_require(data, "maximum_regen_power_kw"),
        g=float(data.get("g", STANDARD_GRAVITY)),
    )
