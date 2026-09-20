"""Canonical HELECAR-D column names and aliases."""

from __future__ import annotations

from dataclasses import dataclass

# Canonical internal names -> accepted header aliases (case-insensitive).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date",),
    "time": ("time",),
    "soc": ("soc", "soC"),
    "speed_kmh": ("speed",),
    "mode": ("mode",),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "lng", "longitude"),
    "alt_m": ("alt", "altitude"),
    "gps_speed_kmh": ("gpsspeed", "gps_speed"),
    "temperature_c": ("temperature", "temp"),
    "humidity_pct": ("humidity",),
    "weather": ("weather",),
    "wind_speed_mps": ("windspeedav", "windspeed", "wind_speed"),
    "traffic": ("traffic",),
    "speed_limit_kmh": ("speedlimit", "speed_limit"),
}

EXPECTED_ANALYSED_CANONICAL = (
    "date",
    "time",
    "soc",
    "speed_kmh",
    "mode",
    "lat",
    "lon",
    "alt_m",
    "gps_speed_kmh",
    "temperature_c",
    "humidity_pct",
    "weather",
    "wind_speed_mps",
    "traffic",
    "speed_limit_kmh",
)

RAW_INDICATOR_NORMALIZED = {
    "exactspeed",
    "odometer",
    "winddirection",
    "pressure",
    "id",
    "data",
}

MAIN_MODEL_FEATURES = (
    "speed_mps",
    "acc_mps2",
    "theta_rad",
    "temperature_c",
    "humidity_pct",
    "wind_speed_mps",
    "traffic",
    "speed_limit_mps",
)

EXCLUDED_MAIN_PREDICTORS = ("date", "time", "lat", "lon", "soc")


def normalize_header(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum())


def alias_lookup() -> dict[str, str]:
    table: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        table[normalize_header(canonical)] = canonical
        for alias in aliases:
            table[normalize_header(alias)] = canonical
    return table


ALIAS_LOOKUP = alias_lookup()


@dataclass(frozen=True)
class ColumnMap:
    """Mapping from canonical name to original CSV header."""

    original_to_canonical: dict[str, str]
    canonical_to_original: dict[str, str]

    def original(self, canonical: str) -> str | None:
        return self.canonical_to_original.get(canonical)
