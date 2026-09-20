"""HELECAR-D CSV loader. Never writes back to the original files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.discovery import CandidateFile
from data.schema import ALIAS_LOOKUP, ColumnMap, normalize_header


@dataclass
class TripRecord:
    trip_id: str
    trajectory: str
    source_path: Path
    kind: str
    frame: pd.DataFrame
    column_map: ColumnMap
    load_notes: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))


def _build_column_map(columns: list[str]) -> ColumnMap:
    original_to_canonical: dict[str, str] = {}
    canonical_to_original: dict[str, str] = {}
    for original in columns:
        key = normalize_header(original)
        if key in ALIAS_LOOKUP:
            canonical = ALIAS_LOOKUP[key]
            original_to_canonical[original] = canonical
            # First alias wins; do not overwrite if two originals map to one canonical.
            canonical_to_original.setdefault(canonical, original)
    return ColumnMap(original_to_canonical, canonical_to_original)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    if series.dtype == object:
        lowered = series.astype(str).str.strip().str.lower()
        mapped = lowered.replace(
            {
                "low": 0,
                "medium": 1,
                "med": 1,
                "high": 2,
                "nan": np.nan,
                "none": np.nan,
                "": np.nan,
            }
        )
        return pd.to_numeric(mapped, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def load_trip_csv(path: Path, trip_id: str, trajectory: str, kind: str) -> TripRecord:
    """Read a trip CSV. Original files are opened read-only and never rewritten."""
    frame = pd.read_csv(path)
    column_map = _build_column_map([str(c) for c in frame.columns])
    notes: list[str] = []

    # Add canonical copies without deleting originals.
    for original, canonical in column_map.original_to_canonical.items():
        if canonical in frame.columns and canonical != original:
            continue
        source = frame[original]
        if canonical in {
            "soc",
            "speed_kmh",
            "gps_speed_kmh",
            "lat",
            "lon",
            "alt_m",
            "temperature_c",
            "humidity_pct",
            "wind_speed_mps",
            "traffic",
            "speed_limit_kmh",
        }:
            frame[canonical] = _coerce_numeric(source)
        else:
            frame[canonical] = source

    missing = [
        name
        for name in ("soc", "speed_kmh", "lat", "lon", "alt_m")
        if name not in frame.columns
    ]
    if missing:
        notes.append(f"missing_canonical_columns={missing}")

    return TripRecord(
        trip_id=trip_id,
        trajectory=trajectory,
        source_path=path,
        kind=kind,
        frame=frame,
        column_map=column_map,
        load_notes=notes,
    )


def load_candidate(candidate: CandidateFile) -> TripRecord:
    return load_trip_csv(candidate.path, candidate.trip_id, candidate.trajectory, candidate.kind)


def load_selected_trips(candidates: list[CandidateFile]) -> list[TripRecord]:
    return [load_candidate(c) for c in candidates]
