"""Load and cache processed HELECAR-D trips for experiments."""

from __future__ import annotations

from typing import Any

from data.discovery import discover_dataset
from data.loader import load_selected_trips
from data.preprocessing import ProcessedTrip, preprocess_trips
from data.quantization import q_from_trips
from manifest import file_sha256


def load_processed(config: dict[str, Any]) -> list[ProcessedTrip]:
    discovery = discover_dataset(config)
    if discovery.n_selected == 0:
        raise SystemExit(
            "No HELECAR-D analysed trips found. Download the dataset locally "
            "(do not commit it) and re-run."
        )
    raw = load_selected_trips(discovery.selected)
    return preprocess_trips(raw, config)


def dataset_file_hashes(trips: list[ProcessedTrip]) -> dict[str, str]:
    out: dict[str, str] = {}
    for trip in trips:
        path = trip.source_path
        if path is not None and path.exists():
            out[trip.trip_id] = file_sha256(path)
    return out


def representative_test_ids(trips: list[ProcessedTrip], n_per_traj: int = 1) -> list[str]:
    """Deterministic one-per-trajectory held-out IDs for smoke runs."""
    chosen: list[str] = []
    for traj in ("T1", "T2", "T3"):
        cands = sorted((t for t in trips if t.trajectory == traj), key=lambda t: t.trip_id)
        if not cands:
            continue
        mid = len(cands) // 2
        for k in range(min(n_per_traj, len(cands))):
            chosen.append(cands[(mid + k) % len(cands)].trip_id)
    return chosen


def empirical_q(trips: list[ProcessedTrip]) -> float:
    return q_from_trips(trips)
