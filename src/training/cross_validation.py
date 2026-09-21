"""Leave-one-trip-out fold construction. Never splits rows across trips."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from data.preprocessing import ProcessedTrip


@dataclass(frozen=True)
class LotoFold:
    test_id: str
    train_ids: tuple[str, ...]
    inner_train_ids: tuple[str, ...]
    val_id: str


def _stable_index(key: str, modulus: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % modulus


def select_validation_trip(train_ids: list[str], test_id: str, seed: int) -> str:
    ids = sorted(train_ids)
    if not ids:
        raise ValueError("No training trips from which to select a validation trip.")
    return ids[_stable_index(f"{test_id}:{seed}", len(ids))]


def leave_one_trip_out(trips: list[ProcessedTrip], seed: int = 0) -> list[LotoFold]:
    ids = [t.trip_id for t in trips]
    folds: list[LotoFold] = []
    for test_id in ids:
        train_ids = [i for i in ids if i != test_id]
        val_id = select_validation_trip(train_ids, test_id, seed)
        inner = tuple(i for i in train_ids if i != val_id)
        folds.append(
            LotoFold(
                test_id=test_id,
                train_ids=tuple(train_ids),
                inner_train_ids=inner,
                val_id=val_id,
            )
        )
    return folds


def trips_by_id(trips: list[ProcessedTrip]) -> dict[str, ProcessedTrip]:
    return {t.trip_id: t for t in trips}
