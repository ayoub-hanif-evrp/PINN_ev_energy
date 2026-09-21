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
    val_ids: tuple[str, ...]

    @property
    def val_id(self) -> str:
        return self.val_ids[0] if self.val_ids else ""


def _stable_index(key: str, modulus: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % max(int(modulus), 1)


def select_validation_trip(train_ids: list[str], test_id: str, seed: int) -> str:
    ids = sorted(train_ids)
    if not ids:
        raise ValueError("No training trips from which to select a validation trip.")
    return ids[_stable_index(f"{test_id}:{seed}", len(ids))]


def select_validation_ids(
    trips: list[ProcessedTrip],
    train_ids: list[str] | tuple[str, ...],
    test_id: str,
    seed: int,
) -> tuple[str, ...]:
    """One validation trip per available trajectory (T1/T2/T3), deterministic."""
    by_id = {t.trip_id: t for t in trips}
    train = [i for i in train_ids if i != test_id]
    chosen: list[str] = []
    for traj in ("T1", "T2", "T3"):
        cands = sorted(i for i in train if by_id.get(i) is not None and by_id[i].trajectory == traj)
        if not cands:
            continue
        pick = cands[_stable_index(f"{test_id}:{seed}:{traj}", len(cands))]
        chosen.append(pick)
    if not chosen:
        if not train:
            raise ValueError("No training trips remain for validation.")
        chosen = [select_validation_trip(train, test_id, seed)]
    return tuple(chosen)


def leave_one_trip_out(trips: list[ProcessedTrip], seed: int = 0) -> list[LotoFold]:
    ids = [t.trip_id for t in trips]
    folds: list[LotoFold] = []
    for test_id in ids:
        train_ids = [i for i in ids if i != test_id]
        val_ids = select_validation_ids(trips, train_ids, test_id, seed)
        inner = tuple(i for i in train_ids if i not in set(val_ids))
        folds.append(
            LotoFold(
                test_id=test_id,
                train_ids=tuple(train_ids),
                inner_train_ids=inner,
                val_ids=val_ids,
            )
        )
        if test_id in folds[-1].train_ids:
            raise AssertionError("test_id leaked into train_ids")
        if test_id in folds[-1].inner_train_ids or test_id in folds[-1].val_ids:
            raise AssertionError("test_id leaked into inner train or validation")
        if set(folds[-1].val_ids) & set(folds[-1].inner_train_ids):
            raise AssertionError("validation IDs overlap inner training IDs")
    return folds


def trips_by_id(trips: list[ProcessedTrip]) -> dict[str, ProcessedTrip]:
    return {t.trip_id: t for t in trips}
