"""Deterministic training-subset sampling for data-scarcity experiments."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from data.preprocessing import ProcessedTrip


def _rng_seed(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def stratified_subset(
    trips: list[ProcessedTrip],
    n: int,
    *,
    test_id: str,
    repeat: int,
    seed: int,
) -> list[str]:
    """Sample n training IDs from trips, never including test_id.

    For n>=3, try to include one trip from each of T1/T2/T3 when available.
    """
    pool = [t for t in trips if t.trip_id != test_id]
    if n >= len(pool):
        return sorted(t.trip_id for t in pool)
    by_traj: dict[str, list[ProcessedTrip]] = defaultdict(list)
    for t in pool:
        by_traj[t.trajectory].append(t)
    for traj in by_traj:
        by_traj[traj] = sorted(by_traj[traj], key=lambda t: t.trip_id)

    import numpy as np

    rng = np.random.default_rng(_rng_seed(f"{test_id}:{n}:{repeat}:{seed}"))
    chosen: list[str] = []
    if n >= 3:
        for traj in ("T1", "T2", "T3"):
            cands = by_traj.get(traj, [])
            if not cands:
                continue
            pick = cands[int(rng.integers(0, len(cands)))]
            chosen.append(pick.trip_id)
            if len(chosen) >= n:
                break
    remaining = [t.trip_id for t in pool if t.trip_id not in set(chosen)]
    need = n - len(chosen)
    if need > 0:
        extra = list(rng.choice(remaining, size=min(need, len(remaining)), replace=False))
        chosen.extend(str(x) for x in extra)
    assert test_id not in chosen
    return sorted(chosen)
