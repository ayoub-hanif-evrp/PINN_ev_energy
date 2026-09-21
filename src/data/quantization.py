"""Empirical SoC quantization diagnostics. Do not assume q = 1%."""

from __future__ import annotations

from math import gcd
from typing import Any

import numpy as np
import pandas as pd


def _gcd_many(values: list[int]) -> int:
    acc = 0
    for v in values:
        acc = gcd(acc, int(abs(v)))
    return acc


def estimate_soc_quantization(soc: np.ndarray | pd.Series, round_decimals: int = 4) -> dict[str, Any]:
    """
    Estimate the effective SoC step q from observed values.

    Returns both:
    - q_mode: mode of nonzero successive |ΔSoC|
    - q_gcd: GCD of unique-level spacings (scaled to 10^{-round_decimals})

    This is an empirical resolution estimate, not a BMS model.
    """
    x = np.asarray(soc, dtype=float)
    x = x[np.isfinite(x)]
    empty = {
        "q": np.nan,
        "q_mode": np.nan,
        "q_gcd": np.nan,
        "n_unique": 0,
        "n_transitions": 0,
        "n_increases": 0,
        "n_decreases": 0,
        "n_zero_steps": 0,
        "min": np.nan,
        "max": np.nan,
        "start": np.nan,
        "end": np.nan,
        "unique_increments": [],
        "transition_values": [],
        "transition_counts": [],
    }
    if x.size == 0:
        return empty

    unique = np.unique(np.round(x, round_decimals))
    diffs_unique = np.diff(unique)
    diffs_unique = diffs_unique[diffs_unique > 0]

    steps = np.diff(x)
    n_zero = int(np.sum(np.abs(steps) < 10 ** (-round_decimals)))
    nonzero = steps[np.abs(steps) >= 10 ** (-round_decimals)]
    n_inc = int(np.sum(nonzero > 0))
    n_dec = int(np.sum(nonzero < 0))

    abs_steps = np.round(np.abs(nonzero), round_decimals)
    if abs_steps.size:
        values, counts = np.unique(abs_steps, return_counts=True)
        q_mode = float(values[int(np.argmax(counts))])
        trans_values = values.tolist()
        trans_counts = counts.tolist()
    else:
        q_mode = np.nan
        trans_values, trans_counts = [], []

    scale = 10**round_decimals
    ints = [int(round(v * scale)) for v in diffs_unique if v > 0]
    ints = [v for v in ints if v > 0]
    q_gcd = float(_gcd_many(ints)) / scale if ints else np.nan

    q = q_mode if np.isfinite(q_mode) else q_gcd
    unique_increments = sorted({float(v) for v in np.round(nonzero, round_decimals)})

    return {
        "q": float(q) if np.isfinite(q) else np.nan,
        "q_mode": float(q_mode) if np.isfinite(q_mode) else np.nan,
        "q_gcd": float(q_gcd) if np.isfinite(q_gcd) else np.nan,
        "n_unique": int(unique.size),
        "n_transitions": int(nonzero.size),
        "n_increases": n_inc,
        "n_decreases": n_dec,
        "n_zero_steps": n_zero,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "start": float(x[0]),
        "end": float(x[-1]),
        "unique_increments": unique_increments[:50],
        "transition_values": trans_values,
        "transition_counts": trans_counts,
    }


def combine_quantization(per_trip: list[dict[str, Any]]) -> dict[str, Any]:
    """Dataset-level q from pooled successive |ΔSoC| modes, plus median of per-trip q."""
    qs = [d["q"] for d in per_trip if np.isfinite(d.get("q", np.nan))]
    modes = [d["q_mode"] for d in per_trip if np.isfinite(d.get("q_mode", np.nan))]
    gcds = [d["q_gcd"] for d in per_trip if np.isfinite(d.get("q_gcd", np.nan))]
    pooled_values: list[float] = []
    pooled_counts: list[int] = []
    from collections import Counter

    counter: Counter[float] = Counter()
    for d in per_trip:
        for v, c in zip(d.get("transition_values", []), d.get("transition_counts", [])):
            counter[float(v)] += int(c)
    if counter:
        q_pooled_mode = counter.most_common(1)[0][0]
        pooled_values = [v for v, _ in counter.most_common()]
        pooled_counts = [c for _, c in counter.most_common()]
    else:
        q_pooled_mode = np.nan
    q = float(q_pooled_mode) if np.isfinite(q_pooled_mode) else (float(np.median(qs)) if qs else np.nan)
    return {
        "q": q,
        "q_pooled_mode": float(q_pooled_mode) if np.isfinite(q_pooled_mode) else np.nan,
        "q_median_trip": float(np.median(qs)) if qs else np.nan,
        "q_mode_median": float(np.median(modes)) if modes else np.nan,
        "q_gcd_median": float(np.median(gcds)) if gcds else np.nan,
        "n_trips": len(per_trip),
        "transition_values": pooled_values,
        "transition_counts": pooled_counts,
        "note": (
            "Empirical SoC step from observed transitions. "
            "Not a claim about the unknown BMS quantization process."
        ),
    }


def q_from_trips(trips: list[Any]) -> float:
    """Estimate q from the given trips only (never mix in held-out test SoC)."""
    per = []
    for trip in trips:
        if "soc" not in getattr(trip, "frame", {}):
            continue
        per.append(estimate_soc_quantization(trip.frame["soc"]))
    combined = combine_quantization(per)
    q = float(combined["q"]) if np.isfinite(combined.get("q", np.nan)) else 0.02
    if not np.isfinite(q) or q <= 0:
        return 0.02
    return q
