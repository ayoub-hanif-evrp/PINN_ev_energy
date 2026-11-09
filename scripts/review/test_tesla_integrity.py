#!/usr/bin/env python
"""Automated scientific-integrity checks for the Tesla review pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "review"))

from config import load_config  # noqa: E402
from data.schema import main_features_from_config  # noqa: E402
from physics.parameters import load_vehicle_parameters  # noqa: E402
from tesla_data import load_tesla_pool  # noqa: E402


def test_no_power_in_features(config_path: str) -> None:
    config = load_config(config_path)
    feats = main_features_from_config(config)
    banned = {"p_meas_kw", "batt_current_a", "batt_voltage_v", "soc", "e_meas_kwh"}
    assert not (set(feats) & banned), feats


def test_current_sign_discharge_positive() -> None:
    # Dataset audit convention: positive RawBattCurrent132 = discharge.
    # Spot-check: mean p_meas on moving high-speed samples should be positive.
    index = pd.read_csv(ROOT / "data" / "tesla" / "trip_index.csv")
    row = index.loc[index["vehicle"] == "BEV1"].sort_values("e_meas_kwh", ascending=False).iloc[0]
    df = pd.read_csv(ROOT / "data" / "tesla" / "trips" / f"{row.trip_id}.csv")
    moving = df["speed_kmh"] > 30
    assert float(df.loc[moving, "p_meas_kw"].mean()) > 0.0


def test_pool_no_duplicate_days_preferred() -> None:
    for vehicle in ("bev1", "bev2"):
        path = ROOT / "review_results" / "tesla" / "pools" / f"{vehicle}_pool.json"
        if not path.exists():
            continue
        ids = json.loads(path.read_text(encoding="utf-8"))["trip_ids"]
        assert len(ids) == len(set(ids))


def test_soc_and_meas_partially_independent() -> None:
    """SoC-derived and V×I energy should correlate but not be identical."""
    config = load_config("configs/review/tesla_bev1_loto.yaml")
    trips, _ = load_tesla_pool(config, ROOT)
    params = load_vehicle_parameters(config=config)
    diffs = []
    for t in trips:
        soc = t.frame["soc"].to_numpy(dtype=float)
        e_soc = (soc[0] - soc[-1]) / 100.0 * params.battery_capacity_kwh
        e_meas = float(t.stats["e_meas_kwh"])
        diffs.append(abs(e_soc - e_meas))
    mean_diff = float(np.mean(diffs))
    assert mean_diff > 0.05, f"references nearly identical mean_diff={mean_diff}"


def test_flat_grade_assumption() -> None:
    config = load_config("configs/review/tesla_bev1_loto.yaml")
    trips, _ = load_tesla_pool(config, ROOT)
    for t in trips[:3]:
        assert np.allclose(t.frame["theta_rad"].to_numpy(dtype=float), 0.0)


def main() -> int:
    tests = [
        ("features_no_power", lambda: test_no_power_in_features("configs/review/tesla_bev1_loto.yaml")),
        ("features_no_power_bev2", lambda: test_no_power_in_features("configs/review/tesla_bev2_loto.yaml")),
        ("current_sign", test_current_sign_discharge_positive),
        ("pool_unique", test_pool_no_duplicate_days_preferred),
        ("soc_meas_independence", test_soc_and_meas_partially_independent),
        ("flat_grade", test_flat_grade_assumption),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
