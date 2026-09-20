"""Window generation tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from config import load_config
from data.loader import load_trip_csv
from data.preprocessing import preprocess_trip
from data.quantization import estimate_soc_quantization
from data.windows import generate_trip_windows, generate_windows
from paths import project_root

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _processed_pair(config):
    t1 = load_trip_csv(FIXTURES / "analysed" / "T1" / "T1_01_01_2021.csv", "T1_01_01_2021", "T1", "analysed")
    t2 = load_trip_csv(FIXTURES / "analysed" / "T2" / "T2_01_01_2021.csv", "T2_01_01_2021", "T2", "analysed")
    return [preprocess_trip(t1, config), preprocess_trip(t2, config)]


def test_windows_never_cross_trip_boundaries():
    config = load_config(project_root() / "configs" / "base.yaml")
    trips = _processed_pair(config)
    q = estimate_soc_quantization(trips[0].frame["soc"])["q"]
    windows = generate_windows(trips, 6.0, q, config=config, overlapping=True)
    lengths = {t.trip_id: t.n_rows for t in trips}
    assert windows
    for w in windows:
        assert w.start >= 0
        assert w.end < lengths[w.trip_id]
        assert w.end >= w.start
        assert w.trip_id in lengths


def test_default_soc_events_reject_few_second_windows():
    config = load_config(project_root() / "configs" / "base.yaml")
    trips = _processed_pair(config)
    windows = generate_trip_windows(trips[0], 6.0, 0.02, config=config, overlapping=True)
    events = [w for w in windows if w.scale.startswith("soc_event")]
    # Fixture trips are ~15 s; 60 s minimum duration must suppress 1q/2q-style events.
    assert events == []
    assert all(w.duration_s >= 60.0 for w in events)


def test_soc_event_windows_use_absolute_percent_thresholds():
    config = deepcopy(load_config(project_root() / "configs" / "base.yaml"))
    config["windows"]["soc_event_min_duration_s"] = 0.0
    config["windows"]["soc_event_dsoc_pct"] = [0.1, 0.2, 0.5]
    trips = _processed_pair(config)
    windows = generate_trip_windows(trips[0], 6.0, 0.02, config=config, overlapping=True)
    events = [w for w in windows if w.scale.startswith("soc_event")]
    assert events
    scales = {w.scale for w in events}
    assert "soc_event_0.1" in scales
    for w in events:
        thresh = float(w.scale.replace("soc_event_", ""))
        assert abs(w.dsoc) + 1e-9 >= thresh
        # Must not be a 1q/2q/3q label (q≈0.02).
        assert not w.scale.startswith("soc_event_k")


def test_full_trip_window_present():
    config = load_config(project_root() / "configs" / "base.yaml")
    trips = _processed_pair(config)
    windows = generate_trip_windows(trips[0], 6.0, 0.02, config=config)
    full = [w for w in windows if w.scale == "full_trip"]
    assert len(full) == 1
    assert full[0].start == 0
    assert full[0].end == trips[0].n_rows - 1
