"""Dataset discovery and loading tests."""

from __future__ import annotations

from pathlib import Path

from data.discovery import discover_dataset, infer_trip_id, infer_trajectory
from data.loader import load_trip_csv
from manifest import file_sha256

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_infer_trip_id_strips_double_csv():
    assert infer_trip_id(Path("T1_06_30_2021_2.csv.csv")) == "T1_06_30_2021_2"


def test_infer_trajectory_from_path():
    assert infer_trajectory(Path("foo/Analysed/T3/T3_04_22_2021_1.csv"), "T3_04_22_2021_1") == "T3"


def test_discovery_prefers_analysed_over_raw():
    result = discover_dataset(
        config={"discovery": {"prefer_analysed": True, "skip_dir_names": []}},
        search_roots=[FIXTURES],
        min_score=8,
    )
    assert result.n_selected == 2
    by_id = {c.trip_id: c for c in result.selected}
    assert by_id["T1_01_01_2021"].kind == "analysed"
    assert "Analysed" in by_id["T1_01_01_2021"].path.as_posix() or "analysed" in by_id["T1_01_01_2021"].path.as_posix()
    assert by_id["T2_01_01_2021"].trajectory == "T2"


def test_loader_does_not_mutate_source_csv(tmp_path):
    src = FIXTURES / "analysed" / "T1" / "T1_01_01_2021.csv"
    before = file_sha256(src)
    rec = load_trip_csv(src, "T1_01_01_2021", "T1", "analysed")
    after = file_sha256(src)
    assert before == after
    assert "soc" in rec.frame.columns
    assert "Soc" in rec.frame.columns or "soc" in {c.lower() for c in rec.frame.columns}
    assert rec.n_rows == 16


def test_canonical_speed_column():
    src = FIXTURES / "analysed" / "T1" / "T1_01_01_2021.csv"
    rec = load_trip_csv(src, "T1_01_01_2021", "T1", "analysed")
    assert rec.frame["speed_kmh"].iloc[4] == 36.0
