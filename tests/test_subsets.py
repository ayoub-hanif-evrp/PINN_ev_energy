"""Subset sampling never includes the held-out trip."""

from dataclasses import replace
from pathlib import Path

from config import load_config
from data.loader import load_trip_csv
from data.preprocessing import preprocess_trip
from experiment.subsets import stratified_subset
from paths import project_root

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_stratified_subset_excludes_test_and_covers_trajectories():
    config = load_config(project_root() / "configs" / "base.yaml")
    t1 = preprocess_trip(load_trip_csv(FIXTURES / "analysed" / "T1" / "T1_01_01_2021.csv", "T1_a", "T1", "analysed"), config)
    t2 = preprocess_trip(load_trip_csv(FIXTURES / "analysed" / "T2" / "T2_01_01_2021.csv", "T2_a", "T2", "analysed"), config)
    t3 = replace(t1, trip_id="T3_a", trajectory="T3")
    extra = replace(t1, trip_id="T3_b", trajectory="T3")
    test = replace(t1, trip_id="T1_b", trajectory="T1")
    pool = [t1, t2, t3, extra]
    chosen = stratified_subset(pool, 3, test_id=test.trip_id, repeat=0, seed=0)
    assert test.trip_id not in chosen
    assert len(chosen) == 3
    traj = {t.trip_id: t.trajectory for t in pool}
    assert {"T1", "T2", "T3"}.issubset({traj[i] for i in chosen})
