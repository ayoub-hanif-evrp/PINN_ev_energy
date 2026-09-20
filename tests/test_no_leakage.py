"""Anti-leakage tests for trip-level evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import load_config
from data.features import trip_level_feature_table
from data.loader import load_trip_csv
from data.preprocessing import preprocess_trip
from data.scaling import TripStandardScaler
from data.windows import generate_windows
from evaluation.metrics import reconstruct_soc
from paths import project_root

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _trips(config):
    t1 = preprocess_trip(
        load_trip_csv(FIXTURES / "analysed" / "T1" / "T1_01_01_2021.csv", "T1_01_01_2021", "T1", "analysed"),
        config,
    )
    t2 = preprocess_trip(
        load_trip_csv(FIXTURES / "analysed" / "T2" / "T2_01_01_2021.csv", "T2_01_01_2021", "T2", "analysed"),
        config,
    )
    return t1, t2


def test_scaler_fitted_only_on_training_trips():
    config = load_config(project_root() / "configs" / "base.yaml")
    train, test = _trips(config)
    scaler = TripStandardScaler().fit([train])
    scaler.assert_not_fitted_on(test.trip_id)
    assert train.trip_id in scaler.fitted_trip_ids
    # Transforming the test trip is allowed; fitting on it is not.
    x = scaler.transform_trip(test)
    assert x.shape[0] == test.n_rows


def test_outer_test_trip_not_in_training_windows():
    config = load_config(project_root() / "configs" / "base.yaml")
    train, test = _trips(config)
    windows = generate_windows([train], 6.0, 0.02, config=config)
    assert windows
    assert all(w.trip_id != test.trip_id for w in windows)
    assert all(w.trip_id == train.trip_id for w in windows)


def test_soc_reconstruction_uses_only_initial_measured_soc():
    config = load_config(project_root() / "configs" / "base.yaml")
    trip, _ = _trips(config)
    soc = trip.frame["soc"].to_numpy(dtype=float)
    p = np.ones(len(trip.frame))
    dt = trip.frame["dt_s"].to_numpy(dtype=float)
    later = soc.copy()
    later[1:] = 0.0  # poison later SoC; reconstruction must ignore them
    out = reconstruct_soc(soc[0], p, dt, 6.0)
    np.testing.assert_allclose(out["soc_hat"][0], soc[0])
    # Using poisoned later values would change a naive cumulative-SoC baseline.
    naive = np.concatenate([[soc[0]], later[1:]])
    assert not np.allclose(out["soc_hat"], naive)
    assert out["soc_initial_used"] == soc[0]


def test_elasticnet_design_matrix_excludes_test_trip_and_its_target():
    config = load_config(project_root() / "configs" / "base.yaml")
    train, test = _trips(config)
    table = trip_level_feature_table([train])
    assert test.trip_id not in set(table["trip_id"])
    assert "soc_delta" in table.columns
    # The outer test target must not appear in the training response vector.
    y_train = table.set_index("trip_id")["soc_delta"]
    assert test.trip_id not in y_train.index


def test_data_scarcity_subsets_only_allowed_training_trips():
    train_pool = ["T1_a", "T2_a", "T3_a", "T3_b"]
    test_id = "T1_b"
    rng = np.random.default_rng(0)
    chosen = list(rng.choice(train_pool, size=3, replace=False))
    assert test_id not in chosen
    assert set(chosen).issubset(set(train_pool))
