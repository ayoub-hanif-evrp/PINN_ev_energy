from data.feature_audit import assert_feature_units_and_coverage, feature_distribution_table
from data.schema import MAIN_MODEL_FEATURES
from pathlib import Path

from config import load_config
from data.loader import load_trip_csv
from data.preprocessing import preprocess_trip
from paths import project_root

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_feature_units_and_progress_columns():
    config = load_config(project_root() / "configs" / "base.yaml")
    trip = preprocess_trip(
        load_trip_csv(FIXTURES / "analysed" / "T1" / "T1_01_01_2021.csv", "T1_01_01_2021", "T1", "analysed"),
        config,
    )
    table = feature_distribution_table([trip])
    assert set(MAIN_MODEL_FEATURES).issubset(set(table["feature"]))
    assert "elapsed_s" in set(table["feature"])
    wind = table.loc[table["feature"] == "wind_speed_mps"].iloc[0]
    assert wind["max"] < 80
    assert_feature_units_and_coverage(table)
