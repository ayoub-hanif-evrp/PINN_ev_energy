"""Trip-level Ridge regression. SoC-free features. Training trips only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.features import assert_no_soc_in_predictors, ridge_feature_matrix, ridge_target
from data.schema import RIDGE_FEATURES

# Frozen before any Ridge test evaluation. Do not retune from held-out results.
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


class TripRidge:
    """Predict observed trip energy (kWh) from non-SoC trip summaries.

    StandardScaler sits inside the pipeline, so each inner fold scales on its
    own training rows. Alpha is chosen by leave-one-trip-out on the rows passed
    to ``fit`` only. The outer held-out trip must not be in that table.
    """

    def __init__(
        self,
        alphas: tuple[float, ...] | list[float] = RIDGE_ALPHAS,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ):
        self.alphas = [float(a) for a in alphas]
        self.feature_names = tuple(feature_names) if feature_names is not None else RIDGE_FEATURES
        self.model_: Pipeline | None = None
        self.best_params_: dict[str, Any] | None = None
        self.fitted_trip_ids: list[str] = []

    def fit(self, table: pd.DataFrame) -> "TripRidge":
        x = ridge_feature_matrix(table, feature_names=self.feature_names)
        assert_no_soc_in_predictors(list(x.columns))
        y = ridge_target(table, "e_obs_kwh")
        if "trip_id" in table.columns:
            self.fitted_trip_ids = [str(v) for v in table["trip_id"].tolist()]
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        )
        if len(table) < 2:
            pipe.set_params(model__alpha=1.0)
            pipe.fit(x, y)
            self.model_ = pipe
            self.best_params_ = {"alpha": 1.0, "selection": "fixed_because_n_lt_2"}
            return self
        search = GridSearchCV(
            pipe,
            {"model__alpha": self.alphas},
            cv=LeaveOneOut(),
            scoring="neg_mean_absolute_error",
            refit=True,
        )
        search.fit(x, y)
        self.model_ = search.best_estimator_
        self.best_params_ = {
            "alpha": float(search.best_params_["model__alpha"]),
            "selection": "leave_one_trip_out_on_fit_rows_only",
        }
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("TripRidge has not been fit.")
        x = ridge_feature_matrix(table, feature_names=self.feature_names)
        assert_no_soc_in_predictors(list(x.columns))
        allowed = set(self.feature_names)
        leaked = [
            c
            for c in x.columns
            if c in {"soc_start", "soc_end", "soc_delta", "e_obs_kwh"} or str(c) not in allowed
        ]
        if leaked:
            raise AssertionError(f"Ridge inference leakage: {leaked}")
        return np.asarray(self.model_.predict(x), dtype=float)

    def assert_not_fitted_on(self, trip_id: str) -> None:
        if trip_id in self.fitted_trip_ids:
            raise AssertionError(f"Leakage: Ridge was fit including test trip {trip_id}")
