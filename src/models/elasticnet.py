"""Trip-level ElasticNet with an explicit predictor whitelist (no SoC leakage)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.features import assert_no_soc_in_predictors, elasticnet_feature_matrix, elasticnet_target
from data.schema import ELASTICNET_FEATURES


class TripElasticNet:
    """Predicts observed trip energy (kWh) from non-SoC trip summaries.

    Standardization and hyperparameter selection use the training table only.
    Outer-test rows must not be present in ``fit``.
    """

    def __init__(
        self,
        alphas: tuple[float, ...] | list[float] = (0.001, 0.01, 0.1, 1.0),
        l1_ratios: tuple[float, ...] | list[float] = (0.25, 0.5, 0.75, 1.0),
        random_state: int = 0,
    ):
        self.alphas = list(alphas)
        self.l1_ratios = list(l1_ratios)
        self.random_state = int(random_state)
        self.model_: Pipeline | None = None
        self.best_params_: dict[str, Any] | None = None
        self.fitted_trip_ids: list[str] = []

    def fit(self, table: pd.DataFrame) -> "TripElasticNet":
        x = elasticnet_feature_matrix(table)
        assert_no_soc_in_predictors(x.columns)
        y = elasticnet_target(table, "e_obs_kwh")
        if "trip_id" in table.columns:
            self.fitted_trip_ids = [str(v) for v in table["trip_id"].tolist()]
        n = len(table)
        if n < 2:
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            )
            pipe.fit(x, y)
            self.model_ = pipe
            self.best_params_ = {"family": "ridge", "alpha": 1.0}
            return self
        cv = LeaveOneOut()
        enet = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(max_iter=20000, random_state=self.random_state)),
            ]
        )
        gs_enet = GridSearchCV(
            enet,
            {"model__alpha": self.alphas, "model__l1_ratio": self.l1_ratios},
            cv=cv,
            scoring="neg_mean_absolute_error",
            refit=True,
        )
        gs_enet.fit(x, y)
        ridge = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        )
        gs_ridge = GridSearchCV(
            ridge,
            {"model__alpha": self.alphas},
            cv=cv,
            scoring="neg_mean_absolute_error",
            refit=True,
        )
        gs_ridge.fit(x, y)
        if gs_ridge.best_score_ > gs_enet.best_score_:
            self.model_ = gs_ridge.best_estimator_
            self.best_params_ = {"family": "ridge", **gs_ridge.best_params_}
        else:
            self.model_ = gs_enet.best_estimator_
            self.best_params_ = {"family": "elasticnet", **gs_enet.best_params_}
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("TripElasticNet has not been fit.")
        x = elasticnet_feature_matrix(table)
        assert_no_soc_in_predictors(x.columns)
        leaked = [c for c in x.columns if c in {"soc_start", "soc_end", "soc_delta", "e_obs_kwh"} or str(c) not in ELASTICNET_FEATURES]
        if leaked:
            raise AssertionError(f"ElasticNet inference leakage: {leaked}")
        return np.asarray(self.model_.predict(x), dtype=float)

    def assert_not_fitted_on(self, trip_id: str) -> None:
        if trip_id in self.fitted_trip_ids:
            raise AssertionError(f"Leakage: ElasticNet was fit including test trip {trip_id}")
