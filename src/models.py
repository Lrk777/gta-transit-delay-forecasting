"""Baselines and stronger regressors for daily delay forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Predictor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class LagBaseline:
    """Predict using a single lag column already present in X (by index)."""

    lag_feature_index: int
    name: str = "naive_lag"

    def fit(self, X: np.ndarray, y: np.ndarray) -> LagBaseline:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X[:, self.lag_feature_index], dtype=float)


def build_models(cfg: dict[str, Any], feature_names: list[str]) -> dict[str, Predictor]:
    target = cfg["target"]["column"]
    try:
        lag1_idx = feature_names.index(f"{target}_lag_1")
    except ValueError as e:
        raise KeyError("lag_1 feature required for naive baseline") from e
    try:
        lag7_idx = feature_names.index(f"{target}_lag_7")
    except ValueError as e:
        raise KeyError("lag_7 feature required for seasonal baseline") from e

    hgb_params = cfg["models"]["hist_gradient_boosting"]
    ridge_alpha = float(cfg["models"]["ridge"]["alpha"])

    hgb_kwargs = {
        "max_depth": hgb_params["max_depth"],
        "learning_rate": hgb_params["learning_rate"],
        "max_iter": hgb_params["max_iter"],
        "min_samples_leaf": hgb_params["min_samples_leaf"],
        "l2_regularization": hgb_params["l2_regularization"],
        "random_state": cfg["seed"],
        "early_stopping": bool(hgb_params.get("early_stopping", False)),
        "n_iter_no_change": int(hgb_params.get("n_iter_no_change", 10)),
    }
    # When train.py supplies X_val, disable internal holdout fraction
    if hgb_params.get("validation_fraction", 0.1) is None:
        hgb_kwargs["validation_fraction"] = 0.1  # required by sklearn ctor; overridden via fit kwargs path

    models: dict[str, Predictor] = {
        "naive_lag1": LagBaseline(lag_feature_index=lag1_idx, name="naive_lag1"),
        "seasonal_lag7": LagBaseline(lag_feature_index=lag7_idx, name="seasonal_lag7"),
        "ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=ridge_alpha)),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(**hgb_kwargs),
    }
    return models
