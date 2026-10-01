"""Regression metrics and error-analysis helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    # Symmetric-ish MAPE guard for near-zero targets
    denom = np.clip(np.abs(y_true), 1.0, None)
    mape = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)
    bias = float(np.mean(y_pred - y_true))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_pct": mape,
        "bias": bias,
        "n": int(len(y_true)),
    }


def pick_best_by_val(val_metrics: dict[str, dict[str, float]], key: str = "mae") -> str:
    if not val_metrics:
        raise ValueError("No validation metrics to select from")
    return min(val_metrics.keys(), key=lambda m: val_metrics[m][key])


def residual_frame(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extras: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).values,
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    out["residual"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["residual"].abs()
    out["day_of_week"] = out["date"].dt.day_name()
    out["month"] = out["date"].dt.month
    if extras is not None:
        for col in extras.columns:
            out[col] = extras[col].values
    return out


def slice_error_summary(resid: pd.DataFrame) -> dict[str, Any]:
    by_dow = (
        resid.groupby("day_of_week", sort=False)["abs_error"]
        .agg(["mean", "median", "count"])
        .round(2)
        .to_dict(orient="index")
    )
    # Precip buckets if available
    precip_summary: dict[str, Any] = {}
    if "precipitation_sum_lag_1" in resid.columns:
        bins = [-0.01, 0.1, 2.0, 10.0, 1e9]
        labels = ["dry", "light", "moderate", "heavy"]
        resid = resid.copy()
        resid["precip_bucket"] = pd.cut(
            resid["precipitation_sum_lag_1"], bins=bins, labels=labels
        )
        precip_summary = (
            resid.groupby("precip_bucket", observed=True)["abs_error"]
            .agg(["mean", "median", "count"])
            .round(2)
            .to_dict(orient="index")
        )
        precip_summary = {str(k): v for k, v in precip_summary.items()}

    worst = (
        resid.nlargest(10, "abs_error")[["date", "y_true", "y_pred", "abs_error", "day_of_week"]]
        .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
        .round(2)
        .to_dict(orient="records")
    )
    return {
        "mae_by_day_of_week": by_dow,
        "mae_by_precip_bucket_lag1": precip_summary,
        "worst_abs_error_days": worst,
    }
