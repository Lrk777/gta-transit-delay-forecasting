"""Unit tests for feature engineering, splits, and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import (
    add_calendar_features,
    add_lag_rolling_features,
    build_feature_matrix,
    feature_columns,
    join_sources,
    time_based_split,
)
from src.ingest_transit import _normalize_incident_frame, aggregate_daily
from src.metrics import pick_best_by_val, regression_metrics


def test_normalize_and_aggregate_incidents():
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "Route": ["1", "2", "1"],
            "Min Delay": [10, 20, 5],
            "Incident": ["Mechanical", "Diversion", "Security"],
        }
    )
    norm = _normalize_incident_frame(raw)
    daily = aggregate_daily(norm)
    assert list(daily.columns) == [
        "date",
        "total_delay_min",
        "n_incidents",
        "mean_delay_min",
        "median_delay_min",
        "p90_delay_min",
    ]
    assert daily.loc[daily["date"] == "2024-01-01", "total_delay_min"].iloc[0] == 30
    assert daily.loc[daily["date"] == "2024-01-01", "n_incidents"].iloc[0] == 2


def test_join_requires_overlapping_dates():
    transit = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "total_delay_min": [100, 200],
        }
    )
    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "precipitation_sum": [0.0, 1.0],
        }
    )
    joined = join_sources(transit, weather)
    assert len(joined) == 1
    assert joined.iloc[0]["total_delay_min"] == 200


def test_lag_features_do_not_use_same_day_target():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "total_delay_min": np.arange(10, dtype=float) * 100,
        }
    )
    out = add_lag_rolling_features(df, "total_delay_min", lags=[1], windows=[3])
    # lag_1 on day i equals target on day i-1
    assert out.loc[5, "total_delay_min_lag_1"] == out.loc[4, "total_delay_min"]
    # rolling mean uses shift(1), so day 5 roll window is days 2,3,4
    expected = out.loc[2:4, "total_delay_min"].mean()
    assert out.loc[5, "total_delay_min_roll3_mean"] == pytest.approx(expected)


def test_time_based_split_rejects_shuffle_leakage():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "total_delay_min": np.arange(30),
        }
    )
    train, val, test = time_based_split(df, train_end="2024-01-15", val_end="2024-01-22")
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
    with pytest.raises(ValueError):
        time_based_split(df, train_end="2024-01-22", val_end="2024-01-15")


def test_calendar_features():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-06", "2024-01-07"])})  # Sat, Sun
    out = add_calendar_features(df, ["day_of_week", "is_weekend", "month"])
    assert out.loc[0, "is_weekend"] == 1
    assert out.loc[0, "day_of_week"] == 5
    assert out.loc[0, "month"] == 1


def test_regression_metrics_and_selection():
    y = np.array([100.0, 200.0, 300.0])
    pred = np.array([110.0, 190.0, 310.0])
    m = regression_metrics(y, pred)
    assert m["mae"] == pytest.approx(10.0)
    assert m["n"] == 3
    best = pick_best_by_val(
        {"a": {"mae": 20.0}, "b": {"mae": 10.0}, "c": {"mae": 15.0}},
        key="mae",
    )
    assert best == "b"


def test_feature_matrix_from_fixture_config(tmp_path, monkeypatch):
    cfg = load_config("configs/fixture.yaml")
    # Redirect processed outputs into tmp
    for key in [
        "transit_daily_csv",
        "weather_daily_csv",
        "feature_matrix_csv",
        "metrics_json",
        "error_analysis_json",
        "predictions_csv",
    ]:
        cfg["paths"][key] = str(tmp_path / cfg["paths"][key])
    cfg["paths"]["raw_dir"] = str(tmp_path / "raw")
    cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    cfg["paths"]["reports_dir"] = str(tmp_path / "reports")
    cfg["paths"]["figures_dir"] = str(tmp_path / "reports" / "figures")

    from src.ingest_transit import ingest_transit
    from src.ingest_weather import ingest_weather

    transit = ingest_transit(cfg, use_fixture=True)
    weather = ingest_weather(cfg, use_fixture=True)
    joined = join_sources(transit, weather)
    matrix = build_feature_matrix(cfg, joined)
    cols = feature_columns(cfg, matrix)
    assert "total_delay_min_lag_1" in cols
    assert "precipitation_sum_lag_1" in cols
    assert "precipitation_sum" not in cols  # same-day weather excluded from model features
    assert matrix["date"].is_monotonic_increasing
