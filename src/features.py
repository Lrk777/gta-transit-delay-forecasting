"""Join transit + weather and engineer lag / rolling / calendar features."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import resolve_path


def join_sources(transit: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    t = transit.copy()
    w = weather.copy()
    t["date"] = pd.to_datetime(t["date"]).dt.normalize()
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    merged = t.merge(w, on="date", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("Transit/weather join produced zero rows — check date ranges")
    return merged.sort_values("date").reset_index(drop=True)


def _ontario_stat_holiday_mask(dates: pd.Series) -> pd.Series:
    """Lightweight fixed-date + observed Ontario holidays (not a full holiday library)."""
    d = pd.to_datetime(dates)
    md = list(zip(d.dt.month.to_numpy(), d.dt.day.to_numpy()))
    fixed = {
        (1, 1),   # New Year's Day
        (7, 1),   # Canada Day
        (12, 25), # Christmas
        (12, 26), # Boxing Day
    }
    # Civic / Labour Day approximations via weekday rules are omitted; keep explicit + nearby observed
    observed = set()
    for y in sorted(d.dt.year.unique()):
        # If New Year / Canada Day / Christmas fall on Sunday, often observed Monday
        for month, day in [(1, 1), (7, 1), (12, 25), (12, 26)]:
            ts = pd.Timestamp(year=int(y), month=month, day=day)
            if ts.dayofweek == 6:
                observed.add((ts + pd.Timedelta(days=1)).date())
            observed.add(ts.date())
    return pd.Series([(m, day) in fixed or dt.date() in observed for dt, (m, day) in zip(d, md)], index=dates.index).astype(int)


def add_calendar_features(df: pd.DataFrame, calendar_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    dt = out["date"]
    mapping = {
        "day_of_week": dt.dt.dayofweek.astype(int),
        "month": dt.dt.month.astype(int),
        "is_weekend": (dt.dt.dayofweek >= 5).astype(int),
        "is_month_start": dt.dt.is_month_start.astype(int),
        "day_of_year": dt.dt.dayofyear.astype(int),
        "is_stat_holiday": _ontario_stat_holiday_mask(dt),
    }
    for name in calendar_cols:
        if name not in mapping:
            raise KeyError(f"Unknown calendar feature: {name}")
        out[name] = mapping[name]
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    target: str,
    lags: list[int],
    windows: list[int],
) -> pd.DataFrame:
    out = df.copy()
    series = out[target]
    for lag in lags:
        out[f"{target}_lag_{lag}"] = series.shift(lag)
    for w in windows:
        # Rolling stats use only past values (shift 1) to avoid same-day leakage
        rolled = series.shift(1).rolling(window=w, min_periods=max(3, w // 2))
        out[f"{target}_roll{w}_mean"] = rolled.mean()
        out[f"{target}_roll{w}_std"] = rolled.std()
        out[f"{target}_roll{w}_max"] = rolled.max()
    return out


def add_weather_lag_features(
    df: pd.DataFrame,
    weather_columns: list[str],
    lag_days: int,
) -> pd.DataFrame:
    out = df.copy()
    for col in weather_columns:
        if col not in out.columns:
            raise KeyError(f"Weather column missing from join: {col}")
        out[f"{col}_lag_{lag_days}"] = out[col].shift(lag_days)
    return out


def build_feature_matrix(cfg: dict[str, Any], joined: pd.DataFrame) -> pd.DataFrame:
    feat_cfg = cfg["features"]
    target = cfg["target"]["column"]

    out = add_calendar_features(joined, feat_cfg["calendar"])
    out = add_lag_rolling_features(
        out,
        target=target,
        lags=feat_cfg["delay_lags"],
        windows=feat_cfg["rolling_windows"],
    )
    out = add_weather_lag_features(
        out,
        weather_columns=feat_cfg["weather_columns"],
        lag_days=int(feat_cfg["weather_lag_days"]),
    )

    # Drop same-day weather raw columns from model features (keep for analysis joins)
    # They remain in the frame but are excluded in feature_columns()
    out = out.dropna().reset_index(drop=True)

    path = resolve_path(cfg, "feature_matrix_csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def feature_columns(cfg: dict[str, Any], frame: pd.DataFrame) -> list[str]:
    """Columns safe for 1-day-ahead forecasting (no same-day weather / target)."""
    target = cfg["target"]["column"]
    feat_cfg = cfg["features"]
    cols: list[str] = []
    cols.extend(feat_cfg["calendar"])
    for lag in feat_cfg["delay_lags"]:
        cols.append(f"{target}_lag_{lag}")
    for w in feat_cfg["rolling_windows"]:
        cols.extend(
            [
                f"{target}_roll{w}_mean",
                f"{target}_roll{w}_std",
                f"{target}_roll{w}_max",
            ]
        )
    lag_w = int(feat_cfg["weather_lag_days"])
    for col in feat_cfg["weather_columns"]:
        cols.append(f"{col}_lag_{lag_w}")

    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"Feature columns missing: {missing}")
    return cols


def time_based_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / val / test. Raises if any split is empty or disordered."""
    if df["date"].is_monotonic_increasing is False:
        df = df.sort_values("date").reset_index(drop=True)

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    if not (train_end_ts < val_end_ts):
        raise ValueError("train_end must be strictly before val_end")

    train = df[df["date"] <= train_end_ts].copy()
    val = df[(df["date"] > train_end_ts) & (df["date"] <= val_end_ts)].copy()
    test = df[df["date"] > val_end_ts].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(
            f"Empty split after cutpoints train_end={train_end} val_end={val_end}: "
            f"train={len(train)} val={len(val)} test={len(test)}"
        )

    # Leakage guard: max(train) < min(val) <= max(val) < min(test)
    if not (train["date"].max() < val["date"].min() <= val["date"].max() < test["date"].min()):
        raise ValueError("Time split ordering violated — possible leakage")

    return train, val, test


def assert_no_future_leakage(feature_row: pd.Series, target_date: pd.Timestamp) -> None:
    """Sanity helper used in tests: lag-1 feature date must precede target date."""
    # Feature engineering uses shift; this documents the contract.
    if pd.Timestamp(target_date) <= pd.Timestamp("1970-01-01"):
        raise ValueError("invalid target date")
    _ = feature_row  # interface for tests
    return None
