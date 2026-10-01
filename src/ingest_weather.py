"""Ingest Toronto daily weather from Open-Meteo historical API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import resolve_path

USER_AGENT = "gta-transit-delay-forecasting/0.1 (+https://github.com/Lrk777/gta-transit-delay-forecasting)"


def fetch_open_meteo(cfg: dict[str, Any]) -> dict[str, Any]:
    w = cfg["weather"]
    params = {
        "latitude": w["latitude"],
        "longitude": w["longitude"],
        "start_date": w["start_date"],
        "end_date": w["end_date"],
        "daily": ",".join(w["daily_variables"]),
        "timezone": w["timezone"],
    }
    resp = requests.get(
        w["url"],
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=90,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "daily" not in payload:
        raise ValueError(f"Unexpected Open-Meteo keys: {list(payload)}")
    return payload


def daily_json_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    daily = payload["daily"]
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def ingest_weather(
    cfg: dict[str, Any],
    *,
    use_fixture: bool | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    use_fixture = cfg.get("use_fixture", False) if use_fixture is None else use_fixture
    out_path = resolve_path(cfg, "weather_daily_csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if use_fixture:
        fixture = Path(cfg["_root"]) / cfg["fixture_weather"]
        df = pd.read_csv(fixture, parse_dates=["date"])
        df.to_csv(out_path, index=False)
        return df

    if out_path.exists() and not refresh and out_path.stat().st_size > 0:
        return pd.read_csv(out_path, parse_dates=["date"])

    payload = fetch_open_meteo(cfg)
    df = daily_json_to_frame(payload)
    raw_dir = resolve_path(cfg, "raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "toronto_daily_openmeteo.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    df.to_csv(out_path, index=False)
    return df
