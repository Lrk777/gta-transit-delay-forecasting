"""Ingest TTC Bus Delay Data from City of Toronto Open Data (CKAN)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import resolve_path

USER_AGENT = "gta-transit-delay-forecasting/0.1 (+https://github.com/Lrk777/gta-transit-delay-forecasting)"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_package_resources(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return CKAN resources keyed by resource name."""
    transit = cfg["transit"]
    resp = _session().get(
        transit["ckan_package_url"],
        params={"id": transit["package_id"]},
        timeout=90,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN package_show failed: {payload}")
    resources = payload["result"]["resources"]
    return {r["name"]: r for r in resources}


def download_year_workbook(resource: dict[str, Any], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = _session().get(resource["url"], timeout=180)
    resp.raise_for_status()
    if resp.content[:2] != b"PK":
        raise RuntimeError(
            f"Expected XLSX (ZIP) bytes for {resource.get('name')}; "
            f"got magic={resp.content[:20]!r}"
        )
    dest.write_bytes(resp.content)
    return dest


def _normalize_incident_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    required = {"date", "min delay"}
    missing = required - set(cols)
    if missing:
        raise ValueError(f"Incident frame missing columns {missing}; have {list(df.columns)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[cols["date"]], errors="coerce"),
            "min_delay": pd.to_numeric(df[cols["min delay"]], errors="coerce"),
            "route": df[cols["route"]].astype(str) if "route" in cols else pd.NA,
            "incident": df[cols["incident"]].astype(str) if "incident" in cols else pd.NA,
        }
    )
    out = out.dropna(subset=["date", "min_delay"])
    out["date"] = out["date"].dt.normalize()
    out["min_delay"] = out["min_delay"].clip(lower=0)
    return out


def load_workbook(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    return _normalize_incident_frame(df)


def aggregate_daily(incidents: pd.DataFrame) -> pd.DataFrame:
    g = incidents.groupby("date", as_index=False).agg(
        total_delay_min=("min_delay", "sum"),
        n_incidents=("min_delay", "size"),
        mean_delay_min=("min_delay", "mean"),
        median_delay_min=("min_delay", "median"),
        p90_delay_min=("min_delay", lambda s: float(s.quantile(0.9))),
    )
    g = g.sort_values("date").reset_index(drop=True)
    return g


def ingest_transit(
    cfg: dict[str, Any],
    *,
    use_fixture: bool | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download yearly delay workbooks (or fixture) and write daily aggregates."""
    use_fixture = cfg.get("use_fixture", False) if use_fixture is None else use_fixture
    out_path = resolve_path(cfg, "transit_daily_csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if use_fixture:
        fixture = Path(cfg["_root"]) / cfg["fixture_transit"]
        daily = pd.read_csv(fixture, parse_dates=["date"])
        daily.to_csv(out_path, index=False)
        return daily

    if out_path.exists() and not refresh and out_path.stat().st_size > 0:
        return pd.read_csv(out_path, parse_dates=["date"])

    raw_dir = resolve_path(cfg, "raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    resources = fetch_package_resources(cfg)
    frames: list[pd.DataFrame] = []
    for name in cfg["transit"]["resource_names"]:
        if name not in resources:
            raise KeyError(
                f"CKAN resource {name!r} not found. Available: {sorted(resources)[:12]}…"
            )
        dest = raw_dir / f"{name}.xlsx"
        if not dest.exists() or dest.stat().st_size < 1000:
            download_year_workbook(resources[name], dest)
        frames.append(load_workbook(dest))

    incidents = pd.concat(frames, ignore_index=True)
    daily = aggregate_daily(incidents)
    # Keep provenance sidecar
    meta_path = raw_dir / "transit_ingest_meta.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"package_id={cfg['transit']['package_id']}",
                f"resources={','.join(cfg['transit']['resource_names'])}",
                f"n_incidents={len(incidents)}",
                f"n_days={len(daily)}",
                f"date_min={daily['date'].min().date()}",
                f"date_max={daily['date'].max().date()}",
                f"attribution={cfg['transit']['attribution'].strip()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    daily.to_csv(out_path, index=False)
    return daily


def load_transit_daily_from_bytes(xlsx_bytes: bytes) -> pd.DataFrame:
    """Helper for tests: parse a single workbook from memory."""
    df = pd.read_excel(io.BytesIO(xlsx_bytes))
    return aggregate_daily(_normalize_incident_frame(df))
