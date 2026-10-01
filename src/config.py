"""Load YAML experiment config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    cfg["_config_path"] = str(cfg_path)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    root = Path(cfg["_root"])
    return root / cfg["paths"][key]
