"""One-command pipeline: ingest → features → train → evaluate summary."""

from __future__ import annotations

import argparse

from src.config import load_config
from src.evaluate import main as evaluate_main
from src.train import train_and_evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run full GTA transit delay pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Offline mode using tests/fixtures daily CSVs",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.fixture:
        cfg["use_fixture"] = True
    train_and_evaluate(cfg, use_fixture=cfg.get("use_fixture", False))
    return evaluate_main(["--config", args.config] if args.config else [])


if __name__ == "__main__":
    raise SystemExit(main())
