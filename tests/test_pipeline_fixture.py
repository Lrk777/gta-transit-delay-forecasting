"""Slow / integration test: full train on fixture config."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.train import train_and_evaluate


@pytest.mark.slow
def test_full_pipeline_fixture(tmp_path):
    cfg = load_config("configs/fixture.yaml")
    for key, val in list(cfg["paths"].items()):
        cfg["paths"][key] = str(tmp_path / val)

    payload = train_and_evaluate(cfg, use_fixture=True)
    assert payload["selected_model"] in payload["test_metrics"]
    assert payload["split_sizes"]["train"] > 0
    assert Path(cfg["paths"]["metrics_json"]).exists()
    assert Path(cfg["paths"]["predictions_csv"]).exists()
    # Baselines must be present
    assert "naive_lag1" in payload["test_metrics"]
    assert "hist_gradient_boosting" in payload["test_metrics"]
