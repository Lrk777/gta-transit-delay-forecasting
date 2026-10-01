"""Train baselines + HistGradientBoosting with time-based validation selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.features import build_feature_matrix, feature_columns, join_sources, time_based_split
from src.ingest_transit import ingest_transit
from src.ingest_weather import ingest_weather
from src.metrics import pick_best_by_val, regression_metrics, residual_frame, slice_error_summary
from src.models import build_models


def _xy(df: pd.DataFrame, cols: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    return df[cols].to_numpy(dtype=float), df[target].to_numpy(dtype=float)


def prepare_data(cfg: dict[str, Any], *, use_fixture: bool | None = None) -> pd.DataFrame:
    transit = ingest_transit(cfg, use_fixture=use_fixture)
    weather = ingest_weather(cfg, use_fixture=use_fixture)
    joined = join_sources(transit, weather)
    return build_feature_matrix(cfg, joined)


def train_and_evaluate(cfg: dict[str, Any], *, use_fixture: bool | None = None) -> dict[str, Any]:
    rng_seed = int(cfg["seed"])
    np.random.seed(rng_seed)

    matrix = prepare_data(cfg, use_fixture=use_fixture)
    target = cfg["target"]["column"]
    cols = feature_columns(cfg, matrix)
    train, val, test = time_based_split(
        matrix,
        train_end=cfg["splits"]["train_end"],
        val_end=cfg["splits"]["val_end"],
    )

    models = build_models(cfg, cols)
    X_train, y_train = _xy(train, cols, target)
    X_val, y_val = _xy(val, cols, target)
    X_test, y_test = _xy(test, cols, target)

    val_metrics: dict[str, dict[str, float]] = {}
    test_metrics: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}

    for name, model in models.items():
        if name == "hist_gradient_boosting" and cfg["models"]["hist_gradient_boosting"].get(
            "early_stopping", False
        ):
            # Explicit chronological val set — do not carve a random fraction from train
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)
        fitted[name] = model
        val_metrics[name] = regression_metrics(y_val, model.predict(X_val))
        test_metrics[name] = regression_metrics(y_test, model.predict(X_test))

    best = pick_best_by_val(val_metrics, key="mae")
    best_model = fitted[best]
    y_pred_test = best_model.predict(X_test)

    extras = test[
        [c for c in ["precipitation_sum_lag_1", "snowfall_sum_lag_1", "is_weekend"] if c in test.columns]
    ].reset_index(drop=True)
    resid = residual_frame(test["date"], y_test, y_pred_test, extras=extras)
    error_analysis = slice_error_summary(resid)

    # Persist artifacts
    reports = resolve_path(cfg, "reports_dir")
    figures = resolve_path(cfg, "figures_dir")
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    pred_path = resolve_path(cfg, "predictions_csv")
    resid.to_csv(pred_path, index=False)

    models_dir = reports / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / f"{best}.joblib")

    _plot_parity(y_test, y_pred_test, figures / "parity_test.png", title=f"Test parity — {best}")
    _plot_residuals_over_time(resid, figures / "residuals_over_time.png")
    _plot_mae_by_dow(error_analysis["mae_by_day_of_week"], figures / "mae_by_dow.png")
    _plot_model_bars(test_metrics, figures / "test_mae_by_model.png", best=best)

    payload = {
        "seed": rng_seed,
        "target": target,
        "n_features": len(cols),
        "features": cols,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "split_ranges": {
            "train": [str(train["date"].min().date()), str(train["date"].max().date())],
            "val": [str(val["date"].min().date()), str(val["date"].max().date())],
            "test": [str(test["date"].min().date()), str(test["date"].max().date())],
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "selected_model": best,
        "selection_rule": "lowest validation MAE",
        "data_notes": {
            "transit": cfg["transit"]["attribution"].strip(),
            "weather": cfg["weather"]["attribution"],
            "forecast_contract": (
                "1-day-ahead: lag/rolling delay features and lag-1 weather only; "
                "no same-day weather or target leakage; chronological splits."
            ),
        },
    }
    metrics_path = resolve_path(cfg, "metrics_json")
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    err_path = resolve_path(cfg, "error_analysis_json")
    err_path.write_text(
        json.dumps({"selected_model": best, **error_analysis}, indent=2, default=str),
        encoding="utf-8",
    )

    return payload


def _plot_parity(y_true, y_pred, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.55, s=18, c="#1f4e79")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#c45c26", lw=1.5, label="ideal")
    ax.set_xlabel("Actual total delay (min)")
    ax.set_ylabel("Predicted total delay (min)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_residuals_over_time(resid: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axhline(0, color="#888", lw=1)
    ax.plot(resid["date"], resid["residual"], color="#1f4e79", lw=1.0)
    ax.set_title("Test residuals over time (pred − actual)")
    ax.set_ylabel("Minutes")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_mae_by_dow(by_dow: dict, path: Path) -> None:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    labels = [d for d in order if d in by_dow]
    vals = [by_dow[d]["mean"] for d in labels]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(labels, vals, color="#2a6f7a")
    ax.set_ylabel("MAE (minutes)")
    ax.set_title("Test MAE by day of week")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_model_bars(test_metrics: dict[str, dict[str, float]], path: Path, best: str) -> None:
    names = list(test_metrics.keys())
    maes = [test_metrics[n]["mae"] for n in names]
    colors = ["#c45c26" if n == best else "#1f4e79" for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(names, maes, color=colors)
    ax.set_ylabel("Test MAE (minutes)")
    ax.set_title("Test MAE by model (selected in orange)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train GTA transit delay forecasters")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Use checked-in fixture CSVs (offline / CI)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.fixture:
        cfg["use_fixture"] = True
    payload = train_and_evaluate(cfg, use_fixture=cfg.get("use_fixture", False))
    print(json.dumps(
        {
            "selected_model": payload["selected_model"],
            "val_mae": {k: v["mae"] for k, v in payload["val_metrics"].items()},
            "test_mae": {k: v["mae"] for k, v in payload["test_metrics"].items()},
            "split_sizes": payload["split_sizes"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
