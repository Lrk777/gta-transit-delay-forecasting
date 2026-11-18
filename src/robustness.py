"""Robustness checks for the headline result ("~10% better than yesterday's value").

1. A stronger simple baseline: the trailing 7-day mean (already a leakage-safe feature).
2. Weather ablation: refit Ridge and HGB without the four lag-1 weather columns.
3. 95% intervals for test-set MAE differences with a moving-block bootstrap (7-day blocks,
   because errors on neighbouring days are correlated).

Usage: python -m src.robustness [--fixture]
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from .config import load_config, resolve_path
from .features import feature_columns, time_based_split
from .models import LagBaseline, build_models
from .train import _xy, prepare_data


def block_bootstrap_diff(err_a: np.ndarray, err_b: np.ndarray, block: int = 7, n_boot: int = 2000, seed: int = 42) -> dict:
    """CI for mean(err_a - err_b). Negative = A has lower absolute error."""
    d = np.asarray(err_a) - np.asarray(err_b)
    n = len(d)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "share_a_better": float((boots < 0).mean())}


def run(cfg: dict[str, Any], use_fixture: bool | None = None) -> dict[str, Any]:
    matrix = prepare_data(cfg, use_fixture=use_fixture)
    target = cfg["target"]["column"]
    cols = feature_columns(cfg, matrix)
    weather = [c for c in cols if any(c.startswith(w) for w in cfg["features"]["weather_columns"])]
    no_weather = [c for c in cols if c not in weather]
    train, val, test = time_based_split(matrix, train_end=cfg["splits"]["train_end"], val_end=cfg["splits"]["val_end"])

    abs_err: dict[str, np.ndarray] = {}
    val_mae: dict[str, float] = {}
    for label, use_cols in [("", cols), ("_no_weather", no_weather)]:
        models = build_models(cfg, use_cols)
        Xtr, ytr = _xy(train, use_cols, target)
        Xva, yva = _xy(val, use_cols, target)
        Xte, yte = _xy(test, use_cols, target)
        for name in ["ridge", "hist_gradient_boosting"]:
            m = models[name]
            if name == "hist_gradient_boosting" and cfg["models"][name].get("early_stopping", False):
                m.fit(Xtr, ytr, X_val=Xva, y_val=yva)
            else:
                m.fit(Xtr, ytr)
            abs_err[name + label] = np.abs(yte - m.predict(Xte))
            val_mae[name + label] = float(np.abs(yva - m.predict(Xva)).mean())
        if label == "":
            for bname, feat in [("naive_lag1", f"{target}_lag_1"), ("rolling_mean7", f"{target}_roll7_mean")]:
                b = LagBaseline(lag_feature_index=use_cols.index(feat), name=bname).fit(Xtr, ytr)
                abs_err[bname] = np.abs(yte - b.predict(Xte))
                val_mae[bname] = float(np.abs(yva - b.predict(Xva)).mean())

    mae = {k: float(v.mean()) for k, v in abs_err.items()}
    seed = int(cfg["seed"])
    comparisons = {
        "ridge_vs_naive_lag1": ("ridge", "naive_lag1"),
        "ridge_vs_rolling_mean7": ("ridge", "rolling_mean7"),
        "hgb_vs_rolling_mean7": ("hist_gradient_boosting", "rolling_mean7"),
        "ridge_vs_hgb": ("ridge", "hist_gradient_boosting"),
        "ridge_with_vs_without_weather": ("ridge", "ridge_no_weather"),
        "hgb_with_vs_without_weather": ("hist_gradient_boosting", "hist_gradient_boosting_no_weather"),
        "ridge_no_weather_vs_naive_lag1": ("ridge_no_weather", "naive_lag1"),
        "ridge_no_weather_vs_rolling_mean7": ("ridge_no_weather", "rolling_mean7"),
    }
    cis = {k: block_bootstrap_diff(abs_err[a], abs_err[b], seed=seed) for k, (a, b) in comparisons.items()}
    selected = min(val_mae, key=val_mae.get)
    out = {"test_days": len(test), "selected_on_validation": selected, "val_mae": val_mae, "test_mae": mae, "bootstrap_95ci": cis, "weather_features": weather}
    path = resolve_path(cfg, "reports_dir") / "robustness.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--fixture", action="store_true")
    a = p.parse_args(argv)
    out = run(load_config(a.config), use_fixture=True if a.fixture else None)
    for k, v in sorted(out["val_mae"].items(), key=lambda kv: kv[1]):
        print(f"{k:<38} val MAE {v:8.1f}   test MAE {out['test_mae'][k]:8.1f}")
    print("selected on validation:", out["selected_on_validation"])
    for k, v in out["bootstrap_95ci"].items():
        print(f"{k:<32} diff {v['mean_diff']:8.1f}  95% CI [{v['ci_low']:.1f}, {v['ci_high']:.1f}]  P(first better)={v['share_a_better']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
