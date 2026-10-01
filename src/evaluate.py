"""Pretty-print metrics / error analysis from a completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config, resolve_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate / summarize delay forecast run")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    metrics_path = resolve_path(cfg, "metrics_json")
    err_path = resolve_path(cfg, "error_analysis_json")
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}; run: python -m src.train")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print("=== Split sizes ===")
    print(json.dumps(metrics["split_sizes"], indent=2))
    print("\n=== Validation MAE ===")
    for name, m in metrics["val_metrics"].items():
        mark = " *" if name == metrics["selected_model"] else ""
        print(f"  {name:24s}  mae={m['mae']:.1f}  rmse={m['rmse']:.1f}  r2={m['r2']:.3f}{mark}")
    print("\n=== Test metrics ===")
    for name, m in metrics["test_metrics"].items():
        mark = " *" if name == metrics["selected_model"] else ""
        print(
            f"  {name:24s}  mae={m['mae']:.1f}  rmse={m['rmse']:.1f}  "
            f"r2={m['r2']:.3f}  mape={m['mape_pct']:.1f}%{mark}"
        )

    if err_path.exists():
        err = json.loads(err_path.read_text(encoding="utf-8"))
        print("\n=== Error analysis (selected model, test) ===")
        print("MAE by day of week:")
        for day, stats in err.get("mae_by_day_of_week", {}).items():
            print(f"  {day:10s}  mean={stats['mean']:.1f}  n={stats['count']}")
        if err.get("mae_by_precip_bucket_lag1"):
            print("MAE by lag-1 precip bucket:")
            for bucket, stats in err["mae_by_precip_bucket_lag1"].items():
                print(f"  {bucket:10s}  mean={stats['mean']:.1f}  n={stats['count']}")
        print("Worst abs-error days:")
        for row in err.get("worst_abs_error_days", [])[:5]:
            print(
                f"  {row['date']}  true={row['y_true']:.0f}  "
                f"pred={row['y_pred']:.0f}  abs={row['abs_error']:.0f}  ({row['day_of_week']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
