from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict

import pandas as pd

from mtf.backtest_engine import BacktestConfig, build_dataset, walk_forward_backtest
from mtf.constants import PRIMARY_TARGET_TF, TF_LIST, WINDOW_BARS
from mtf.data_provider import get_multi_timeframe_candles


def _run_for_symbol(symbol: str, config: BacktestConfig, window: int) -> Dict[str, pd.DataFrame]:
    bars_by_tf = get_multi_timeframe_candles(symbol, limit=window)
    feature_df, label_series = build_dataset(bars_by_tf, target_tf=config.target_tf)
    results = walk_forward_backtest(feature_df, label_series, config)
    return {
        "features": feature_df,
        "labels": label_series,
        "predictions": results.predictions,
        "metrics": results.metrics,
        "per_regime": results.per_regime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-timeframe walk-forward backtest")
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols")
    parser.add_argument("--target_tf", type=str, default=PRIMARY_TARGET_TF)
    parser.add_argument("--window", type=int, default=WINDOW_BARS)
    parser.add_argument("--train_window", type=int, default=1000)
    parser.add_argument("--refit_every", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.target_tf not in TF_LIST:
        raise ValueError(f"Unsupported target timeframe: {args.target_tf}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("data", "backtests", run_id)
    os.makedirs(output_dir, exist_ok=True)

    config = BacktestConfig(
        target_tf=args.target_tf,
        train_window=args.train_window,
        refit_every=args.refit_every,
    )

    summary = {
        "run_id": run_id,
        "config": {
            "symbols": symbols,
            "target_tf": args.target_tf,
            "window": args.window,
            "train_window": args.train_window,
            "refit_every": args.refit_every,
        },
        "results": {},
    }

    for symbol in symbols:
        result = _run_for_symbol(symbol, config, args.window)
        pred_path = os.path.join(output_dir, f"{symbol.lower()}_predictions.parquet")
        result["predictions"].to_parquet(pred_path)

        metrics = result["metrics"]
        summary["results"][symbol] = {
            "metrics": metrics,
            "per_regime": result["per_regime"],
            "predictions_path": pred_path,
        }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(summary["config"], f, indent=2)


if __name__ == "__main__":
    main()
