from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mtf.backtest_engine import BacktestConfig, build_dataset, compute_metrics, walk_forward_backtest
from mtf.constants import PRIMARY_TARGET_TF, TF_LIST, WINDOW_BARS
from mtf.data_provider import get_multi_timeframe_candles
from mtf.date_ranges import filter_by_range, parse_ranges


def _run_for_symbol(
    symbol: str,
    config: BacktestConfig,
    window: int,
    ranges: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]],
) -> Dict[str, pd.DataFrame]:
    bars_by_tf = get_multi_timeframe_candles(symbol, limit=window)
    if not ranges:
        feature_df, label_series = build_dataset(bars_by_tf, target_tf=config.target_tf)
        results = walk_forward_backtest(feature_df, label_series, config)
        return {
            "features": feature_df,
            "labels": label_series,
            "predictions": results.predictions,
            "metrics": results.metrics,
            "per_regime": results.per_regime,
        }

    range_payloads = []
    combined_predictions: List[pd.DataFrame] = []
    for idx, (start_ts, end_ts) in enumerate(ranges, start=1):
        ranged_bars = filter_by_range(bars_by_tf, start_ts, end_ts)
        feature_df, label_series = build_dataset(ranged_bars, target_tf=config.target_tf)
        if feature_df.empty or len(feature_df) <= config.train_window:
            range_payloads.append(
                {
                    "range": {
                        "start": start_ts.isoformat(),
                        "end": end_ts.isoformat(),
                    },
                    "error": "Not enough data for walk-forward backtest.",
                }
            )
            continue
        results = walk_forward_backtest(feature_df, label_series, config)
        combined_predictions.append(results.predictions)
        range_payloads.append(
            {
                "range": {
                    "start": start_ts.isoformat(),
                    "end": end_ts.isoformat(),
                },
                "metrics": results.metrics,
                "per_regime": results.per_regime,
            }
        )

    if combined_predictions:
        merged_predictions = pd.concat(combined_predictions).sort_index()
        merged_features, _ = build_dataset(bars_by_tf, target_tf=config.target_tf)
        agg_metrics, agg_regime = compute_metrics(merged_predictions, merged_features)
    else:
        merged_predictions = pd.DataFrame(columns=["prob_up", "target"]).astype(float)
        agg_metrics, agg_regime = {}, {}

    return {
        "features": pd.DataFrame(),
        "labels": pd.Series(dtype=int),
        "predictions": merged_predictions,
        "metrics": agg_metrics,
        "per_regime": agg_regime,
        "ranges": range_payloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-timeframe walk-forward backtest")
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols")
    parser.add_argument("--target_tf", type=str, default=PRIMARY_TARGET_TF)
    parser.add_argument("--window", type=int, default=WINDOW_BARS)
    parser.add_argument("--train_window", type=int, default=1000)
    parser.add_argument("--refit_every", type=int, default=10)
    parser.add_argument(
        "--ranges",
        type=str,
        default=None,
        help="Comma-separated date ranges: start:end (UTC/ISO-8601).",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.target_tf not in TF_LIST:
        raise ValueError(f"Unsupported target timeframe: {args.target_tf}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("data", "backtests", run_id)
    os.makedirs(output_dir, exist_ok=True)

    ranges = parse_ranges(args.ranges)

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
            "ranges": args.ranges,
        },
        "results": {},
    }

    for symbol in symbols:
        result = _run_for_symbol(symbol, config, args.window, ranges)
        pred_path = os.path.join(output_dir, f"{symbol.lower()}_predictions.parquet")
        if not result["predictions"].empty:
            result["predictions"].to_parquet(pred_path)

        metrics = result["metrics"]
        summary["results"][symbol] = {
            "metrics": metrics,
            "per_regime": result["per_regime"],
            "ranges": result.get("ranges"),
            "predictions_path": pred_path,
        }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(summary["config"], f, indent=2)


if __name__ == "__main__":
    main()
