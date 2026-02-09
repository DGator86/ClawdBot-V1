from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mtf.backtest_engine import build_dataset
from mtf.constants import TF_LIST, WINDOW_BARS
from mtf.data_provider import get_latest_closed_bar, get_multi_timeframe_candles
from mtf.date_ranges import filter_by_ranges_union, parse_ranges
from mtf.feature_engine import assemble_feature_row, build_feature_frames
from mtf.models import EnsembleModel
from mtf.utils import is_timeframe_boundary, utc_now


@dataclass
class LiveConfig:
    symbols: list[str]
    target_tf: str = "1h"
    window: int = WINDOW_BARS
    train_window: int = 1000
    refit_every: int = 10
    heartbeat_seconds: int = 1
    ranges: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None


class LivePredictor:
    def __init__(self, config: LiveConfig):
        self.config = config
        self.models: Dict[str, EnsembleModel] = {}
        self.bars_by_tf: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.feature_frames: Dict[str, object] = {}
        self._hour_close_counts: Dict[str, int] = {}

    def initialize(self) -> None:
        for symbol in self.config.symbols:
            bars = get_multi_timeframe_candles(symbol, limit=self.config.window)
            if self.config.ranges:
                bars = filter_by_ranges_union(bars, self.config.ranges)
                if any(df.empty for df in bars.values()):
                    raise ValueError(
                        f"Filtered history for {symbol} has empty timeframe data. "
                        "Check --ranges for coverage."
                    )
            self.bars_by_tf[symbol] = bars
            feature_frames = build_feature_frames(bars)
            self.feature_frames[symbol] = feature_frames

            feature_df, label_series = build_dataset(bars, target_tf=self.config.target_tf)
            model = EnsembleModel()
            model.fit(feature_df.tail(self.config.train_window).values,
                      label_series.tail(self.config.train_window).values)
            self.models[symbol] = model
            self._hour_close_counts[symbol] = 0

    def _update_timeframe(self, symbol: str, timeframe: str) -> bool:
        latest = get_latest_closed_bar(symbol, timeframe)
        if latest.empty:
            return False
        bars_df = self.bars_by_tf[symbol][timeframe]
        last_ts = bars_df["timestamp"].iloc[-1] if not bars_df.empty else None
        if last_ts is not None and latest["timestamp"].iloc[0] <= last_ts:
            return False
        updated = pd.concat([bars_df, latest], ignore_index=True)
        self.bars_by_tf[symbol][timeframe] = updated.tail(self.config.window)
        return True

    def _maybe_refit(self, symbol: str) -> None:
        self._hour_close_counts[symbol] += 1
        if self._hour_close_counts[symbol] % self.config.refit_every != 0:
            return
        bars = self.bars_by_tf[symbol]
        feature_df, label_series = build_dataset(bars, target_tf=self.config.target_tf)
        model = EnsembleModel()
        model.fit(feature_df.tail(self.config.train_window).values,
                  label_series.tail(self.config.train_window).values)
        self.models[symbol] = model

    def run(self) -> None:
        self.initialize()
        os.makedirs(os.path.join("data", "predictions"), exist_ok=True)

        while True:
            now = utc_now()
            updated_any = False

            for tf in TF_LIST:
                if not is_timeframe_boundary(now, tf):
                    continue
                for symbol in self.config.symbols:
                    if self._update_timeframe(symbol, tf):
                        updated_any = True
                        if tf == self.config.target_tf:
                            self._maybe_refit(symbol)

            for symbol in self.config.symbols:
                if updated_any:
                    self.feature_frames[symbol] = build_feature_frames(self.bars_by_tf[symbol])
                feature_row = assemble_feature_row(self.feature_frames[symbol], pd.Timestamp(now))
                feature_df = pd.DataFrame([feature_row])
                model = self.models[symbol]
                prob = model.predict_proba(feature_df.values)[0]

                record = {
                    "timestamp": now.isoformat(),
                    "symbol": symbol,
                    "prob_up_next_1h": prob,
                }
                out_path = os.path.join("data", "predictions", f"{symbol.lower()}_{self.config.target_tf}.csv")
                write_header = not os.path.exists(out_path)
                pd.DataFrame([record]).to_csv(out_path, mode="a", header=write_header, index=False)
                print(f"[{record['timestamp']}] {symbol} P(up_next_1h)={prob:.4f}")

            time.sleep(self.config.heartbeat_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live multi-timeframe forecaster")
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols")
    parser.add_argument("--target_tf", type=str, default="1h")
    parser.add_argument("--window", type=int, default=WINDOW_BARS)
    parser.add_argument("--train_window", type=int, default=1000)
    parser.add_argument("--refit_every", type=int, default=10)
    parser.add_argument("--heartbeat_seconds", type=int, default=1)
    parser.add_argument(
        "--ranges",
        type=str,
        default=None,
        help="Comma-separated date ranges: start:end (UTC/ISO-8601).",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    ranges = parse_ranges(args.ranges)

    config = LiveConfig(
        symbols=symbols,
        target_tf=args.target_tf,
        window=args.window,
        train_window=args.train_window,
        refit_every=args.refit_every,
        heartbeat_seconds=args.heartbeat_seconds,
        ranges=ranges,
    )

    LivePredictor(config).run()


if __name__ == "__main__":
    main()
