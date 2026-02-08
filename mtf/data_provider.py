from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from scripts.forecaster import data as forecaster_data

from .constants import TF_LIST, TF_MINUTES, TF_SECONDS
from .types import Bar


@dataclass
class CandleFetchResult:
    bars: List[Bar]
    source: str


def _bars_to_frame(bars: List[Bar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _convert_bars(bars: List[forecaster_data.Bar], timeframe: str) -> List[Bar]:
    return [
        Bar(
            timestamp=bar.timestamp,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            timeframe=timeframe,
        )
        for bar in bars
    ]


def _filter_closed(bars: List[Bar], now_ts: float) -> List[Bar]:
    return [bar for bar in bars if bar.close_timestamp <= now_ts]


def fetch_timeframe_bars(symbol: str, timeframe: str, limit: int) -> CandleFetchResult:
    if timeframe not in TF_LIST:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    now_ts = time.time()
    source_parts: List[str] = []

    coinbase_bars = []
    kraken_bars = []

    granularity = TF_SECONDS[timeframe]
    interval_minutes = TF_MINUTES[timeframe]

    coinbase_bars = forecaster_data.fetch_coinbase_klines(
        symbol, granularity=granularity, limit=limit
    )
    if coinbase_bars:
        source_parts.append("coinbase")

    if len(coinbase_bars) < limit:
        kraken_bars = forecaster_data.fetch_kraken_klines(
            symbol, interval=interval_minutes, limit=limit
        )
        if kraken_bars:
            source_parts.append("kraken")

    merged = forecaster_data._merge_bar_lists(coinbase_bars, kraken_bars)
    merged = _convert_bars(merged, timeframe)

    if len(merged) > limit:
        merged = merged[-limit:]

    merged = _filter_closed(merged, now_ts)

    source = "+".join(source_parts) if source_parts else "none"
    return CandleFetchResult(bars=merged, source=source)


def get_closed_candles(symbol: str, timeframe: str, limit: int = 2000) -> pd.DataFrame:
    result = fetch_timeframe_bars(symbol, timeframe, limit)
    return _bars_to_frame(result.bars)


def get_latest_closed_bar(symbol: str, timeframe: str) -> pd.DataFrame:
    df = get_closed_candles(symbol, timeframe, limit=3)
    if df.empty:
        return df
    return df.tail(1).reset_index(drop=True)


def get_multi_timeframe_candles(
    symbol: str,
    limit: int = 2000,
) -> Dict[str, pd.DataFrame]:
    return {
        tf: get_closed_candles(symbol, tf, limit=limit)
        for tf in TF_LIST
    }
