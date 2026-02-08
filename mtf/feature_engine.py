from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .constants import TF_LIST, TF_SECONDS


@dataclass
class FeatureFrames:
    frames: Dict[str, pd.DataFrame]

    def asof(self, timeframe: str, ts: pd.Timestamp) -> Optional[pd.Series]:
        frame = self.frames.get(timeframe)
        if frame is None or frame.empty:
            return None
        idx = frame.index
        if ts < idx.min():
            return None
        return frame.loc[:ts].iloc[-1]


def _safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num / denom.replace(0, np.nan)


def compute_tf_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df = df.sort_values("timestamp")
    df["log_close"] = np.log(df["close"].replace(0, np.nan))
    df["logret_1"] = df["log_close"].diff()
    df["logret_2"] = df["log_close"].diff(2)
    df["logret_5"] = df["log_close"].diff(5)
    df["logret_20"] = df["log_close"].diff(20)

    df["vol_10"] = df["logret_1"].rolling(10).std()
    df["vol_20"] = df["logret_1"].rolling(20).std()
    df["vol_50"] = df["logret_1"].rolling(50).std()

    tr_components = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    )
    df["true_range"] = tr_components.max(axis=1)
    df["atr_14"] = df["true_range"].rolling(14).mean()

    df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()
    df["ema_slope"] = df["ema_fast"].diff()
    df["macd"] = df["ema_fast"] - df["ema_slow"]

    rolling_mean = df["close"].rolling(20).mean()
    rolling_std = df["close"].rolling(20).std()
    df["ma_20"] = rolling_mean
    df["zscore_20"] = _safe_divide(df["close"] - rolling_mean, rolling_std)

    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["volume_z"] = _safe_divide(df["volume"] - vol_mean, vol_std)

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_pct"] = _safe_divide(df["close"] - df["open"], candle_range)
    df["wick_upper_pct"] = _safe_divide(df["high"] - df[["open", "close"]].max(axis=1), candle_range)
    df["wick_lower_pct"] = _safe_divide(df[["open", "close"]].min(axis=1) - df["low"], candle_range)
    df["close_pos"] = _safe_divide(df["close"] - df["low"], candle_range)

    roll_high = df["high"].rolling(20).max()
    roll_low = df["low"].rolling(20).min()
    df["dist_to_high"] = _safe_divide(df["close"] - roll_high, roll_high)
    df["dist_to_low"] = _safe_divide(df["close"] - roll_low, roll_low)

    df["close_time"] = df["timestamp"] + pd.to_timedelta(TF_SECONDS[timeframe], unit="s")
    df = df.set_index("close_time")

    keep_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "logret_1",
        "logret_2",
        "logret_5",
        "logret_20",
        "vol_10",
        "vol_20",
        "vol_50",
        "atr_14",
        "ema_slope",
        "macd",
        "ma_20",
        "zscore_20",
        "volume_z",
        "body_pct",
        "wick_upper_pct",
        "wick_lower_pct",
        "close_pos",
        "dist_to_high",
        "dist_to_low",
    ]
    return df[keep_cols]


def build_feature_frames(bars_by_tf: Dict[str, pd.DataFrame]) -> FeatureFrames:
    frames: Dict[str, pd.DataFrame] = {}
    for tf, df in bars_by_tf.items():
        frames[tf] = compute_tf_features(df, tf)
    return FeatureFrames(frames=frames)


def _momentum_slope(series: pd.Series, window: int) -> float:
    if series is None or series.empty or len(series) < window:
        return float("nan")
    values = series.iloc[-window:].values
    x = np.arange(len(values))
    if np.all(np.isnan(values)):
        return float("nan")
    coeffs = np.polyfit(x, values, 1)
    return float(coeffs[0])


def assemble_feature_row(
    feature_frames: FeatureFrames,
    now_ts: pd.Timestamp,
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    last_price = None
    last_price_row = feature_frames.asof("1m", now_ts)
    if last_price_row is not None:
        last_price = float(last_price_row.get("close", np.nan))

    for tf in TF_LIST:
        row = feature_frames.asof(tf, now_ts)
        if row is None:
            continue
        for col, val in row.items():
            features[f"{tf}__{col}"] = float(val) if pd.notna(val) else float("nan")

    if last_price is not None:
        features["price"] = last_price

    now_dt = now_ts.to_pydatetime()
    next_hour = now_dt.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
    seconds_to_close = (next_hour - now_dt).total_seconds()
    features["time_to_close_1h"] = float(seconds_to_close)

    for tf in ["1h", "4h", "12h"]:
        row = feature_frames.asof(tf, now_ts)
        if row is None or last_price is None:
            features[f"price_vs_ma_{tf}"] = float("nan")
            continue
        ma_col = row.get("ma_20")
        if pd.isna(ma_col) or ma_col == 0 or last_price == 0:
            features[f"price_vs_ma_{tf}"] = float("nan")
        else:
            features[f"price_vs_ma_{tf}"] = float((last_price - ma_col) / ma_col)

    one_min_frame = feature_frames.frames.get("1m")
    if one_min_frame is not None and not one_min_frame.empty:
        close_series = one_min_frame["close"].dropna()
        features["mom_slope_1m_3"] = _momentum_slope(close_series, 3)
        features["mom_slope_1m_5"] = _momentum_slope(close_series, 5)
        features["mom_slope_1m_15"] = _momentum_slope(close_series, 15)

        if len(close_series) >= 2:
            returns = close_series.pct_change().dropna()
            features["intrahour_vol"] = float(returns[-60:].std()) if len(returns) >= 2 else float("nan")
            features["intrahour_drawdown"] = float((close_series[-60:].min() - close_series.iloc[-1]) / close_series.iloc[-1]) if len(close_series) >= 2 else float("nan")
            features["intrahour_runup"] = float((close_series[-60:].max() - close_series.iloc[-1]) / close_series.iloc[-1]) if len(close_series) >= 2 else float("nan")

    return features
