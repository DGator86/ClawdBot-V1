import pandas as pd

from mtf.backtest_engine import build_dataset
from mtf.feature_engine import assemble_feature_row, build_feature_frames


def _make_df(start, periods, freq, base=100.0):
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + (pd.Series(range(periods)) * 0.1).values,
            "volume": 1000,
        }
    )


def test_no_future_bars_used_in_features():
    bars_by_tf = {
        "1m": _make_df("2024-01-01 00:00:00", 5, "1min"),
        "5m": _make_df("2024-01-01 00:00:00", 2, "5min"),
        "15m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "30m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "1h": _make_df("2024-01-01 00:00:00", 2, "1h"),
        "4h": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "12h": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "1d": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
    }
    feature_frames = build_feature_frames(bars_by_tf)
    now_ts = pd.Timestamp("2024-01-01 00:04:00", tz="UTC")
    feature_row = assemble_feature_row(feature_frames, now_ts)

    for key in feature_row:
        if key.startswith("1h__"):
            raise AssertionError("1h features should not be present before close")


def test_label_alignment_for_next_close():
    bars_by_tf = {
        "1m": _make_df("2024-01-01 00:00:00", 10, "1min"),
        "5m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "15m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "30m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "1h": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 01:00:00",
                        "2024-01-01 02:00:00",
                    ],
                    utc=True,
                ),
                "open": [100, 110, 105],
                "high": [101, 112, 108],
                "low": [99, 108, 102],
                "close": [110, 105, 120],
                "volume": [1000, 1100, 1200],
            }
        ),
        "4h": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "12h": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "1d": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
    }

    features, labels = build_dataset(bars_by_tf, target_tf="1h")
    assert labels.iloc[0] == 0
    assert labels.iloc[1] == 1
    assert len(features) == len(labels)


def test_missing_bars_are_handled():
    bars_by_tf = {
        "1m": _make_df("2024-01-01 00:00:00", 5, "1min"),
        "5m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "15m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "30m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "1h": _make_df("2024-01-01 00:00:00", 2, "1h"),
        "4h": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "12h": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        "1d": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
    }

    feature_frames = build_feature_frames(bars_by_tf)
    now_ts = pd.Timestamp("2024-01-01 01:00:00", tz="UTC")
    feature_row = assemble_feature_row(feature_frames, now_ts)
    assert feature_row
