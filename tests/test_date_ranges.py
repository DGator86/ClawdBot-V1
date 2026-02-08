from datetime import timezone

import pandas as pd
import pytest

from mtf.date_ranges import filter_by_ranges_union, parse_ranges


def _make_bars(start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=periods, freq="H", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": range(periods),
            "high": range(periods),
            "low": range(periods),
            "close": range(periods),
        }
    )


def test_parse_ranges_returns_utc_timestamps() -> None:
    ranges = parse_ranges("2023-01-01:2023-01-02,2023-01-03T00:00:00Z:2023-01-04T00:00:00Z")
    assert ranges is not None
    assert len(ranges) == 2
    assert ranges[0][0].tzinfo is not None
    assert ranges[0][0].tzinfo == timezone.utc


def test_parse_ranges_rejects_overlaps() -> None:
    with pytest.raises(ValueError, match="overlap"):
        parse_ranges("2023-01-01:2023-01-03,2023-01-02:2023-01-04")


def test_filter_by_ranges_union() -> None:
    bars = _make_bars("2023-01-01", periods=6)
    bars_by_tf = {"1h": bars.copy(), "4h": bars.copy()}
    ranges = parse_ranges("2023-01-01T01:00:00Z:2023-01-01T02:00:00Z,2023-01-01T05:00:00Z:2023-01-01T06:00:00Z")
    assert ranges is not None
    filtered = filter_by_ranges_union(bars_by_tf, ranges)
    assert len(filtered["1h"]) == 3
    assert filtered["1h"]["timestamp"].iloc[0] == pd.Timestamp("2023-01-01T01:00:00Z")
