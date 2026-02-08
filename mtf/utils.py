from __future__ import annotations

from datetime import datetime, timezone

from .constants import TF_LIST


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_timeframe_boundary(now: datetime, timeframe: str) -> bool:
    if timeframe == "1m":
        return now.second == 0
    if timeframe == "5m":
        return now.second == 0 and now.minute % 5 == 0
    if timeframe == "15m":
        return now.second == 0 and now.minute % 15 == 0
    if timeframe == "30m":
        return now.second == 0 and now.minute % 30 == 0
    if timeframe == "1h":
        return now.second == 0 and now.minute == 0
    if timeframe == "4h":
        return now.second == 0 and now.minute == 0 and now.hour % 4 == 0
    if timeframe == "12h":
        return now.second == 0 and now.minute == 0 and now.hour % 12 == 0
    if timeframe == "1d":
        return now.second == 0 and now.minute == 0 and now.hour == 0
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def ordered_timeframes() -> list[str]:
    return TF_LIST.copy()
