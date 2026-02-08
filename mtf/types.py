from __future__ import annotations

from dataclasses import dataclass

from .constants import TF_SECONDS


@dataclass(frozen=True)
class Bar:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str

    @property
    def close_timestamp(self) -> float:
        return self.timestamp + TF_SECONDS[self.timeframe]
