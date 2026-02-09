from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Optional

from .constants import TF_LIST, WINDOW_BARS
from .types import Bar


@dataclass
class BarBuffer:
    timeframe: str
    maxlen: int = WINDOW_BARS
    bars: Deque[Bar] = field(default_factory=deque)

    def append(self, bar: Bar) -> None:
        if self.bars and bar.timestamp <= self.bars[-1].timestamp:
            return
        self.bars.append(bar)
        while len(self.bars) > self.maxlen:
            self.bars.popleft()

    def extend(self, bars: Iterable[Bar]) -> None:
        for bar in bars:
            self.append(bar)

    def last_closed(self, now_ts: float) -> Optional[Bar]:
        for bar in reversed(self.bars):
            if bar.close_timestamp <= now_ts:
                return bar
        return None

    def to_list(self) -> list[Bar]:
        return list(self.bars)


class BarManager:
    def __init__(self, maxlen: int = WINDOW_BARS):
        self._buffers: Dict[str, BarBuffer] = {
            tf: BarBuffer(timeframe=tf, maxlen=maxlen)
            for tf in TF_LIST
        }

    def update_bars(self, timeframe: str, bars: Iterable[Bar]) -> None:
        if timeframe not in self._buffers:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        self._buffers[timeframe].extend(bars)

    def last_closed_bar(self, timeframe: str, now_ts: float) -> Optional[Bar]:
        if timeframe not in self._buffers:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return self._buffers[timeframe].last_closed(now_ts)

    def bars_for_timeframe(self, timeframe: str) -> list[Bar]:
        if timeframe not in self._buffers:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return self._buffers[timeframe].to_list()
