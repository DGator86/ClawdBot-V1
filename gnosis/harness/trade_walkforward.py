"""
Trade-index walk-forward harness.

Key idea:
- Folds are defined in *trades* (print rows), so they are invariant to domains.D0.n_trades.
- Purge/embargo are also in trades (default = horizon_trades).
- When you later aggregate prints->bars inside a fold, bar boundaries are local to that fold.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class TradeFold:
    fold_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


class TradeWalkForwardHarness:
    """
    Nested walk-forward with purge/embargo in *trades* (not bars).

    Config keys supported (uses whichever is present):
      - outer_folds (default 8)
      - train_trades OR train_bars
      - val_trades   OR val_bars
      - test_trades  OR test_bars
      - purge_trades ("HORIZON" or int) default "HORIZON"
      - embargo_trades ("HORIZON" or int) default "HORIZON"
      - horizon_trades OR horizon_bars (needs trades_per_bar if using horizon_bars)

    trades_per_bar is only used to convert *_bars -> *_trades and horizon_bars -> horizon_trades.
    """

    def __init__(self, config: dict, trades_per_bar: int, horizon_bars_default: int = 10):
        self.config = config or {}
        self.outer_folds = int(self.config.get("outer_folds", 8))

        tpb = max(1, int(trades_per_bar))

        # train/val/test in trades
        self.train_trades = int(self.config.get("train_trades", int(self.config.get("train_bars", 500)) * tpb))
        self.val_trades   = int(self.config.get("val_trades",   int(self.config.get("val_bars",  100)) * tpb))
        self.test_trades  = int(self.config.get("test_trades",  int(self.config.get("test_bars", 100)) * tpb))

        # horizon in trades
        horizon_trades = self.config.get("horizon_trades", None)
        if horizon_trades is None:
            horizon_bars = int(self.config.get("horizon_bars", horizon_bars_default))
            horizon_trades = horizon_bars * tpb
        self.horizon_trades = int(horizon_trades)

        # purge/embargo in trades
        purge_trades = self.config.get("purge_trades", "HORIZON")
        embargo_trades = self.config.get("embargo_trades", "HORIZON")

        self.purge_trades = int(self.horizon_trades if purge_trades == "HORIZON" else purge_trades)
        self.embargo_trades = int(self.horizon_trades if embargo_trades == "HORIZON" else embargo_trades)

        # Safety
        self.train_trades = max(1, self.train_trades)
        self.val_trades = max(1, self.val_trades)
        self.test_trades = max(1, self.test_trades)
        self.purge_trades = max(0, self.purge_trades)
        self.embargo_trades = max(0, self.embargo_trades)

    def generate_folds(self, n_trades: int) -> Iterator[TradeFold]:
        n = int(n_trades)
        if n <= 0:
            return

        # total window per fold (train + purge + val + embargo + test + purge on each side)
        # NOTE: We keep TWO purge gaps: train->val and val->test.
        window = (
            self.train_trades +
            self.purge_trades +
            self.val_trades +
            self.embargo_trades +
            self.test_trades
        )

        if n < window + 10:
            # Not enough data
            return

        # Step forward in trades between folds
        if self.outer_folds <= 1:
            step = 0
        else:
            remaining = n - window
            step = max(1, remaining // (self.outer_folds - 1))

        for i in range(self.outer_folds):
            start = i * step
            train_start = start
            train_end = train_start + self.train_trades

            val_start = train_end + self.purge_trades
            val_end = val_start + self.val_trades

            test_start = val_end + self.embargo_trades
            test_end = test_start + self.test_trades

            if test_end > n:
                break

            yield TradeFold(
                fold_idx=i,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
