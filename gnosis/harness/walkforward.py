"""Walk-forward validation harness with purge/embargo."""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Fold:
    """Represents a single walk-forward fold."""
    fold_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


class WalkForwardHarness:
    """Nested walk-forward with purge and embargo."""

    def __init__(self, config: dict):
        self.outer_folds = config.get("outer_folds", 8)
        self.train_days = config.get("train_days", 180)
        self.val_days = config.get("val_days", 30)
        self.test_days = config.get("test_days", 30)
        self.purge_trades = config.get("purge_trades", "HORIZON")
        self.embargo_trades = config.get("embargo_trades", "HORIZON")

    def generate_folds(self, df: pd.DataFrame, horizon_trades: int = 2000) -> Iterator[Fold]:
        """Generate walk-forward folds with purge/embargo.

        Uses proportional allocation based on configured day ratios.
        """
        n_bars = len(df)
        if n_bars < 50:
            return  # Not enough data for any folds

        total_days = self.train_days + self.val_days + self.test_days

        # Calculate bar proportions based on day ratios
        train_ratio = self.train_days / total_days
        val_ratio = self.val_days / total_days
        test_ratio = self.test_days / total_days

        # Purge/embargo in bars (based on horizon)
        purge_bars = max(1, horizon_trades // 200)  # D0 has 200 trades per bar
        embargo_bars = purge_bars

        # Total bars needed for one fold (train + val + test + gaps)
        gap_bars = purge_bars + embargo_bars

        # Reserve space for gaps and compute usable bars per fold
        # We need: outer_folds * (train + val + test) + gaps between them
        bars_per_fold = (n_bars - gap_bars * self.outer_folds) // self.outer_folds
        if bars_per_fold < 30:
            # Fall back to simpler approach: use all data with proportional splits
            bars_per_fold = n_bars // max(2, self.outer_folds)

        train_bars = max(10, int(bars_per_fold * train_ratio))
        val_bars = max(5, int(bars_per_fold * val_ratio))
        test_bars = max(5, int(bars_per_fold * test_ratio))

        # Step size between folds
        step = (n_bars - train_bars - val_bars - test_bars - gap_bars) // max(1, self.outer_folds - 1)
        step = max(1, step)

        for i in range(self.outer_folds):
            train_start = i * step
            train_end = train_start + train_bars

            # Purge gap between train and val
            val_start = train_end + purge_bars
            val_end = val_start + val_bars

            # Embargo gap between val and test
            test_start = val_end + embargo_bars
            test_end = min(test_start + test_bars, n_bars)

            if test_end <= test_start or val_end > n_bars:
                break

            yield Fold(
                fold_idx=i,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )


def compute_future_returns(df: pd.DataFrame, horizon_bars: int = 10) -> pd.DataFrame:
    """Compute future returns for prediction targets."""
    result = df.copy()
    result["future_return"] = result.groupby("symbol")["close"].transform(
        lambda x: x.shift(-horizon_bars) / x - 1
    )
    return result
