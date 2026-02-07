"""Signal generation from forecasts with trade filters."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SignalConfig:
    """Configuration for signal generation."""

    mode: str = "x_hat_threshold"  # "x_hat_threshold", "quantile_skew"
    long_threshold: float = 0.0  # Go long when x_hat > threshold
    use_abstain: bool = True  # Respect abstain flag from predictions
    min_confidence: float = 0.0  # Min S_pmax to take position

    # Trade frequency filters
    min_bars_between_trades: int = 0  # Minimum bars between entries (0 = no limit)
    min_signal_strength: float = 0.0  # Minimum |x_hat| to trade

    # Risk filters
    max_volatility: float = 0.0  # Max sigma_hat to trade (0 = no limit)
    min_reward_risk: float = 0.0  # Min expected return / sigma_hat (0 = no limit)


class SignalGenerator:
    """Convert forecasts into trading signals (LONG=1, FLAT=0).

    Includes filters to reduce trade frequency and improve quality.
    """

    def __init__(self, config: SignalConfig):
        self.config = config
        self._last_trade_bar: dict = {}  # symbol -> last bar_idx with trade

    def reset(self) -> None:
        """Reset state (for new backtest runs)."""
        self._last_trade_bar = {}

    def generate_single(
        self,
        row: pd.Series,
        symbol: Optional[str] = None,
        bar_idx: Optional[int] = None
    ) -> int:
        """Generate signal for a single prediction row.

        Args:
            row: Series with x_hat, abstain, and optionally S_pmax
            symbol: Symbol for cooldown tracking (optional)
            bar_idx: Current bar index for cooldown tracking (optional)

        Returns:
            1 for LONG, 0 for FLAT
        """
        # Get symbol and bar_idx from row if not provided
        if symbol is None:
            symbol = row.get("symbol", "UNKNOWN")
        if bar_idx is None:
            bar_idx = row.get("bar_idx", 0)

        # === FILTER 1: Respect abstain flag ===
        if self.config.use_abstain and row.get("abstain", False):
            return 0

        # === FILTER 2: Confidence threshold ===
        if "S_pmax" in row and self.config.min_confidence > 0:
            s_pmax = row["S_pmax"]
            if pd.isna(s_pmax) or s_pmax < self.config.min_confidence:
                return 0

        # === FILTER 3: Minimum bar separation (cooldown) ===
        if self.config.min_bars_between_trades > 0:
            last_bar = self._last_trade_bar.get(symbol, -999)
            bars_since_trade = bar_idx - last_bar
            if bars_since_trade < self.config.min_bars_between_trades:
                return 0

        # === FILTER 4: Signal strength threshold ===
        x_hat = row.get("x_hat", 0.0)
        if pd.isna(x_hat):
            x_hat = 0.0

        if self.config.min_signal_strength > 0:
            if abs(x_hat) < self.config.min_signal_strength:
                return 0

        # === FILTER 5: Maximum volatility ===
        if self.config.max_volatility > 0:
            sigma_hat = row.get("sigma_hat", 0.0)
            if pd.notna(sigma_hat) and sigma_hat > self.config.max_volatility:
                return 0

        # === FILTER 6: Reward/Risk ratio ===
        if self.config.min_reward_risk > 0:
            sigma_hat = row.get("sigma_hat", 0.0)
            if pd.notna(sigma_hat) and sigma_hat > 0:
                reward_risk = abs(x_hat) / sigma_hat
                if reward_risk < self.config.min_reward_risk:
                    return 0

        # === SIGNAL GENERATION ===
        signal = 0

        if self.config.mode == "x_hat_threshold":
            signal = 1 if x_hat > self.config.long_threshold else 0

        elif self.config.mode == "quantile_skew":
            q05 = row.get("q05", 0.0)
            q50 = row.get("q50", 0.0)
            q95 = row.get("q95", 0.0)
            if pd.isna(q05) or pd.isna(q50) or pd.isna(q95):
                signal = 0
            else:
                upside = q95 - q50
                downside = q50 - q05
                # Go long if more upside than downside
                if downside > 0 and upside > downside:
                    signal = 1

        # Update cooldown tracker if we're generating a trade
        if signal != 0 and self.config.min_bars_between_trades > 0:
            self._last_trade_bar[symbol] = bar_idx

        return signal

    def generate(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Add 'signal' column to predictions DataFrame (vectorized).

        Args:
            predictions_df: DataFrame with forecast columns

        Returns:
            Copy of DataFrame with 'signal' column added (1=LONG, 0=FLAT)
        """
        self.reset()  # Clear cooldown state

        df = predictions_df.copy()
        signals = []

        for idx, row in df.iterrows():
            symbol = row.get("symbol", "UNKNOWN")
            bar_idx = row.get("bar_idx", idx)
            sig = self.generate_single(row, symbol=symbol, bar_idx=bar_idx)
            signals.append(sig)
        n = len(df)

        # Start with all signals as 0 (FLAT)
        signals = np.zeros(n, dtype=int)

        # Build mask of rows that pass the abstain filter
        if self.config.use_abstain and "abstain" in df.columns:
            abstain_mask = df["abstain"].fillna(False).astype(bool).values
        else:
            abstain_mask = np.zeros(n, dtype=bool)

        # Build mask of rows that pass the confidence filter
        if "S_pmax" in df.columns and self.config.min_confidence > 0:
            conf_mask = df["S_pmax"].fillna(0).values < self.config.min_confidence
        else:
            conf_mask = np.zeros(n, dtype=bool)

        # Combine filters: rows that are NOT filtered out
        valid_mask = ~abstain_mask & ~conf_mask

        if self.config.mode == "x_hat_threshold":
            # Get x_hat values, replacing NaN with threshold (will result in no signal)
            x_hat = df["x_hat"].fillna(self.config.long_threshold).values
            # Signal is 1 where x_hat > threshold AND row is valid
            signals = np.where(valid_mask & (x_hat > self.config.long_threshold), 1, 0)

        elif self.config.mode == "quantile_skew":
            q05 = df.get("q05", pd.Series(0.0, index=df.index)).fillna(0).values
            q50 = df.get("q50", pd.Series(0.0, index=df.index)).fillna(0).values
            q95 = df.get("q95", pd.Series(0.0, index=df.index)).fillna(0).values

            upside = q95 - q50
            downside = q50 - q05

            # Go long if more upside than downside (and downside > 0)
            skew_condition = (downside > 0) & (upside > downside)
            signals = np.where(valid_mask & skew_condition, 1, 0)

        df["signal"] = signals
        return df
