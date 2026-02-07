"""Trade execution simulation with fees and slippage."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ExecutionConfig:
    """Configuration for execution simulation."""

    fee_bps: float = 7.5  # Taker fee in basis points
    spread_bps: float = 2.0  # Half-spread (bid-ask)
    slippage_bps: float = 0.0  # Optional fixed slippage
    slippage_model: str = "fixed"  # "fixed" or "vol_proportional"
    slippage_k: float = 0.35  # Vol proportional factor


@dataclass
class Fill:
    """Represents a single trade fill."""

    timestamp: pd.Timestamp
    symbol: str
    bar_idx: int  # Bar where fill occurred
    side: str  # "BUY" or "SELL"
    quantity: float  # Base currency units
    price: float  # Execution price (after spread/slippage)
    fee: float  # Fee in quote currency (USD)
    notional: float  # Total notional in quote currency
    decision_bar_idx: Optional[int] = None  # Bar where decision was made (for no-lookahead verification)


class ExecutionSimulator:
    """Simulate trade execution with fees and slippage."""

    def __init__(self, config: ExecutionConfig, seed: int = 1337):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def compute_execution_price(
        self, side: str, mid_price: float, volatility: float = 0.0
    ) -> float:
        """Compute execution price after spread and slippage.

        Args:
            side: "BUY" or "SELL"
            mid_price: Mid/close price
            volatility: Optional volatility (sigma_hat) for vol-proportional slippage

        Returns:
            Execution price
        """
        # Apply spread (always pay the spread)
        spread_cost = mid_price * (self.config.spread_bps / 10000)
        if side == "BUY":
            price = mid_price + spread_cost  # Buy at ask
        else:
            price = mid_price - spread_cost  # Sell at bid

        # Apply slippage
        if self.config.slippage_model == "fixed":
            slippage = mid_price * (self.config.slippage_bps / 10000)
        elif self.config.slippage_model == "vol_proportional":
            slippage = mid_price * self.config.slippage_k * volatility
        else:
            slippage = 0.0

        if side == "BUY":
            price += slippage  # Worse price for buyer
        else:
            price -= slippage  # Worse price for seller

        return price

    def compute_fee(self, notional: float) -> float:
        """Compute fee for a given notional amount.

        Args:
            notional: Trade notional in quote currency

        Returns:
            Fee amount in quote currency
        """
        return notional * (self.config.fee_bps / 10000)

    def execute(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        bar_idx: int,
        side: str,
        quantity: float,
        mid_price: float,
        volatility: float = 0.0,
        decision_bar_idx: Optional[int] = None,
    ) -> Fill:
        """Simulate trade execution.

        Args:
            timestamp: Execution timestamp
            symbol: Trading symbol
            bar_idx: Bar index where fill occurs
            side: "BUY" or "SELL"
            quantity: Amount in base currency
            mid_price: Mid/close price
            volatility: Optional volatility for slippage
            decision_bar_idx: Bar index where decision was made (for no-lookahead verification)

        Returns:
            Fill object with execution details
        """
        exec_price = self.compute_execution_price(side, mid_price, volatility)
        notional = quantity * exec_price
        fee = self.compute_fee(notional)

        return Fill(
            timestamp=timestamp,
            symbol=symbol,
            bar_idx=bar_idx,
            side=side,
            quantity=quantity,
            price=exec_price,
            fee=fee,
            notional=notional,
            decision_bar_idx=decision_bar_idx,
        )
