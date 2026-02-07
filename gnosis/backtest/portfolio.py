"""Portfolio state tracking and accounting."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .execution import Fill


@dataclass
class PortfolioState:
    """Snapshot of portfolio state at a point in time."""

    timestamp: pd.Timestamp
    symbol: str
    bar_idx: int
    cash: float
    position: float  # Base currency units held
    price: float  # Current price for mark-to-market
    equity: float  # Total equity = cash + position * price


class PortfolioTracker:
    """Track portfolio state over time with per-symbol positions."""

    def __init__(self, initial_capital: float, seed: int = 1337):
        self.initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: Dict[str, float] = {}  # symbol -> quantity
        self._equity_history: List[PortfolioState] = []
        self._fills: List[Fill] = []
        self._trade_pnls: List[Dict] = []  # Track PnL per trade
        self._entry_prices: Dict[str, float] = {}  # symbol -> avg entry price

    @property
    def cash(self) -> float:
        return self._cash

    def get_position(self, symbol: str) -> float:
        """Get position size for a symbol."""
        return self._positions.get(symbol, 0.0)

    def get_equity(self, prices: Dict[str, float]) -> float:
        """Compute total equity given current prices.

        Args:
            prices: Dict of symbol -> current price

        Returns:
            Total equity (cash + sum of position values)
        """
        position_value = sum(
            qty * prices.get(sym, 0.0) for sym, qty in self._positions.items()
        )
        return self._cash + position_value

    def apply_fill(self, fill: Fill) -> float:
        """Update portfolio state from a fill.

        Args:
            fill: Fill object with trade details

        Returns:
            Realized PnL from this trade (0 if opening position)
        """
        current_pos = self._positions.get(fill.symbol, 0.0)
        entry_price = self._entry_prices.get(fill.symbol, 0.0)

        realized_pnl = 0.0

        actual_qty = fill.quantity  # Track actual quantity for trade log
        actual_notional = fill.notional  # Track actual notional for trade log

        if fill.side == "BUY":
            # Calculate total cost for this buy
            total_cost = fill.notional + fill.fee

            # Safety check: don't let cash go negative
            # If order would exceed cash, reduce quantity proportionally
            if total_cost > self._cash and self._cash > 0:
                # Scale down to fit available cash (with small safety margin)
                scale = (self._cash / total_cost) * 0.999
                actual_qty = fill.quantity * scale
                actual_notional = actual_qty * fill.price
                actual_fee = actual_notional * (fill.fee / fill.notional) if fill.notional > 0 else 0
                total_cost = actual_notional + actual_fee
            elif self._cash <= 0:
                # No cash available, skip this buy
                return 0.0
            else:
                actual_qty = fill.quantity
                actual_notional = fill.notional
                total_cost = fill.notional + fill.fee

            # Buying increases position
            new_pos = current_pos + actual_qty
            # Update average entry price
            if current_pos <= 0:
                # Opening new long position
                self._entry_prices[fill.symbol] = fill.price
            else:
                # Adding to existing long
                prev_cost = current_pos * entry_price
                self._entry_prices[fill.symbol] = (prev_cost + actual_notional) / new_pos if new_pos > 0 else fill.price
            self._positions[fill.symbol] = new_pos
            self._cash -= total_cost
        else:
            # Selling decreases position
            if current_pos > 0:
                # Closing or reducing long position
                # Clamp sell quantity to current position (no shorting in long-only mode)
                sell_qty = min(fill.quantity, current_pos)
                actual_qty = sell_qty
                realized_pnl = sell_qty * (fill.price - entry_price) - fill.fee
                new_pos = current_pos - sell_qty
                if new_pos <= 0:
                    # Position fully closed
                    self._entry_prices.pop(fill.symbol, None)
                self._positions[fill.symbol] = max(0.0, new_pos)
                # Cash is based on actual sell quantity, not requested quantity
                actual_notional = sell_qty * fill.price
                self._cash += actual_notional - fill.fee
            else:
                # No position to sell (shouldn't happen in long-only)
                actual_qty = 0.0
                # No cash change if nothing to sell

        self._fills.append(fill)
        self._trade_pnls.append(
            {
                "timestamp": fill.timestamp,
                "symbol": fill.symbol,
                "decision_bar_idx": fill.decision_bar_idx,  # Bar where decision was made
                "fill_bar_idx": fill.bar_idx,  # Bar where fill occurred
                "side": fill.side,
                "quantity": actual_qty,  # Actual quantity traded (clamped for SELL)
                "price": fill.price,
                "fee": fill.fee,
                "notional": actual_qty * fill.price,  # Actual notional
                "pnl": realized_pnl,
            }
        )

        return realized_pnl

    def mark_to_market(
        self, timestamp: pd.Timestamp, symbol: str, bar_idx: int, price: float
    ) -> PortfolioState:
        """Record portfolio state at a point in time.

        Args:
            timestamp: Current timestamp
            symbol: Symbol being marked
            bar_idx: Current bar index
            price: Current price for the symbol

        Returns:
            PortfolioState snapshot
        """
        position = self._positions.get(symbol, 0.0)
        position_value = position * price
        equity = self._cash + position_value

        state = PortfolioState(
            timestamp=timestamp,
            symbol=symbol,
            bar_idx=bar_idx,
            cash=self._cash,
            position=position,
            price=price,
            equity=equity,
        )
        self._equity_history.append(state)
        return state

    def get_equity_curve(self) -> pd.DataFrame:
        """Return time series of equity values.

        Returns:
            DataFrame with columns: timestamp, symbol, bar_idx, cash, position, price, equity
        """
        if not self._equity_history:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "bar_idx",
                    "cash",
                    "position",
                    "price",
                    "equity",
                ]
            )

        records = [
            {
                "timestamp": s.timestamp,
                "symbol": s.symbol,
                "bar_idx": s.bar_idx,
                "cash": s.cash,
                "position": s.position,
                "price": s.price,
                "equity": s.equity,
            }
            for s in self._equity_history
        ]
        return pd.DataFrame(records)

    def get_trades(self) -> pd.DataFrame:
        """Return trade log.

        Returns:
            DataFrame with columns for trade details including decision/fill bar indices
        """
        if not self._trade_pnls:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "decision_bar_idx",
                    "fill_bar_idx",
                    "side",
                    "quantity",
                    "price",
                    "fee",
                    "notional",
                    "pnl",
                ]
            )
        return pd.DataFrame(self._trade_pnls)
