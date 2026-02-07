"""Backtest orchestration."""
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .execution import ExecutionConfig, ExecutionSimulator
from .portfolio import PortfolioTracker
from .positions import PositionConfig, PositionManager
from .signals import SignalConfig, SignalGenerator
from .stats import StatsCalculator


@dataclass
class BacktestConfig:
    """Master configuration for backtest."""

    signal: SignalConfig
    position: PositionConfig
    execution: ExecutionConfig
    initial_capital: float = 10000.0
    random_seed: int = 1337


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    trades_df: pd.DataFrame
    equity_curve: pd.DataFrame
    stats: dict
    config: BacktestConfig


# Minimum trade size to avoid dust trades
MIN_TRADE_SIZE = 1e-8


@dataclass
class PendingOrder:
    """Order pending execution on next bar."""

    decision_bar_idx: int  # Bar where decision was made
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    volatility: float


class BacktestRunner:
    """Orchestrate the full backtest.

    EXECUTION MODEL (NO LOOKAHEAD):
    - Decision at bar t: we observe close[t], generate signal
    - Fill at bar t+1: order executes at close[t+1] with slippage+fee
    - This ensures we never use same-bar close as fill price
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.signal_gen = SignalGenerator(config.signal)
        self.position_mgr = PositionManager(config.position)
        self.executor = ExecutionSimulator(config.execution, seed=config.random_seed)

    def run(self, predictions_df: pd.DataFrame) -> BacktestResult:
        """Run full backtest on predictions.

        IMPORTANT: No lookahead - decisions at bar t, fills at bar t+1.

        Args:
            predictions_df: DataFrame with forecast columns

        Returns:
            BacktestResult with trades, equity curve, and stats
        """
        # Set seed for reproducibility
        np.random.seed(self.config.random_seed)

        # Initialize portfolio
        portfolio = PortfolioTracker(
            initial_capital=self.config.initial_capital, seed=self.config.random_seed
        )

        # Sort chronologically by symbol, then timestamp (NO LOOKAHEAD)
        df = predictions_df.sort_values(["symbol", "timestamp_end"]).reset_index(
            drop=True
        )

        # Process each symbol independently
        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol].reset_index(drop=True)
            self._process_symbol(symbol_df, portfolio)

        # Compute results
        equity_curve = portfolio.get_equity_curve()
        trades_df = portfolio.get_trades()

        stats = StatsCalculator.compute(
            equity_curve=equity_curve,
            trades_df=trades_df,
            initial_capital=self.config.initial_capital,
        )

        return BacktestResult(
            trades_df=trades_df,
            equity_curve=equity_curve,
            stats=stats,
            config=self.config,
        )

    def _process_symbol(
        self, symbol_df: pd.DataFrame, portfolio: PortfolioTracker
    ) -> None:
        """Process bars for a single symbol with deferred fills.

        Execution model:
        - Decision at bar t: observe close[t], generate signal
        - Fill at bar t+1: execute at close[t+1] (next available bar in data)

        Args:
            symbol_df: DataFrame filtered to one symbol, sorted by time
            portfolio: Portfolio tracker to update
        """
        symbol = symbol_df["symbol"].iloc[0]
        pending_order: Optional[PendingOrder] = None
        prev_row_idx: int = -1  # Track DataFrame row index for next-bar validation

        for idx in range(len(symbol_df)):
            row = symbol_df.iloc[idx]
            current_bar_idx = row["bar_idx"]
            current_time = row["timestamp_end"]
            current_price = row["close"]
            volatility = row.get("sigma_hat", 0.0)
            if pd.isna(volatility):
                volatility = 0.0

            # STEP 1: Execute any pending order from previous bar
            if pending_order is not None:
                # Fill at CURRENT bar's close (this was decided on PREVIOUS bar)
                # Verify no same-bar fill and this is the next available bar in data
                # Note: bar_idx may have gaps due to walk-forward validation folds,
                # but we must fill on the next available bar (idx == prev_row_idx + 1)
                if current_bar_idx <= pending_order.decision_bar_idx:
                    raise AssertionError(
                        f"Fill must be after decision bar (no same-bar fill): "
                        f"decision_bar={pending_order.decision_bar_idx}, fill_bar={current_bar_idx}"
                    )
                if idx != prev_row_idx + 1:
                    # Cancel pending order if we skipped rows (shouldn't happen)
                    pending_order = None
                else:
                    fill = self.executor.execute(
                        timestamp=current_time,
                        symbol=symbol,
                        bar_idx=current_bar_idx,
                        side=pending_order.side,
                        quantity=pending_order.quantity,
                        mid_price=current_price,
                        volatility=pending_order.volatility,
                        decision_bar_idx=pending_order.decision_bar_idx,
                    )
                    portfolio.apply_fill(fill)
                    pending_order = None




            # STEP 2: Mark portfolio to market at current bar
            portfolio.mark_to_market(
                timestamp=current_time,
                symbol=symbol,
                bar_idx=current_bar_idx,
                price=current_price,
            )

            # STEP 3: Generate signal and decide on order for NEXT bar
            # Skip if this is the last bar (no next bar to fill at)
            if idx >= len(symbol_df) - 1:
                continue

            signal = self.signal_gen.generate_single(
                row, symbol=symbol, bar_idx=current_bar_idx
            )

            # Compute current equity and available cash for position sizing
            current_equity = portfolio.get_equity({symbol: current_price})
            available_cash = portfolio.cash

            # Get confidence for position sizing
            confidence = row.get("S_pmax", None)
            if pd.isna(confidence):
                confidence = None

            # Compute target position based on signal
            target_qty = self.position_mgr.compute_target_position(
                signal=signal,
                equity=current_equity,
                price=current_price,
                available_cash=available_cash,
                volatility=volatility,
                confidence=confidence,
            )

            # Current position for this symbol
            current_qty = portfolio.get_position(symbol)

            # Compute required trade
            trade_qty = target_qty - current_qty

            # Create pending order if trade needed (will execute on NEXT bar)
            if abs(trade_qty) > MIN_TRADE_SIZE:
                side = "BUY" if trade_qty > 0 else "SELL"
                qty = abs(trade_qty)
                # Cash-only (spot) clamp: prevent BUY from exceeding available cash (fees+spread+slip+buffer)
                if side == "BUY":
                    exec_cfg = self.executor.config
                    fee_rate = exec_cfg.fee_bps / 10000.0
                    buffer_bps = float(getattr(self.config.position, "cost_buffer_bps", 0.0))
                    all_in_bps = float(exec_cfg.spread_bps) + float(exec_cfg.slippage_bps) + max(0.0, buffer_bps)
                    est_price = float(current_price) * (1.0 + all_in_bps / 10000.0)
                    denom = est_price * (1.0 + fee_rate)
                    cash = float(portfolio.cash)
                    if (not isfinite(cash)) or cash <= 0 or (not isfinite(denom)) or denom <= 0:
                        continue
                    max_qty = cash / denom
                    if max_qty <= 0:
                        continue
                    qty = min(qty, max_qty)
                if qty <= MIN_TRADE_SIZE:
                    continue

                pending_order = PendingOrder(
                    decision_bar_idx=current_bar_idx,
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    volatility=volatility,
                )

            # Track row index for next-bar validation
            prev_row_idx = idx

    def save_results(self, result: BacktestResult, out_dir: Path) -> None:
        """Save backtest results to output directory.

        Args:
            result: BacktestResult to save
            out_dir: Output directory path
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save trades as parquet
        result.trades_df.to_parquet(out_dir / "trades.parquet", index=False)

        # Save equity curve as parquet
        result.equity_curve.to_parquet(out_dir / "equity_curve.parquet", index=False)

        # Save stats as JSON
        with open(out_dir / "stats.json", "w") as f:
            json.dump(result.stats, f, indent=2, default=str)


def load_config_from_yaml(yaml_path: Path) -> BacktestConfig:
    """Load backtest config from YAML file.

    Args:
        yaml_path: Path to backtest.yaml

    Returns:
        BacktestConfig object
    """
    import yaml

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    bt = raw.get("backtest", raw)

    signal_cfg = SignalConfig(**bt.get("signal", {}))
    position_cfg = PositionConfig(**bt.get("position", {}))
    execution_cfg = ExecutionConfig(**bt.get("execution", {}))

    return BacktestConfig(
        signal=signal_cfg,
        position=position_cfg,
        execution=execution_cfg,
        initial_capital=bt.get("initial_capital", 10000.0),
        random_seed=bt.get("random_seed", 1337),
    )
