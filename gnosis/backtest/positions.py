"""Position sizing and management with risk controls."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionConfig:
    """Configuration for position sizing."""

    mode: str = "fixed_pct"  # "fixed_pct", "notional_fixed", "vol_adjusted"
    equity_pct: float = 1.0  # % of equity per position (0-1)
    notional_usd: float = 1000.0  # Fixed notional (if mode=notional_fixed)
    long_only: bool = True  # No shorting (V1 requirement)
    cost_buffer_bps: float = 20.0  # Buffer for spread+fees to prevent over-allocation

    # Vol-adjusted sizing parameters
    target_risk_pct: float = 0.02  # Target risk per trade (2% of equity)
    vol_lookback: int = 20  # Bars for vol calculation
    max_position_pct: float = 1.0  # Maximum position as % of equity

    # Confidence-weighted sizing
    confidence_weight: bool = False  # Scale position by S_pmax


class PositionManager:
    """Manage position sizing and target positions with risk controls."""

    def __init__(self, config: PositionConfig):
        self.config = config
        self._median_vol: Optional[float] = None

    def set_median_vol(self, median_vol: float) -> None:
        """Set median volatility for vol-adjusted sizing.

        Args:
            median_vol: Median realized volatility from training data
        """
        self._median_vol = median_vol

    def compute_target_position(
        self,
        signal: int,
        equity: float,
        price: float,
        available_cash: float = None,
        volatility: float = None,
        confidence: float = None,
    ) -> float:
        """Compute target position in base currency units.

        Args:
            signal: 1 for LONG, 0 for FLAT
            equity: Current portfolio equity
            price: Current price
            available_cash: Available cash for buying (optional, defaults to equity)
            volatility: Current sigma_hat for vol-adjusted sizing
            confidence: S_pmax for confidence-weighted sizing

        Returns:
            Target position size in base currency units
        """
        if signal == 0:
            return 0.0

        if self.config.long_only and signal < 0:
            return 0.0

        if price <= 0:
            return 0.0

        # === COMPUTE BASE POSITION SIZE ===

        if self.config.mode == "vol_adjusted" and volatility is not None and volatility > 0:
            # Vol-adjusted: size inversely proportional to volatility
            # Position = (equity × target_risk) / (price × volatility)
            target_notional = self._compute_vol_adjusted_notional(
                equity, price, volatility
            )

        elif self.config.mode == "fixed_pct":
            target_notional = equity * self.config.equity_pct

        elif self.config.mode == "notional_fixed":
            target_notional = self.config.notional_usd

        else:
            target_notional = equity * self.config.equity_pct

        # === APPLY CONFIDENCE WEIGHTING ===

        if self.config.confidence_weight and confidence is not None:
            # Scale position by confidence (0.5 to 1.0 range for S_pmax)
            # Confidence of 0.5 (random) -> 50% position
            # Confidence of 1.0 (certain) -> 100% position
            conf_factor = max(0.5, min(1.0, confidence))
            target_notional *= conf_factor

        # === APPLY POSITION LIMITS ===

        # Cap at maximum position percentage
        max_notional = equity * self.config.max_position_pct
        target_notional = min(target_notional, max_notional)

        # Apply cost buffer to prevent over-allocation
        # Buffer accounts for spread + fees that will be deducted on execution
        cost_factor = 1 + self.config.cost_buffer_bps / 10000

        # Cap at available cash if provided (apply buffer to cash, not to both)
        if available_cash is not None and available_cash > 0:
            max_from_cash = available_cash / cost_factor
            target_notional = min(target_notional, max_from_cash)

        return target_notional / price

    def _compute_vol_adjusted_notional(
        self,
        equity: float,
        price: float,
        volatility: float,
    ) -> float:
        """Compute notional using vol-adjusted sizing.

        Position size is inversely proportional to volatility.
        Higher vol -> smaller position to maintain constant risk.

        Args:
            equity: Current portfolio equity
            price: Current price
            volatility: Current realized vol (sigma_hat)

        Returns:
            Target notional value
        """
        # Risk budget = equity × target_risk_pct
        risk_budget = equity * self.config.target_risk_pct

        # Position value = risk_budget / volatility
        # This ensures expected loss at 1σ = risk_budget
        if volatility > 0:
            target_notional = risk_budget / volatility
        else:
            # Fall back to fixed percentage if no volatility
            target_notional = equity * self.config.equity_pct

        # Optionally scale by median vol ratio
        if self._median_vol is not None and self._median_vol > 0:
            # If current vol > median, reduce position further
            vol_ratio = self._median_vol / volatility
            target_notional *= min(vol_ratio, 2.0)  # Cap at 2x

        return target_notional

