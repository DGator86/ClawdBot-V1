"""Quantum-Inspired Bitcoin Price Prediction Engine.

Treats price as a particle moving through steering fields:
- Funding rate fields (restoring force)
- Liquidation cascade fields (potential wells with avalanche dynamics)
- Order book gradient fields (gravitational attraction to liquidity)
- Cross-asset coupling fields (external forces)
- Gamma fields (options market maker hedging)
- Time-of-day effects (intraday volatility patterns)

Implements regime-switching dynamics where physics parameters
vary based on market state (trending, ranging, volatile).

Outputs probabilistic forecasts with confidence intervals.

All steering field parameters are exposed as hyperparameters for ML tuning.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class RegimeParameters:
    """Physics parameters for each market regime."""
    funding_strength: float
    imbalance_strength: float
    momentum_decay: float
    volatility_base: float
    jump_intensity: float
    mean_reversion: float
    drag_coefficient: float


@dataclass
class SteeringForces:
    """All forces acting on the price particle."""
    funding_force: float = 0.0
    imbalance_force: float = 0.0
    gravity_force: float = 0.0
    spring_force: float = 0.0
    depth_pressure: float = 0.0
    momentum_force: float = 0.0
    liquidation_force: float = 0.0
    friction_force: float = 0.0

    @property
    def total_drift(self) -> float:
        """Net drift from all forces."""
        return (
            self.funding_force +
            self.imbalance_force +
            self.gravity_force +
            self.spring_force +
            self.depth_pressure +
            self.momentum_force +
            self.liquidation_force +
            self.friction_force
        )


@dataclass
class PredictionResult:
    """Complete prediction with confidence intervals."""
    current_price: float
    regime: MarketRegime
    point_estimate: float
    expected_return_pct: float
    probability_up: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    forces: SteeringForces
    simulation_stats: Dict[str, float]


class QuantumPriceEngine:
    """Advanced physics engine with regime-switching and steering fields.

    Implements:
    1. Regime detection (trending/ranging/volatile)
    2. Multi-scale force aggregation
    3. Order book gravity model
    4. VWAP spring forces (mean reversion)
    5. Jump-diffusion for liquidation cascades
    6. Stochastic volatility
    7. Monte Carlo with confidence intervals
    """

    # Regime-specific physics parameters (empirically calibrated)
    REGIME_PARAMS = {
        MarketRegime.TRENDING: RegimeParameters(
            funding_strength=12.0,
            imbalance_strength=0.06,
            momentum_decay=0.7,
            volatility_base=0.012,
            jump_intensity=0.02,
            mean_reversion=0.15,
            drag_coefficient=0.15
        ),
        MarketRegime.RANGING: RegimeParameters(
            funding_strength=18.0,
            imbalance_strength=0.10,
            momentum_decay=0.3,
            volatility_base=0.018,
            jump_intensity=0.01,
            mean_reversion=0.35,
            drag_coefficient=0.25
        ),
        MarketRegime.VOLATILE: RegimeParameters(
            funding_strength=8.0,
            imbalance_strength=0.04,
            momentum_decay=0.4,
            volatility_base=0.025,
            jump_intensity=0.08,
            mean_reversion=0.10,
            drag_coefficient=0.10
        ),
    }

    # Multi-scale timeframe weights
    TIMEFRAME_WEIGHTS = {
        "1m": 0.4,   # Microstructure effects
        "5m": 0.3,   # Short-term momentum
        "15m": 0.2,  # Medium-term trends
        "1h": 0.1,   # Macro direction
    }

    # Physics constants
    GRAVITY_G = 1e-8          # Order book gravitational constant (reduced)
    SPRING_K = 0.5            # VWAP mean reversion strength (much weaker)
    JUMP_MAGNITUDE = 0.015    # Average jump size (1.5%)
    VOLATILITY_FLOOR = 0.02   # Minimum hourly volatility (2%)

    def __init__(
        self,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None,
    ):
        """Initialize quantum price engine.

        Args:
            n_simulations: Number of Monte Carlo paths
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    def detect_regime(
        self,
        df: pd.DataFrame,
        lookback: int = 120,
    ) -> MarketRegime:
        """Detect current market regime from price action.

        Uses multiple indicators:
        - Trend strength (directional consistency)
        - Volatility level
        - Price range compression
        - Momentum consistency

        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to analyze

        Returns:
            MarketRegime classification
        """
        if len(df) < lookback:
            lookback = len(df)

        recent = df.tail(lookback)

        # Calculate returns
        if "returns" in recent.columns:
            returns = recent["returns"].dropna()
        else:
            returns = recent["close"].pct_change().dropna()

        if len(returns) < 10:
            return MarketRegime.RANGING

        # 1. Trend strength (proportion of same-direction moves)
        up_moves = (returns > 0).sum()
        down_moves = (returns < 0).sum()
        total_moves = up_moves + down_moves
        trend_strength = abs(up_moves - down_moves) / total_moves if total_moves > 0 else 0

        # 2. Volatility level
        volatility = returns.std() * np.sqrt(60)  # Annualized to hourly

        # 3. Price range compression
        high_low_range = (recent["high"].max() - recent["low"].min()) / recent["close"].mean()

        # 4. Momentum consistency
        rolling_returns = returns.rolling(10, min_periods=5).mean()
        momentum_consistency = (rolling_returns > 0).sum() / len(rolling_returns) if len(rolling_returns) > 0 else 0.5
        momentum_bias = abs(momentum_consistency - 0.5) * 2

        # Regime classification logic
        if trend_strength > 0.6 and momentum_bias > 0.4 and volatility < 0.025:
            return MarketRegime.TRENDING
        elif volatility > 0.04 or high_low_range > 0.08:
            return MarketRegime.VOLATILE
        else:
            return MarketRegime.RANGING

    def calculate_funding_force(
        self,
        funding_rate: float,
        params: RegimeParameters,
    ) -> float:
        """Calculate restoring force from funding rate.

        Funding rate creates pressure toward equilibrium:
        - Positive funding (longs pay shorts) -> downward pressure
        - Negative funding (shorts pay longs) -> upward pressure

        F_funding = -funding_rate * strength
        """
        return -funding_rate * params.funding_strength

    def calculate_order_book_imbalance(
        self,
        bid_volume: float,
        ask_volume: float,
        params: RegimeParameters,
    ) -> float:
        """Calculate force from order book imbalance.

        Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        More bids than asks -> upward pressure
        """
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0

        imbalance = (bid_volume - ask_volume) / total
        return imbalance * params.imbalance_strength

    def calculate_gravity_force(
        self,
        current_price: float,
        support_levels: List[Tuple[float, float]],  # (price, volume)
        resistance_levels: List[Tuple[float, float]],
    ) -> float:
        """Calculate gravitational attraction to large orders.

        Large orders act as gravitational masses:
        F = G * m1 * m2 / r^2

        Support attracts price up, resistance attracts down.
        """
        net_force = 0.0

        # Support (bid) gravity - attracts price downward toward support
        for price, volume in support_levels:
            distance = abs(current_price - price) / current_price
            if distance > 0.0001:  # Avoid division by zero
                force = self.GRAVITY_G * volume / (distance ** 2)
                if price < current_price:
                    net_force -= force  # Pulls down toward support
                else:
                    net_force += force

        # Resistance (ask) gravity - attracts price upward toward resistance
        for price, volume in resistance_levels:
            distance = abs(current_price - price) / current_price
            if distance > 0.0001:
                force = self.GRAVITY_G * volume / (distance ** 2)
                if price > current_price:
                    net_force += force  # Pulls up toward resistance
                else:
                    net_force -= force

        return np.tanh(net_force) * 0.01  # Clamp to prevent explosions

    def calculate_spring_force(
        self,
        current_price: float,
        equilibrium_price: float,  # VWAP or SMA
        params: RegimeParameters,
    ) -> float:
        """Calculate spring-like restoring force toward equilibrium.

        Hooke's Law: F = -k * x

        Price tends to revert toward VWAP/moving average.
        Force is capped to prevent extreme predictions.
        """
        displacement = (current_price - equilibrium_price) / equilibrium_price
        # Cap displacement effect to prevent extreme forces
        displacement = np.clip(displacement, -0.05, 0.05)
        force = -self.SPRING_K * displacement * params.mean_reversion
        # Additional cap on force magnitude
        return np.clip(force, -0.01, 0.01)

    def calculate_liquidation_force(
        self,
        current_price: float,
        liquidation_levels: List[Tuple[float, float]],  # (price, volume)
    ) -> float:
        """Calculate force from liquidation cascade potential.

        Liquidation clusters create:
        - Repulsion when price approaches (market makers defend)
        - Attraction/acceleration once breached (cascade effect)
        """
        net_force = 0.0

        for liq_price, liq_volume in liquidation_levels:
            distance_pct = (current_price - liq_price) / current_price

            # Close to liquidation level (within 1%)
            if abs(distance_pct) < 0.01:
                # Repulsion zone - market makers defend
                repulsion = -np.sign(distance_pct) * liq_volume * 1e-6 / (abs(distance_pct) + 0.001)
                net_force += repulsion

            # Very close (within 0.2%) - cascade acceleration
            elif abs(distance_pct) < 0.002:
                # Attraction - cascade pulls price through
                attraction = np.sign(distance_pct) * liq_volume * 1e-5
                net_force += attraction

        return np.clip(net_force, -0.01, 0.01)

    def calculate_momentum_force(
        self,
        returns_multi_scale: Dict[str, float],
        params: RegimeParameters,
    ) -> float:
        """Calculate momentum force aggregated across timeframes.

        F_momentum = sum(w_i * momentum_i) * decay
        """
        total_momentum = 0.0

        for timeframe, weight in self.TIMEFRAME_WEIGHTS.items():
            momentum = returns_multi_scale.get(timeframe, 0.0)
            total_momentum += weight * momentum

        return total_momentum * params.momentum_decay

    def forecast_volatility(
        self,
        returns: pd.Series,
        current_vol: float,
        params: RegimeParameters,
    ) -> float:
        """Forecast volatility using GARCH-like dynamics.

        σ²(t+1) = ω + α * r²(t) + β * σ²(t)
        """
        # GARCH(1,1) parameters
        omega = 0.00001
        alpha = 0.15
        beta = 0.80

        if len(returns) < 2:
            return params.volatility_base

        lagged_return = returns.iloc[-1] if not pd.isna(returns.iloc[-1]) else 0
        lagged_return_squared = lagged_return ** 2

        next_variance = omega + alpha * lagged_return_squared + beta * (current_vol ** 2)
        next_volatility = np.sqrt(max(next_variance, 0.0001))

        return next_volatility

    def simulate_paths(
        self,
        current_price: float,
        forces: SteeringForces,
        volatility: float,
        params: RegimeParameters,
        n_steps: int,
        dt: float = 1/60,  # 1 minute
    ) -> np.ndarray:
        """Run Monte Carlo simulation with jump-diffusion.

        dP = μ(F) * P * dt + σ * P * dW + J * P * dN

        where:
        - μ(F) is drift from steering forces
        - σ is stochastic volatility
        - dW is Brownian motion
        - J is jump size
        - dN is Poisson process

        Returns:
            Array of final prices shape (n_simulations,)
        """
        n_sims = self.n_simulations

        # Apply drag force to total drift
        net_drift = forces.total_drift
        drag = -params.drag_coefficient * net_drift
        final_drift = net_drift + drag

        # Initialize paths
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = current_price

        # Pre-generate random numbers
        brownian_shocks = np.random.normal(0, 1, (n_sims, n_steps))
        jump_times = np.random.poisson(params.jump_intensity * dt, (n_sims, n_steps))
        jump_sizes = np.random.normal(0, self.JUMP_MAGNITUDE, (n_sims, n_steps))

        # Simulate paths
        for step in range(n_steps):
            # Geometric Brownian Motion
            diffusion = (
                (final_drift - 0.5 * volatility**2) * dt +
                volatility * np.sqrt(dt) * brownian_shocks[:, step]
            )

            # Jump component (Poisson compound process)
            jumps = jump_times[:, step] * jump_sizes[:, step]

            # Total log return
            total_log_return = diffusion + jumps

            # Update prices
            paths[:, step + 1] = paths[:, step] * np.exp(total_log_return)

            # Cap extreme movements (15% max per minute)
            max_move = 0.15
            price_change = (paths[:, step + 1] - paths[:, step]) / paths[:, step]
            extreme_mask = np.abs(price_change) > max_move
            if np.any(extreme_mask):
                capped = np.sign(price_change[extreme_mask]) * max_move
                paths[extreme_mask, step + 1] = paths[extreme_mask, step] * (1 + capped)

        return paths[:, -1]

    def predict(
        self,
        df: pd.DataFrame,
        horizon_minutes: int = 60,
        align_to_hour_end: bool = False,
        funding_rate: float = 0.0,
        bid_volume: float = 0.0,
        ask_volume: float = 0.0,
        vwap: Optional[float] = None,
        support_levels: Optional[List[Tuple[float, float]]] = None,
        resistance_levels: Optional[List[Tuple[float, float]]] = None,
        liquidation_levels: Optional[List[Tuple[float, float]]] = None,
    ) -> PredictionResult:
        """Generate probabilistic price prediction.

        Args:
            df: DataFrame with OHLCV data
            horizon_minutes: Prediction horizon in minutes
            align_to_hour_end: Cap horizon to minutes remaining in the current hour
            funding_rate: Current funding rate (decimal, e.g., 0.0001 = 0.01%)
            bid_volume: Total bid volume in order book
            ask_volume: Total ask volume in order book
            vwap: Volume-weighted average price (for mean reversion)
            support_levels: List of (price, volume) support levels
            resistance_levels: List of (price, volume) resistance levels
            liquidation_levels: List of (price, volume) liquidation clusters

        Returns:
            PredictionResult with point estimate and confidence intervals
        """
        horizon_minutes = self.resolve_horizon_minutes(
            df,
            horizon_minutes=horizon_minutes,
            align_to_hour_end=align_to_hour_end,
        )
        current_price = float(df["close"].iloc[-1])

        # Detect regime
        regime = self.detect_regime(df)
        params = self.REGIME_PARAMS[regime]

        # Calculate equilibrium price (VWAP or SMA)
        if vwap is None:
            if "volume" in df.columns:
                vwap = (df["close"] * df["volume"]).tail(50).sum() / df["volume"].tail(50).sum()
            else:
                vwap = df["close"].tail(50).mean()

        # Calculate all forces
        forces = SteeringForces()

        # Funding force
        forces.funding_force = self.calculate_funding_force(funding_rate, params)

        # Order book imbalance
        forces.imbalance_force = self.calculate_order_book_imbalance(
            bid_volume, ask_volume, params
        )

        # Gravity from support/resistance
        if support_levels and resistance_levels:
            forces.gravity_force = self.calculate_gravity_force(
                current_price, support_levels, resistance_levels
            )

        # Spring force toward VWAP
        forces.spring_force = self.calculate_spring_force(
            current_price, vwap, params
        )

        # Liquidation cascade force
        if liquidation_levels:
            forces.liquidation_force = self.calculate_liquidation_force(
                current_price, liquidation_levels
            )

        # Multi-scale momentum
        returns = df["close"].pct_change() if "returns" not in df.columns else df["returns"]
        momentum_scales = {
            "1m": returns.tail(5).mean() if len(returns) >= 5 else 0,
            "5m": returns.tail(15).mean() if len(returns) >= 15 else 0,
            "15m": returns.tail(45).mean() if len(returns) >= 45 else 0,
            "1h": returns.tail(60).mean() if len(returns) >= 60 else 0,
        }
        forces.momentum_force = self.calculate_momentum_force(momentum_scales, params)

        # Friction (opposes motion)
        forces.friction_force = -params.drag_coefficient * forces.total_drift * 0.1

        # Forecast volatility with floor
        current_vol = returns.tail(60).std() if len(returns) >= 60 else params.volatility_base
        volatility = self.forecast_volatility(returns, current_vol, params)
        # Apply volatility floor for realistic intervals
        volatility = max(volatility, self.VOLATILITY_FLOOR) * np.sqrt(horizon_minutes / 60)

        # Run Monte Carlo simulation
        final_prices = self.simulate_paths(
            current_price=current_price,
            forces=forces,
            volatility=volatility,
            params=params,
            n_steps=horizon_minutes,
        )

        # Calculate statistics
        point_estimate = float(np.median(final_prices))
        mean_estimate = float(np.mean(final_prices))
        probability_up = float(np.mean(final_prices > current_price))

        confidence_intervals = {
            "50%": (float(np.percentile(final_prices, 25)), float(np.percentile(final_prices, 75))),
            "68%": (float(np.percentile(final_prices, 16)), float(np.percentile(final_prices, 84))),
            "80%": (float(np.percentile(final_prices, 10)), float(np.percentile(final_prices, 90))),
            "90%": (float(np.percentile(final_prices, 5)), float(np.percentile(final_prices, 95))),
            "95%": (float(np.percentile(final_prices, 2.5)), float(np.percentile(final_prices, 97.5))),
            "99%": (float(np.percentile(final_prices, 0.5)), float(np.percentile(final_prices, 99.5))),
        }

        simulation_stats = {
            "mean": mean_estimate,
            "std": float(np.std(final_prices)),
            "skew": float(pd.Series(final_prices).skew()),
            "kurtosis": float(pd.Series(final_prices).kurtosis()),
            "min": float(np.min(final_prices)),
            "max": float(np.max(final_prices)),
        }

        return PredictionResult(
            current_price=current_price,
            regime=regime,
            point_estimate=point_estimate,
            expected_return_pct=(point_estimate / current_price - 1) * 100,
            probability_up=probability_up,
            confidence_intervals=confidence_intervals,
            forces=forces,
            simulation_stats=simulation_stats,
        )

    def resolve_horizon_minutes(
        self,
        df: pd.DataFrame,
        horizon_minutes: int,
        align_to_hour_end: bool = False,
    ) -> int:
        """Resolve the effective horizon in minutes."""
        minutes = int(horizon_minutes)
        if align_to_hour_end:
            timestamp = (
                df["timestamp_end"].iloc[-1]
                if "timestamp_end" in df.columns and len(df) > 0
                else pd.Timestamp.utcnow()
            )
            minutes_to_hour_end = self._minutes_to_hour_end(pd.Timestamp(timestamp))
            minutes = min(minutes, minutes_to_hour_end)
        return max(1, minutes)

    @staticmethod
    def _minutes_to_hour_end(timestamp: pd.Timestamp) -> int:
        """Minutes until the next hour boundary (minimum 1)."""
        next_hour = (timestamp + pd.Timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0
        )
        remaining_seconds = (next_hour - timestamp).total_seconds()
        return max(1, int(np.ceil(remaining_seconds / 60)))

    def generate_report(self, result: PredictionResult, horizon_minutes: int = 60) -> str:
        """Generate human-readable prediction report."""
        lines = []
        lines.append("=" * 70)
        lines.append("YOSHI QUANTUM PRICE PREDICTION")
        lines.append("=" * 70)
        lines.append("")

        # Current state
        lines.append(f"Current Price:     ${result.current_price:,.2f}")
        lines.append(f"Market Regime:     {result.regime.value.upper()}")
        lines.append(f"Horizon:           {horizon_minutes} minutes")
        lines.append("")

        # Point estimate
        lines.append(f"Point Estimate:    ${result.point_estimate:,.2f}")
        lines.append(f"Expected Return:   {result.expected_return_pct:+.2f}%")
        lines.append(f"Probability Up:    {result.probability_up:.1%}")
        lines.append("")

        # Confidence intervals
        lines.append("CONFIDENCE INTERVALS:")
        lines.append("-" * 40)
        for level, (lower, upper) in sorted(result.confidence_intervals.items()):
            lines.append(f"  {level:>4}: ${lower:>10,.2f} - ${upper:>10,.2f}")
        lines.append("")

        # Forces breakdown
        lines.append("STEERING FORCES:")
        lines.append("-" * 40)
        f = result.forces
        lines.append(f"  Funding:      {f.funding_force*100:+8.4f}%")
        lines.append(f"  Imbalance:    {f.imbalance_force*100:+8.4f}%")
        lines.append(f"  Gravity:      {f.gravity_force*100:+8.4f}%")
        lines.append(f"  Spring:       {f.spring_force*100:+8.4f}%")
        lines.append(f"  Momentum:     {f.momentum_force*100:+8.4f}%")
        lines.append(f"  Liquidation:  {f.liquidation_force*100:+8.4f}%")
        lines.append(f"  Friction:     {f.friction_force*100:+8.4f}%")
        lines.append(f"  ─────────────────────────")
        lines.append(f"  NET DRIFT:    {f.total_drift*100:+8.4f}%")
        lines.append("")

        # Simulation stats
        lines.append("SIMULATION STATISTICS:")
        lines.append("-" * 40)
        s = result.simulation_stats
        lines.append(f"  Std Dev:      ${s['std']:,.2f}")
        lines.append(f"  Skewness:     {s['skew']:+.3f}")
        lines.append(f"  Kurtosis:     {s['kurtosis']:+.3f}")
        lines.append(f"  Range:        ${s['min']:,.2f} - ${s['max']:,.2f}")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


def compute_quantum_features(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """Compute quantum/physics-inspired features for prediction.

    Adds steering field features to existing DataFrame.
    """
    config = config or {}
    result = df.copy()

    # Ensure we have returns
    if "returns" not in result.columns:
        result["returns"] = result["close"].pct_change()

    # Regime encoding
    engine = QuantumPriceEngine(n_simulations=100)  # Small for feature computation

    # Rolling regime detection
    def detect_regime_encoded(window):
        if len(window) < 20:
            return 1  # Default to ranging
        mini_df = pd.DataFrame({"close": window, "high": window, "low": window})
        mini_df["returns"] = mini_df["close"].pct_change()
        regime = engine.detect_regime(mini_df, lookback=len(window))
        return {"trending": 0, "ranging": 1, "volatile": 2}[regime.value]

    result["regime_code"] = result["close"].rolling(60).apply(
        detect_regime_encoded, raw=False
    )

    # Multi-scale momentum
    for window, name in [(5, "1m"), (15, "5m"), (45, "15m"), (60, "1h")]:
        result[f"momentum_{name}"] = result["returns"].rolling(window).mean()

    # Aggregated momentum (weighted)
    result["momentum_weighted"] = (
        result["momentum_1m"] * 0.4 +
        result["momentum_5m"] * 0.3 +
        result["momentum_15m"] * 0.2 +
        result["momentum_1h"] * 0.1
    )

    # VWAP displacement (spring potential)
    if "volume" in result.columns:
        result["vwap"] = (result["close"] * result["volume"]).rolling(50).sum() / result["volume"].rolling(50).sum()
    else:
        result["vwap"] = result["close"].rolling(50).mean()

    result["vwap_displacement"] = (result["close"] - result["vwap"]) / result["close"]
    result["spring_potential"] = -15.0 * result["vwap_displacement"]  # Hooke's law

    # Volatility forecasting (GARCH-like)
    result["vol_short"] = result["returns"].rolling(10).std()
    result["vol_long"] = result["returns"].rolling(60).std()
    result["vol_ratio"] = result["vol_short"] / (result["vol_long"] + 1e-9)

    # Jump detection (large moves)
    threshold = result["returns"].rolling(60).std() * 3
    result["jump_detected"] = (result["returns"].abs() > threshold).astype(float)
    result["jump_intensity"] = result["jump_detected"].rolling(20).mean()

    # Regime stability (how long in current regime)
    result["regime_change"] = result["regime_code"].diff().abs()
    result["regime_stability"] = 1.0 - result["regime_change"].rolling(20).mean()

    return result


class EnhancedQuantumEngine(QuantumPriceEngine):
    """Enhanced quantum engine integrating all steering field modules.

    This class extends QuantumPriceEngine to incorporate:
    - Cross-exchange funding aggregation
    - Liquidation heatmaps
    - Gamma fields from options
    - Cross-asset coupling (SPX/DXY)
    - Time-of-day volatility effects
    - Multi-level order book analysis

    All components expose ML-tunable hyperparameters.
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        random_seed: Optional[int] = None,
        configs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize enhanced engine with all steering field modules.

        Args:
            n_simulations: Number of Monte Carlo paths
            random_seed: Random seed for reproducibility
            configs: Dict of configs for each module:
                - funding: FundingConfig
                - liquidation: LiquidationConfig
                - gamma: GammaConfig
                - macro: MacroCouplingConfig
                - temporal: TemporalConfig
                - orderbook: OrderBookConfig
        """
        super().__init__(n_simulations, random_seed)

        configs = configs or {}

        # Import modules here to avoid circular imports
        from .funding import FundingRateAggregator, FundingConfig
        from .liquidations import LiquidationHeatmap, LiquidationConfig
        from .gamma import GammaFieldCalculator, GammaConfig
        from .macro import CrossAssetCoupling, MacroCouplingConfig
        from .temporal import TimeOfDayEffects, TemporalConfig
        from .orderbook import MultiLevelOrderBookAnalyzer, OrderBookConfig

        # Initialize all steering field modules
        self.funding_aggregator = FundingRateAggregator(
            configs.get('funding', FundingConfig())
        )
        self.liquidation_heatmap = LiquidationHeatmap(
            configs.get('liquidation', LiquidationConfig())
        )
        self.gamma_calculator = GammaFieldCalculator(
            configs.get('gamma', GammaConfig())
        )
        self.macro_coupling = CrossAssetCoupling(
            configs.get('macro', MacroCouplingConfig())
        )
        self.temporal_effects = TimeOfDayEffects(
            configs.get('temporal', TemporalConfig())
        )
        self.orderbook_analyzer = MultiLevelOrderBookAnalyzer(
            configs.get('orderbook', OrderBookConfig())
        )

    def predict_enhanced(
        self,
        df: pd.DataFrame,
        horizon_minutes: int = 60,
        funding_rates: Optional[Dict[str, float]] = None,
        orderbook: Optional[Dict[str, List]] = None,
        liquidation_levels: Optional[List[Tuple[float, float, float]]] = None,
        options_oi: Optional[float] = None,
        btc_returns: Optional[pd.Series] = None,
        current_time: Optional[datetime] = None,
    ) -> PredictionResult:
        """Generate enhanced probabilistic prediction using all steering fields.

        Args:
            df: DataFrame with OHLCV data
            horizon_minutes: Prediction horizon in minutes
            funding_rates: Dict of exchange -> funding rate
            orderbook: Dict with 'bids' and 'asks' lists
            liquidation_levels: List of (price, long_vol, short_vol)
            options_oi: Options open interest (for gamma estimation)
            btc_returns: BTC return series for macro correlation
            current_time: Current datetime for temporal effects

        Returns:
            PredictionResult with point estimate and confidence intervals
        """
        current_price = float(df["close"].iloc[-1])
        current_time = current_time or datetime.now()

        # Detect regime
        regime = self.detect_regime(df)
        regime_str = regime.value
        params = self.REGIME_PARAMS[regime]

        # Initialize forces
        forces = SteeringForces()

        # === 1. FUNDING FORCE (Cross-Exchange Aggregated) ===
        if funding_rates:
            funding_result = self.funding_aggregator.aggregate(funding_rates, regime_str)
            forces.funding_force = funding_result['funding_force']
        else:
            forces.funding_force = 0.0

        # === 2. ORDER BOOK IMBALANCE (Multi-Level) ===
        if orderbook and orderbook.get('bids') and orderbook.get('asks'):
            snapshot = self.orderbook_analyzer.process_snapshot(
                orderbook['bids'],
                orderbook['asks'],
            )
            ob_result = self.orderbook_analyzer.calculate_force(snapshot, regime_str)
            forces.imbalance_force = ob_result['imbalance_force']
            forces.gravity_force = ob_result['gravity_force']
        else:
            # Fallback to simple calculation
            bid_volume = df.get('bid_volume', pd.Series([0])).iloc[-1] if 'bid_volume' in df else 0
            ask_volume = df.get('ask_volume', pd.Series([0])).iloc[-1] if 'ask_volume' in df else 0
            forces.imbalance_force = self.calculate_order_book_imbalance(
                bid_volume, ask_volume, params
            )

        # === 3. LIQUIDATION CASCADE FORCE ===
        if liquidation_levels:
            from .liquidations import LiquidationLevel
            levels = [
                LiquidationLevel(price=p, long_volume=lv, short_volume=sv)
                for p, lv, sv in liquidation_levels
            ]
            self.liquidation_heatmap.update_levels(levels)
            liq_result = self.liquidation_heatmap.calculate_force(current_price, regime_str)
            forces.liquidation_force = liq_result['liquidation_force']
        elif options_oi:
            # Estimate from futures OI
            self.liquidation_heatmap.estimate_from_leverage_distribution(
                current_price, options_oi
            )
            liq_result = self.liquidation_heatmap.calculate_force(current_price, regime_str)
            forces.liquidation_force = liq_result['liquidation_force']
        else:
            forces.liquidation_force = 0.0

        # === 4. GAMMA FORCE (Options Market) ===
        if options_oi:
            self.gamma_calculator.estimate_from_futures_oi(current_price, options_oi)
            gamma_result = self.gamma_calculator.calculate_force(current_price, regime_str)
            gamma_force = gamma_result['total_gamma_force']
        else:
            gamma_force = 0.0

        # === 5. MACRO COUPLING FORCE (SPX/DXY) ===
        if btc_returns is not None and len(btc_returns) > 100:
            macro_result = self.macro_coupling.calculate_macro_drift(
                btc_returns, current_time, regime_str
            )
            macro_force = macro_result['macro_drift']
        else:
            macro_force = 0.0

        # === 6. SPRING FORCE (VWAP Mean Reversion) ===
        if "volume" in df.columns:
            vwap = (df["close"] * df["volume"]).tail(50).sum() / df["volume"].tail(50).sum()
        else:
            vwap = df["close"].tail(50).mean()
        forces.spring_force = self.calculate_spring_force(current_price, vwap, params)

        # === 7. MOMENTUM FORCE (Multi-Scale) ===
        returns = df["close"].pct_change() if "returns" not in df.columns else df["returns"]
        momentum_scales = {
            "1m": returns.tail(5).mean() if len(returns) >= 5 else 0,
            "5m": returns.tail(15).mean() if len(returns) >= 15 else 0,
            "15m": returns.tail(45).mean() if len(returns) >= 45 else 0,
            "1h": returns.tail(60).mean() if len(returns) >= 60 else 0,
        }
        forces.momentum_force = self.calculate_momentum_force(momentum_scales, params)

        # === 8. TIME-OF-DAY EFFECTS ===
        recent_vol = returns.tail(60).std() if len(returns) >= 60 else params.volatility_base
        temporal_result = self.temporal_effects.calculate_combined_multiplier(
            current_time, recent_vol
        )
        vol_multiplier = temporal_result['combined_multiplier']
        temporal_bias = temporal_result['directional_bias']

        # === 9. FRICTION FORCE ===
        # Include additional forces in total before friction
        total_before_friction = (
            forces.funding_force +
            forces.imbalance_force +
            forces.gravity_force +
            forces.spring_force +
            forces.liquidation_force +
            forces.momentum_force +
            gamma_force +
            macro_force +
            temporal_bias
        )
        forces.friction_force = -params.drag_coefficient * total_before_friction * 0.1

        # === VOLATILITY FORECAST ===
        current_vol = returns.tail(60).std() if len(returns) >= 60 else params.volatility_base
        volatility = self.forecast_volatility(returns, current_vol, params)
        # Apply time-of-day multiplier and floor
        volatility = max(volatility * vol_multiplier, self.VOLATILITY_FLOOR)
        volatility *= np.sqrt(horizon_minutes / 60)

        # === ADD EXTRA FORCES TO TOTAL ===
        # These aren't in SteeringForces dataclass but contribute to drift
        extra_drift = gamma_force + macro_force + temporal_bias

        # Create modified forces for simulation
        sim_forces = SteeringForces(
            funding_force=forces.funding_force,
            imbalance_force=forces.imbalance_force,
            gravity_force=forces.gravity_force,
            spring_force=forces.spring_force,
            depth_pressure=extra_drift,  # Use depth_pressure to carry extra forces
            momentum_force=forces.momentum_force,
            liquidation_force=forces.liquidation_force,
            friction_force=forces.friction_force,
        )

        # === RUN MONTE CARLO SIMULATION ===
        final_prices = self.simulate_paths(
            current_price=current_price,
            forces=sim_forces,
            volatility=volatility,
            params=params,
            n_steps=horizon_minutes,
        )

        # === COMPUTE RESULTS ===
        point_estimate = float(np.median(final_prices))
        mean_estimate = float(np.mean(final_prices))
        probability_up = float(np.mean(final_prices > current_price))

        confidence_intervals = {
            "50%": (float(np.percentile(final_prices, 25)), float(np.percentile(final_prices, 75))),
            "68%": (float(np.percentile(final_prices, 16)), float(np.percentile(final_prices, 84))),
            "80%": (float(np.percentile(final_prices, 10)), float(np.percentile(final_prices, 90))),
            "90%": (float(np.percentile(final_prices, 5)), float(np.percentile(final_prices, 95))),
            "95%": (float(np.percentile(final_prices, 2.5)), float(np.percentile(final_prices, 97.5))),
            "99%": (float(np.percentile(final_prices, 0.5)), float(np.percentile(final_prices, 99.5))),
        }

        simulation_stats = {
            "mean": mean_estimate,
            "std": float(np.std(final_prices)),
            "skew": float(pd.Series(final_prices).skew()),
            "kurtosis": float(pd.Series(final_prices).kurtosis()),
            "min": float(np.min(final_prices)),
            "max": float(np.max(final_prices)),
            "gamma_force": gamma_force,
            "macro_force": macro_force,
            "temporal_bias": temporal_bias,
            "vol_multiplier": vol_multiplier,
        }

        return PredictionResult(
            current_price=current_price,
            regime=regime,
            point_estimate=point_estimate,
            expected_return_pct=(point_estimate / current_price - 1) * 100,
            probability_up=probability_up,
            confidence_intervals=confidence_intervals,
            forces=forces,
            simulation_stats=simulation_stats,
        )

    def get_all_features(
        self,
        df: pd.DataFrame,
        funding_rates: Optional[Dict[str, float]] = None,
        orderbook: Optional[Dict[str, List]] = None,
        current_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get all features from all steering field modules.

        Useful for ML models that need feature vectors.

        Args:
            df: DataFrame with OHLCV data
            funding_rates: Funding rates by exchange
            orderbook: Order book data
            current_time: Current datetime

        Returns:
            Dict of all features
        """
        features = {}
        current_price = float(df["close"].iloc[-1])
        current_time = current_time or datetime.now()
        regime = self.detect_regime(df)
        regime_str = regime.value

        # Returns for various calculations
        returns = df["close"].pct_change() if "returns" not in df.columns else df["returns"]
        recent_vol = returns.tail(60).std() if len(returns) >= 60 else 0.02

        # Regime features
        features['regime_trending'] = 1.0 if regime_str == 'trending' else 0.0
        features['regime_ranging'] = 1.0 if regime_str == 'ranging' else 0.0
        features['regime_volatile'] = 1.0 if regime_str == 'volatile' else 0.0

        # Funding features
        if funding_rates:
            funding_features = self.funding_aggregator.get_funding_features(
                funding_rates, regime_str
            )
            features.update(funding_features)

        # Order book features
        if orderbook and orderbook.get('bids') and orderbook.get('asks'):
            snapshot = self.orderbook_analyzer.process_snapshot(
                orderbook['bids'], orderbook['asks']
            )
            ob_features = self.orderbook_analyzer.get_orderbook_features(
                snapshot, regime_str
            )
            features.update(ob_features)

        # Temporal features
        temporal_features = self.temporal_effects.get_temporal_features(
            current_time, recent_vol
        )
        features.update(temporal_features)

        # Liquidation features (if heatmap has data)
        liq_features = self.liquidation_heatmap.get_liquidation_features(
            current_price, regime_str
        )
        features.update(liq_features)

        # Gamma features (if data available)
        gamma_features = self.gamma_calculator.get_gamma_features(
            current_price, regime_str
        )
        features.update(gamma_features)

        return features
