"""Liquidation Heatmap and Cascade Dynamics.

Models liquidation levels as potential wells that create:
- Repulsion when price approaches (market makers defend)
- Attraction/acceleration once breached (cascade effect)

All parameters are exposed as hyperparameters for ML tuning.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import gaussian_filter1d


@dataclass
class LiquidationConfig:
    """Hyperparameters for liquidation cascade modeling.

    All parameters can be tuned by the improvement loop.
    """
    # Distance thresholds (as fraction of price)
    repulsion_zone_pct: float = 0.01  # 1% - zone where MM defend
    cascade_zone_pct: float = 0.002  # 0.2% - zone where cascade accelerates
    influence_zone_pct: float = 0.05  # 5% - max distance for any effect

    # Force strength parameters
    repulsion_strength: float = 1e-6  # Base repulsion force
    cascade_strength: float = 1e-5  # Cascade acceleration force
    volume_exponent: float = 0.8  # How volume scales force (sublinear)

    # Regime-dependent multipliers
    strength_trending: float = 0.8
    strength_ranging: float = 1.2
    strength_volatile: float = 1.5

    # Density estimation parameters
    density_bandwidth: float = 0.005  # Kernel bandwidth for density estimation
    density_resolution: int = 200  # Number of price levels to estimate

    # Asymmetry parameters (longs vs shorts)
    long_liq_weight: float = 1.0  # Weight for long liquidations (below price)
    short_liq_weight: float = 1.0  # Weight for short liquidations (above price)

    # Force clamping
    max_force: float = 0.02  # Maximum force magnitude (2%)

    # Historical decay
    decay_halflife_hours: float = 24.0  # How fast old liquidation levels decay


@dataclass
class LiquidationLevel:
    """A single liquidation price level."""
    price: float
    long_volume: float  # Volume of longs that liquidate here
    short_volume: float  # Volume of shorts that liquidate here
    timestamp: Optional[float] = None  # When this level was observed

    @property
    def total_volume(self) -> float:
        return self.long_volume + self.short_volume

    @property
    def net_volume(self) -> float:
        """Positive = more longs, negative = more shorts."""
        return self.long_volume - self.short_volume


class LiquidationHeatmap:
    """Manages liquidation level data and computes steering forces.

    The liquidation field creates complex dynamics:
    1. Price is repelled from large liquidation clusters (MM defense)
    2. Once a cluster is breached, price accelerates through (cascade)
    3. The cascade itself creates new liquidations (positive feedback)
    """

    def __init__(self, config: Optional[LiquidationConfig] = None):
        """Initialize heatmap with config.

        Args:
            config: LiquidationConfig with hyperparameters
        """
        self.config = config or LiquidationConfig()
        self._levels: List[LiquidationLevel] = []
        self._density_cache: Optional[Dict] = None

    def update_levels(self, levels: List[LiquidationLevel]):
        """Update liquidation levels from external source.

        Args:
            levels: List of LiquidationLevel objects
        """
        self._levels = levels
        self._density_cache = None  # Invalidate cache

    def add_level(self, level: LiquidationLevel):
        """Add a single liquidation level."""
        self._levels.append(level)
        self._density_cache = None

    def compute_density(
        self,
        current_price: float,
        price_range_pct: float = 0.10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute liquidation density function.

        Args:
            current_price: Current market price
            price_range_pct: Range around current price to analyze

        Returns:
            Tuple of (prices, long_density, short_density)
        """
        if not self._levels:
            n = self.config.density_resolution
            prices = np.linspace(
                current_price * (1 - price_range_pct),
                current_price * (1 + price_range_pct),
                n
            )
            return prices, np.zeros(n), np.zeros(n)

        # Define price grid
        min_price = current_price * (1 - price_range_pct)
        max_price = current_price * (1 + price_range_pct)
        n = self.config.density_resolution
        prices = np.linspace(min_price, max_price, n)

        # Initialize density arrays
        long_density = np.zeros(n)
        short_density = np.zeros(n)

        # Kernel density estimation
        bandwidth = self.config.density_bandwidth * current_price

        for level in self._levels:
            if min_price <= level.price <= max_price:
                # Find closest price index
                idx = np.argmin(np.abs(prices - level.price))

                # Apply volume with decay based on age
                decay = 1.0
                if level.timestamp is not None:
                    age_hours = (pd.Timestamp.now().timestamp() - level.timestamp) / 3600
                    decay = 0.5 ** (age_hours / self.config.decay_halflife_hours)

                long_density[idx] += level.long_volume * decay
                short_density[idx] += level.short_volume * decay

        # Apply Gaussian smoothing for density estimation
        sigma = bandwidth / (prices[1] - prices[0]) if len(prices) > 1 else 1
        long_density = gaussian_filter1d(long_density, sigma=max(sigma, 0.5))
        short_density = gaussian_filter1d(short_density, sigma=max(sigma, 0.5))

        return prices, long_density, short_density

    def calculate_force(
        self,
        current_price: float,
        regime: str = 'ranging',
    ) -> Dict[str, float]:
        """Calculate steering force from liquidation levels.

        The force model:
        - Repulsion zone: F = -sign(distance) * strength * volume^exp / distance
        - Cascade zone: F = sign(distance) * cascade_strength * volume^exp

        Args:
            current_price: Current market price
            regime: Current market regime

        Returns:
            Dict with force components and metadata
        """
        if not self._levels:
            return {
                'liquidation_force': 0.0,
                'long_liq_pressure': 0.0,
                'short_liq_pressure': 0.0,
                'nearest_long_liq': 0.0,
                'nearest_short_liq': 0.0,
                'cascade_risk': 0.0,
            }

        # Get regime multiplier
        regime_mult = {
            'trending': self.config.strength_trending,
            'ranging': self.config.strength_ranging,
            'volatile': self.config.strength_volatile,
        }.get(regime, 1.0)

        net_force = 0.0
        long_pressure = 0.0
        short_pressure = 0.0
        nearest_long = float('inf')
        nearest_short = float('inf')
        cascade_risk = 0.0

        for level in self._levels:
            distance_pct = (current_price - level.price) / current_price

            # Skip if outside influence zone
            if abs(distance_pct) > self.config.influence_zone_pct:
                continue

            # Calculate volume contribution with exponent
            vol = level.total_volume ** self.config.volume_exponent

            # Determine which type of liquidation
            if level.price < current_price:
                # Long liquidations below price
                weight = self.config.long_liq_weight
                vol_weighted = vol * weight * level.long_volume / (level.total_volume + 1e-9)

                # Track nearest
                distance = current_price - level.price
                if distance < nearest_long:
                    nearest_long = distance

                # Repulsion zone (price pushed away from liq level)
                if abs(distance_pct) < self.config.repulsion_zone_pct:
                    force = self.config.repulsion_strength * vol_weighted / (abs(distance_pct) + 0.0001)
                    net_force += force  # Positive = pushed up away from long liqs
                    long_pressure += force

                # Cascade zone (price accelerates through)
                if abs(distance_pct) < self.config.cascade_zone_pct:
                    cascade_force = -self.config.cascade_strength * vol_weighted
                    net_force += cascade_force  # Negative = pulled down into cascade
                    cascade_risk += abs(vol_weighted)

            else:
                # Short liquidations above price
                weight = self.config.short_liq_weight
                vol_weighted = vol * weight * level.short_volume / (level.total_volume + 1e-9)

                # Track nearest
                distance = level.price - current_price
                if distance < nearest_short:
                    nearest_short = distance

                # Repulsion zone
                if abs(distance_pct) < self.config.repulsion_zone_pct:
                    force = self.config.repulsion_strength * vol_weighted / (abs(distance_pct) + 0.0001)
                    net_force -= force  # Negative = pushed down away from short liqs
                    short_pressure += force

                # Cascade zone
                if abs(distance_pct) < self.config.cascade_zone_pct:
                    cascade_force = self.config.cascade_strength * vol_weighted
                    net_force += cascade_force  # Positive = pulled up into cascade
                    cascade_risk += abs(vol_weighted)

        # Apply regime multiplier and clamp
        net_force *= regime_mult
        net_force = np.clip(net_force, -self.config.max_force, self.config.max_force)

        # Normalize cascade risk
        if cascade_risk > 0:
            cascade_risk = min(cascade_risk / 1e6, 1.0)

        return {
            'liquidation_force': net_force,
            'long_liq_pressure': long_pressure * regime_mult,
            'short_liq_pressure': short_pressure * regime_mult,
            'nearest_long_liq': nearest_long if nearest_long < float('inf') else 0,
            'nearest_short_liq': nearest_short if nearest_short < float('inf') else 0,
            'cascade_risk': cascade_risk,
        }

    def get_liquidation_features(
        self,
        current_price: float,
        regime: str = 'ranging',
    ) -> Dict[str, float]:
        """Get liquidation-related features for ML model.

        Returns features that can be used by predictors.
        """
        force_result = self.calculate_force(current_price, regime)

        # Compute density
        prices, long_den, short_den = self.compute_density(current_price)

        # Find peak densities near current price
        mid_idx = len(prices) // 2
        range_idx = int(0.02 * len(prices))  # 2% range

        nearby_long = long_den[max(0, mid_idx - range_idx):mid_idx].sum()
        nearby_short = short_den[mid_idx:min(len(prices), mid_idx + range_idx)].sum()

        features = {
            'liq_force': force_result['liquidation_force'],
            'liq_long_pressure': force_result['long_liq_pressure'],
            'liq_short_pressure': force_result['short_liq_pressure'],
            'liq_nearest_long_pct': force_result['nearest_long_liq'] / current_price if current_price > 0 else 0,
            'liq_nearest_short_pct': force_result['nearest_short_liq'] / current_price if current_price > 0 else 0,
            'liq_cascade_risk': force_result['cascade_risk'],
            'liq_density_long_nearby': nearby_long,
            'liq_density_short_nearby': nearby_short,
            'liq_density_imbalance': (nearby_long - nearby_short) / (nearby_long + nearby_short + 1e-9),
        }

        return features

    def estimate_from_leverage_distribution(
        self,
        current_price: float,
        open_interest: float,
        avg_leverage: float = 10.0,
        long_ratio: float = 0.5,
    ):
        """Estimate liquidation levels from leverage distribution.

        When actual liquidation data isn't available, we can estimate
        based on typical leverage patterns.

        Args:
            current_price: Current market price
            open_interest: Total open interest in contracts
            avg_leverage: Average leverage used by traders
            long_ratio: Ratio of longs to total positions
        """
        # Clear existing levels
        self._levels = []

        # Estimate liquidation prices for various leverage levels
        leverages = [3, 5, 10, 20, 50, 100]

        for lev in leverages:
            # Long liquidation price (below current)
            # Liquidation occurs when: (current - liq_price) / liq_price = 1/leverage
            long_liq_price = current_price * (1 - 1/lev)

            # Short liquidation price (above current)
            short_liq_price = current_price * (1 + 1/lev)

            # Estimate volume at this leverage (fewer traders use extreme leverage)
            vol_fraction = 1.0 / (lev ** 0.5)  # Square root decay
            base_vol = open_interest * vol_fraction * 0.1

            self._levels.append(LiquidationLevel(
                price=long_liq_price,
                long_volume=base_vol * long_ratio,
                short_volume=0,
                timestamp=pd.Timestamp.now().timestamp(),
            ))

            self._levels.append(LiquidationLevel(
                price=short_liq_price,
                long_volume=0,
                short_volume=base_vol * (1 - long_ratio),
                timestamp=pd.Timestamp.now().timestamp(),
            ))

        self._density_cache = None

    def clear(self):
        """Clear all liquidation levels."""
        self._levels = []
        self._density_cache = None


def get_liquidation_hyperparameters() -> List[Dict]:
    """Get hyperparameter definitions for improvement loop."""
    return [
        {
            'name': 'liq_repulsion_zone_pct',
            'path': 'particle.liquidation.repulsion_zone_pct',
            'current_value': 0.01,
            'candidates': [0.005, 0.01, 0.015, 0.02],
            'variable_type': 'continuous',
        },
        {
            'name': 'liq_cascade_zone_pct',
            'path': 'particle.liquidation.cascade_zone_pct',
            'current_value': 0.002,
            'candidates': [0.001, 0.002, 0.003, 0.005],
            'variable_type': 'continuous',
        },
        {
            'name': 'liq_repulsion_strength',
            'path': 'particle.liquidation.repulsion_strength',
            'current_value': 1e-6,
            'candidates': [5e-7, 1e-6, 2e-6, 5e-6],
            'variable_type': 'continuous',
        },
        {
            'name': 'liq_cascade_strength',
            'path': 'particle.liquidation.cascade_strength',
            'current_value': 1e-5,
            'candidates': [5e-6, 1e-5, 2e-5, 5e-5],
            'variable_type': 'continuous',
        },
        {
            'name': 'liq_volume_exponent',
            'path': 'particle.liquidation.volume_exponent',
            'current_value': 0.8,
            'candidates': [0.5, 0.7, 0.8, 0.9, 1.0],
            'variable_type': 'continuous',
        },
        {
            'name': 'liq_density_bandwidth',
            'path': 'particle.liquidation.density_bandwidth',
            'current_value': 0.005,
            'candidates': [0.002, 0.005, 0.01, 0.02],
            'variable_type': 'continuous',
        },
    ]
