"""Cross-Exchange Funding Rate Aggregation.

Aggregates funding rates from multiple exchanges with configurable weights.
All parameters are exposed as hyperparameters for ML tuning.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


@dataclass
class FundingConfig:
    """Hyperparameters for funding rate aggregation.

    All parameters can be tuned by the improvement loop.
    """
    # Exchange weights (must sum to 1.0)
    weight_binance: float = 0.50
    weight_bybit: float = 0.30
    weight_okx: float = 0.15
    weight_deribit: float = 0.05

    # Force calculation parameters
    funding_strength_base: float = 12.0  # Base multiplier for funding force
    funding_strength_trending: float = 10.0
    funding_strength_ranging: float = 18.0
    funding_strength_volatile: float = 8.0

    # Nonlinearity parameters
    funding_exponent: float = 1.0  # Linear by default, can be tuned
    funding_saturation: float = 0.001  # Saturation point for extreme funding

    # Temporal smoothing
    ema_alpha: float = 0.3  # EMA smoothing factor for funding rate

    # Divergence detection
    divergence_threshold: float = 0.0002  # When exchanges disagree
    divergence_penalty: float = 0.5  # Reduce force when divergent

    def get_weights(self) -> Dict[str, float]:
        """Get normalized exchange weights."""
        weights = {
            'binance': self.weight_binance,
            'bybit': self.weight_bybit,
            'okx': self.weight_okx,
            'deribit': self.weight_deribit,
        }
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        return weights


class FundingRateAggregator:
    """Aggregates funding rates across multiple exchanges.

    Funding rate creates a restoring force:
    - Positive funding (longs pay shorts) -> downward pressure
    - Negative funding (shorts pay longs) -> upward pressure

    The aggregation uses configurable weights and detects
    divergence between exchanges as a signal of uncertainty.
    """

    SUPPORTED_EXCHANGES = ['binance', 'bybit', 'okx', 'deribit']

    def __init__(self, config: Optional[FundingConfig] = None):
        """Initialize aggregator with config.

        Args:
            config: FundingConfig with hyperparameters
        """
        self.config = config or FundingConfig()
        self._ema_funding: Optional[float] = None
        self._history: List[Dict[str, float]] = []

    def aggregate(
        self,
        funding_rates: Dict[str, float],
        regime: str = 'ranging',
    ) -> Dict[str, float]:
        """Aggregate funding rates from multiple exchanges.

        Args:
            funding_rates: Dict mapping exchange name to funding rate
                e.g., {'binance': 0.0001, 'bybit': 0.00012, 'okx': 0.00009}
            regime: Current market regime for strength adjustment

        Returns:
            Dict with aggregated metrics:
                - weighted_funding: Volume-weighted average funding
                - funding_force: Steering force from funding
                - divergence: Cross-exchange disagreement
                - confidence: Confidence in the signal
        """
        if not funding_rates:
            return {
                'weighted_funding': 0.0,
                'funding_force': 0.0,
                'divergence': 0.0,
                'confidence': 0.0,
            }

        weights = self.config.get_weights()

        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        valid_rates = []

        for exchange, rate in funding_rates.items():
            if exchange in weights and rate is not None and not np.isnan(rate):
                w = weights[exchange]
                weighted_sum += w * rate
                total_weight += w
                valid_rates.append(rate)

        if total_weight == 0 or len(valid_rates) == 0:
            return {
                'weighted_funding': 0.0,
                'funding_force': 0.0,
                'divergence': 0.0,
                'confidence': 0.0,
            }

        weighted_funding = weighted_sum / total_weight

        # Apply EMA smoothing
        if self._ema_funding is None:
            self._ema_funding = weighted_funding
        else:
            alpha = self.config.ema_alpha
            self._ema_funding = alpha * weighted_funding + (1 - alpha) * self._ema_funding

        smoothed_funding = self._ema_funding

        # Calculate divergence (cross-exchange disagreement)
        if len(valid_rates) >= 2:
            divergence = np.std(valid_rates)
        else:
            divergence = 0.0

        # Determine funding strength based on regime
        strength_map = {
            'trending': self.config.funding_strength_trending,
            'ranging': self.config.funding_strength_ranging,
            'volatile': self.config.funding_strength_volatile,
        }
        base_strength = strength_map.get(regime, self.config.funding_strength_base)

        # Apply divergence penalty
        if divergence > self.config.divergence_threshold:
            strength_multiplier = self.config.divergence_penalty
        else:
            strength_multiplier = 1.0

        # Calculate funding force with nonlinearity
        # F = -sign(f) * strength * |f|^exponent / (|f| + saturation)
        sign = np.sign(smoothed_funding)
        magnitude = abs(smoothed_funding)

        if self.config.funding_exponent != 1.0:
            magnitude = magnitude ** self.config.funding_exponent

        # Saturation prevents extreme forces
        saturated = magnitude / (magnitude + self.config.funding_saturation)

        funding_force = -sign * base_strength * saturated * strength_multiplier

        # Confidence based on agreement and magnitude
        confidence = (1.0 - min(divergence / self.config.divergence_threshold, 1.0)) * \
                     min(abs(weighted_funding) / 0.0005, 1.0)  # Normalize by typical funding

        # Store history
        self._history.append({
            'weighted_funding': weighted_funding,
            'smoothed_funding': smoothed_funding,
            'divergence': divergence,
        })
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        return {
            'weighted_funding': weighted_funding,
            'smoothed_funding': smoothed_funding,
            'funding_force': funding_force,
            'divergence': divergence,
            'confidence': confidence,
            'regime_strength': base_strength,
        }

    def get_funding_features(
        self,
        funding_rates: Dict[str, float],
        regime: str = 'ranging',
    ) -> Dict[str, float]:
        """Get funding-related features for ML model.

        Returns features that can be used by predictors.
        """
        result = self.aggregate(funding_rates, regime)

        features = {
            'funding_weighted': result['weighted_funding'],
            'funding_smoothed': result.get('smoothed_funding', result['weighted_funding']),
            'funding_force': result['funding_force'],
            'funding_divergence': result['divergence'],
            'funding_confidence': result['confidence'],
        }

        # Add individual exchange features
        for exchange in self.SUPPORTED_EXCHANGES:
            rate = funding_rates.get(exchange)
            features[f'funding_{exchange}'] = rate if rate is not None else 0.0

        # Add historical features if available
        if len(self._history) >= 8:
            recent = self._history[-8:]
            features['funding_trend'] = recent[-1]['smoothed_funding'] - recent[0]['smoothed_funding']
            features['funding_volatility'] = np.std([h['weighted_funding'] for h in recent])
        else:
            features['funding_trend'] = 0.0
            features['funding_volatility'] = 0.0

        return features

    def reset(self):
        """Reset internal state."""
        self._ema_funding = None
        self._history = []


def get_funding_hyperparameters() -> List[Dict]:
    """Get hyperparameter definitions for improvement loop.

    Returns list of variable definitions compatible with
    YoshiImprovementLoop.
    """
    return [
        {
            'name': 'funding_weight_binance',
            'path': 'particle.funding.weight_binance',
            'current_value': 0.50,
            'candidates': [0.3, 0.4, 0.5, 0.6, 0.7],
            'variable_type': 'continuous',
        },
        {
            'name': 'funding_weight_bybit',
            'path': 'particle.funding.weight_bybit',
            'current_value': 0.30,
            'candidates': [0.1, 0.2, 0.3, 0.4],
            'variable_type': 'continuous',
        },
        {
            'name': 'funding_strength_base',
            'path': 'particle.funding.funding_strength_base',
            'current_value': 12.0,
            'candidates': [6.0, 9.0, 12.0, 15.0, 18.0, 24.0],
            'variable_type': 'continuous',
        },
        {
            'name': 'funding_exponent',
            'path': 'particle.funding.funding_exponent',
            'current_value': 1.0,
            'candidates': [0.5, 0.75, 1.0, 1.25, 1.5],
            'variable_type': 'continuous',
        },
        {
            'name': 'funding_ema_alpha',
            'path': 'particle.funding.ema_alpha',
            'current_value': 0.3,
            'candidates': [0.1, 0.2, 0.3, 0.5, 0.7],
            'variable_type': 'continuous',
        },
        {
            'name': 'funding_divergence_threshold',
            'path': 'particle.funding.divergence_threshold',
            'current_value': 0.0002,
            'candidates': [0.0001, 0.0002, 0.0003, 0.0005],
            'variable_type': 'continuous',
        },
    ]
