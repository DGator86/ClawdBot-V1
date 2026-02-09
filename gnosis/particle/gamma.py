"""Gamma Fields from Options Market.

Models the steering effect from options market makers' delta hedging:
- Negative gamma: Dealers buy dips, sell rips (stabilizing)
- Positive gamma: Dealers chase price (destabilizing)

Also computes max pain levels where most options expire worthless.

All parameters are exposed as hyperparameters for ML tuning.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm
from datetime import datetime, timedelta


@dataclass
class GammaConfig:
    """Hyperparameters for gamma field modeling.

    All parameters can be tuned by the improvement loop.
    """
    # Force strength parameters
    gamma_strength: float = 0.001  # Base strength of gamma steering
    max_pain_strength: float = 0.0005  # Strength of max pain attraction

    # Distance parameters
    gamma_decay_distance: float = 0.05  # 5% - gamma effect decays over this range
    max_pain_influence_zone: float = 0.03  # 3% - max pain attracts within this zone

    # Time decay
    expiry_relevance_days: float = 7.0  # Options expiring within this window matter most
    expiry_decay_rate: float = 0.3  # How fast relevance decays with time to expiry

    # Regime multipliers
    strength_trending: float = 0.5  # Gamma less important when trending
    strength_ranging: float = 1.5  # Gamma more important when ranging
    strength_volatile: float = 0.8

    # Aggregation parameters
    strike_aggregation_pct: float = 0.01  # Aggregate strikes within 1%
    min_open_interest: float = 100  # Minimum OI to consider a strike

    # Net gamma sign effect
    negative_gamma_stabilizing: float = 1.0  # Multiplier for stabilizing effect
    positive_gamma_destabilizing: float = 1.2  # Multiplier for destabilizing effect

    # Max pain calculation
    max_pain_call_weight: float = 1.0
    max_pain_put_weight: float = 1.0


@dataclass
class OptionStrike:
    """Data for a single options strike."""
    strike: float
    expiry: datetime
    call_oi: float  # Call open interest
    put_oi: float  # Put open interest
    call_gamma: float = 0.0  # Gamma of call at this strike
    put_gamma: float = 0.0  # Gamma of put at this strike
    call_delta: float = 0.0
    put_delta: float = 0.0

    @property
    def total_oi(self) -> float:
        return self.call_oi + self.put_oi

    @property
    def net_gamma(self) -> float:
        """Net gamma exposure (dealer perspective)."""
        # Dealers are typically short options, so flip sign
        return -(self.call_gamma * self.call_oi + self.put_gamma * self.put_oi)


class GammaFieldCalculator:
    """Calculates steering forces from options gamma exposure.

    Options market makers create steering effects through delta hedging:
    - When dealers are short gamma (common), they must:
      - Buy when price rises (hedge short calls going ITM)
      - Sell when price falls (hedge short puts going ITM)
    - This creates a stabilizing effect that dampens moves

    - When dealers are long gamma (rarer), the opposite occurs:
      - They sell into rises, buy into dips
      - This can amplify moves (destabilizing)

    Max pain is the price at which the most options expire worthless,
    creating a weak gravitational attraction near expiry.
    """

    def __init__(self, config: Optional[GammaConfig] = None):
        """Initialize calculator with config.

        Args:
            config: GammaConfig with hyperparameters
        """
        self.config = config or GammaConfig()
        self._strikes: List[OptionStrike] = []
        self._max_pain_cache: Dict[datetime, float] = {}

    def update_options_chain(self, strikes: List[OptionStrike]):
        """Update options chain data.

        Args:
            strikes: List of OptionStrike objects
        """
        self._strikes = [s for s in strikes if s.total_oi >= self.config.min_open_interest]
        self._max_pain_cache = {}  # Invalidate cache

    def _black_scholes_gamma(
        self,
        S: float,  # Spot price
        K: float,  # Strike price
        T: float,  # Time to expiry (years)
        r: float = 0.05,  # Risk-free rate
        sigma: float = 0.5,  # Implied volatility
    ) -> float:
        """Calculate Black-Scholes gamma."""
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        return gamma

    def calculate_net_gamma_exposure(
        self,
        current_price: float,
        implied_vol: float = 0.5,
        risk_free_rate: float = 0.05,
    ) -> float:
        """Calculate net gamma exposure across all strikes.

        Args:
            current_price: Current spot price
            implied_vol: Implied volatility (annualized)
            risk_free_rate: Risk-free interest rate

        Returns:
            Net gamma exposure (negative = dealers short gamma = stabilizing)
        """
        if not self._strikes:
            return 0.0

        net_gamma = 0.0
        now = datetime.now()

        for strike in self._strikes:
            # Time to expiry in years
            time_to_expiry = (strike.expiry - now).total_seconds() / (365.25 * 24 * 3600)

            if time_to_expiry <= 0:
                continue

            # Calculate gamma at this strike
            gamma = self._black_scholes_gamma(
                S=current_price,
                K=strike.strike,
                T=time_to_expiry,
                r=risk_free_rate,
                sigma=implied_vol,
            )

            # Weight by open interest and time relevance
            time_relevance = np.exp(-time_to_expiry * 365 / self.config.expiry_relevance_days
                                    * self.config.expiry_decay_rate)

            # Dealers are typically short, so their gamma exposure is negative of OI
            strike_gamma = -gamma * strike.total_oi * time_relevance
            net_gamma += strike_gamma

        return net_gamma

    def calculate_max_pain(
        self,
        expiry: datetime,
        price_range: Tuple[float, float],
        n_points: int = 100,
    ) -> float:
        """Calculate max pain price for a given expiry.

        Max pain is the price at which total option buyer losses are maximized
        (equivalently, where most options expire worthless).

        Args:
            expiry: Expiration date
            price_range: (min_price, max_price) to search
            n_points: Number of price points to evaluate

        Returns:
            Max pain price
        """
        if expiry in self._max_pain_cache:
            return self._max_pain_cache[expiry]

        # Filter strikes for this expiry
        expiry_strikes = [s for s in self._strikes
                         if abs((s.expiry - expiry).total_seconds()) < 86400]  # Same day

        if not expiry_strikes:
            return (price_range[0] + price_range[1]) / 2

        prices = np.linspace(price_range[0], price_range[1], n_points)
        total_pain = np.zeros(n_points)

        for i, price in enumerate(prices):
            pain = 0.0
            for strike in expiry_strikes:
                # Call pain: max(0, price - strike) * call_oi
                call_pain = max(0, price - strike.strike) * strike.call_oi * self.config.max_pain_call_weight

                # Put pain: max(0, strike - price) * put_oi
                put_pain = max(0, strike.strike - price) * strike.put_oi * self.config.max_pain_put_weight

                pain += call_pain + put_pain

            total_pain[i] = pain

        # Max pain is where total pain is minimized (for option buyers)
        max_pain_price = prices[np.argmin(total_pain)]

        self._max_pain_cache[expiry] = max_pain_price
        return max_pain_price

    def calculate_force(
        self,
        current_price: float,
        regime: str = 'ranging',
        implied_vol: float = 0.5,
    ) -> Dict[str, float]:
        """Calculate steering force from gamma exposure.

        Args:
            current_price: Current spot price
            regime: Current market regime
            implied_vol: Implied volatility

        Returns:
            Dict with force components and metadata
        """
        if not self._strikes:
            return {
                'gamma_force': 0.0,
                'net_gamma': 0.0,
                'max_pain_force': 0.0,
                'nearest_max_pain': 0.0,
                'gamma_regime': 'neutral',
            }

        # Get regime multiplier
        regime_mult = {
            'trending': self.config.strength_trending,
            'ranging': self.config.strength_ranging,
            'volatile': self.config.strength_volatile,
        }.get(regime, 1.0)

        # Calculate net gamma
        net_gamma = self.calculate_net_gamma_exposure(current_price, implied_vol)

        # Determine gamma regime
        if net_gamma < -1000:
            gamma_regime = 'short_gamma'  # Stabilizing
            gamma_multiplier = self.config.negative_gamma_stabilizing
        elif net_gamma > 1000:
            gamma_regime = 'long_gamma'  # Destabilizing
            gamma_multiplier = self.config.positive_gamma_destabilizing
        else:
            gamma_regime = 'neutral'
            gamma_multiplier = 1.0

        # Gamma force: affects volatility expectation, not direction
        # Short gamma = expect lower realized vol (mean reversion)
        # Long gamma = expect higher realized vol (momentum)
        gamma_force = np.sign(net_gamma) * min(abs(net_gamma) * self.config.gamma_strength, 0.01)
        gamma_force *= regime_mult * gamma_multiplier

        # Max pain attraction (near expiry only)
        max_pain_force = 0.0
        nearest_max_pain = 0.0
        now = datetime.now()

        # Find nearest relevant expiry
        upcoming_expiries = []
        for strike in self._strikes:
            days_to_expiry = (strike.expiry - now).days
            if 0 < days_to_expiry <= self.config.expiry_relevance_days:
                upcoming_expiries.append(strike.expiry)

        if upcoming_expiries:
            nearest_expiry = min(upcoming_expiries, key=lambda e: abs((e - now).total_seconds()))
            days_to_nearest = (nearest_expiry - now).days

            # Calculate max pain for nearest expiry
            max_pain = self.calculate_max_pain(
                nearest_expiry,
                (current_price * 0.8, current_price * 1.2),
            )
            nearest_max_pain = max_pain

            # Distance to max pain
            distance_to_max_pain = (max_pain - current_price) / current_price

            # Force increases as expiry approaches and decreases with distance
            if abs(distance_to_max_pain) < self.config.max_pain_influence_zone:
                time_factor = 1.0 / (days_to_nearest + 1)  # Stronger near expiry
                distance_factor = 1.0 - abs(distance_to_max_pain) / self.config.max_pain_influence_zone

                max_pain_force = (
                    np.sign(distance_to_max_pain) *
                    self.config.max_pain_strength *
                    time_factor *
                    distance_factor *
                    regime_mult
                )

        return {
            'gamma_force': gamma_force,
            'net_gamma': net_gamma,
            'max_pain_force': max_pain_force,
            'nearest_max_pain': nearest_max_pain,
            'gamma_regime': gamma_regime,
            'total_gamma_force': gamma_force + max_pain_force,
        }

    def get_gamma_features(
        self,
        current_price: float,
        regime: str = 'ranging',
        implied_vol: float = 0.5,
    ) -> Dict[str, float]:
        """Get gamma-related features for ML model."""
        force_result = self.calculate_force(current_price, regime, implied_vol)

        features = {
            'gamma_force': force_result['gamma_force'],
            'gamma_net': force_result['net_gamma'],
            'gamma_max_pain_force': force_result['max_pain_force'],
            'gamma_max_pain_distance': (force_result['nearest_max_pain'] - current_price) / current_price
                if force_result['nearest_max_pain'] > 0 else 0,
            'gamma_regime_short': 1.0 if force_result['gamma_regime'] == 'short_gamma' else 0.0,
            'gamma_regime_long': 1.0 if force_result['gamma_regime'] == 'long_gamma' else 0.0,
            'gamma_total_force': force_result['total_gamma_force'],
        }

        # Add aggregate strike metrics
        if self._strikes:
            total_call_oi = sum(s.call_oi for s in self._strikes)
            total_put_oi = sum(s.put_oi for s in self._strikes)
            features['gamma_put_call_ratio'] = total_put_oi / (total_call_oi + 1) if total_call_oi > 0 else 1.0

            # Find highest OI strikes
            sorted_strikes = sorted(self._strikes, key=lambda s: s.total_oi, reverse=True)
            if sorted_strikes:
                features['gamma_highest_oi_strike'] = sorted_strikes[0].strike
                features['gamma_highest_oi_distance'] = (sorted_strikes[0].strike - current_price) / current_price
        else:
            features['gamma_put_call_ratio'] = 1.0
            features['gamma_highest_oi_strike'] = current_price
            features['gamma_highest_oi_distance'] = 0.0

        return features

    def estimate_from_futures_oi(
        self,
        current_price: float,
        futures_oi: float,
        typical_leverage: float = 10.0,
    ):
        """Estimate options-like gamma effect from futures OI.

        When actual options data isn't available, we can create
        synthetic gamma estimates from futures positioning.

        Args:
            current_price: Current price
            futures_oi: Total futures open interest
            typical_leverage: Average leverage assumption
        """
        # Create synthetic strikes at key levels
        self._strikes = []
        now = datetime.now()

        # Weekly and monthly expiries
        expiries = [
            now + timedelta(days=7),
            now + timedelta(days=14),
            now + timedelta(days=30),
        ]

        # Create strikes at round numbers and percentages
        strike_pcts = [-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10]

        for expiry in expiries:
            for pct in strike_pcts:
                strike = current_price * (1 + pct)
                strike = round(strike / 100) * 100  # Round to nearest 100

                # Estimate OI based on distance from current price
                distance = abs(pct)
                oi_estimate = futures_oi * 0.1 * (1.0 / (1 + distance * 10))

                # Split between calls and puts based on strike position
                if pct < 0:
                    # Below price: more puts
                    call_oi = oi_estimate * 0.3
                    put_oi = oi_estimate * 0.7
                else:
                    # Above price: more calls
                    call_oi = oi_estimate * 0.7
                    put_oi = oi_estimate * 0.3

                self._strikes.append(OptionStrike(
                    strike=strike,
                    expiry=expiry,
                    call_oi=call_oi,
                    put_oi=put_oi,
                ))

        self._max_pain_cache = {}

    def clear(self):
        """Clear all options data."""
        self._strikes = []
        self._max_pain_cache = {}


def get_gamma_hyperparameters() -> List[Dict]:
    """Get hyperparameter definitions for improvement loop."""
    return [
        {
            'name': 'gamma_strength',
            'path': 'particle.gamma.gamma_strength',
            'current_value': 0.001,
            'candidates': [0.0005, 0.001, 0.002, 0.005],
            'variable_type': 'continuous',
        },
        {
            'name': 'gamma_max_pain_strength',
            'path': 'particle.gamma.max_pain_strength',
            'current_value': 0.0005,
            'candidates': [0.0002, 0.0005, 0.001, 0.002],
            'variable_type': 'continuous',
        },
        {
            'name': 'gamma_expiry_relevance_days',
            'path': 'particle.gamma.expiry_relevance_days',
            'current_value': 7.0,
            'candidates': [3.0, 5.0, 7.0, 14.0],
            'variable_type': 'continuous',
        },
        {
            'name': 'gamma_max_pain_influence_zone',
            'path': 'particle.gamma.max_pain_influence_zone',
            'current_value': 0.03,
            'candidates': [0.02, 0.03, 0.05, 0.07],
            'variable_type': 'continuous',
        },
        {
            'name': 'gamma_negative_stabilizing',
            'path': 'particle.gamma.negative_gamma_stabilizing',
            'current_value': 1.0,
            'candidates': [0.5, 0.8, 1.0, 1.2, 1.5],
            'variable_type': 'continuous',
        },
        {
            'name': 'gamma_positive_destabilizing',
            'path': 'particle.gamma.positive_gamma_destabilizing',
            'current_value': 1.2,
            'candidates': [0.8, 1.0, 1.2, 1.5, 2.0],
            'variable_type': 'continuous',
        },
    ]
