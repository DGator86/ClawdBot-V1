"""Cross-Asset Coupling Fields.

Models the correlation between Bitcoin and macro assets:
- SPX (S&P 500): Risk-on/risk-off correlation
- DXY (Dollar Index): Inverse correlation with USD strength
- Gold: Alternative store of value correlation
- VIX: Volatility regime indicator

dP_BTC/dt = β_SPX * dP_SPX/dt + β_DXY * dP_DXY/dt + crypto-specific forces

All parameters are exposed as hyperparameters for ML tuning.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


@dataclass
class MacroCouplingConfig:
    """Hyperparameters for cross-asset coupling.

    All parameters can be tuned by the improvement loop.
    """
    # Base coupling strengths (betas)
    beta_spx: float = 0.3  # Correlation with S&P 500
    beta_dxy: float = -0.2  # Inverse correlation with dollar
    beta_gold: float = 0.1  # Correlation with gold
    beta_vix: float = -0.15  # Inverse correlation with VIX

    # Regime-dependent beta adjustments
    beta_mult_trending: float = 0.8  # Reduce macro influence when crypto trending
    beta_mult_ranging: float = 1.2  # Increase macro influence when ranging
    beta_mult_volatile: float = 1.5  # Strong macro influence in volatile periods

    # Rolling window for beta estimation
    beta_lookback_hours: int = 168  # 1 week
    beta_min_samples: int = 50  # Minimum samples for reliable beta

    # Correlation regime detection
    correlation_threshold_high: float = 0.6  # High correlation regime
    correlation_threshold_low: float = 0.2  # Decoupled regime

    # Force calculation
    macro_force_strength: float = 0.5  # Overall strength multiplier
    macro_force_max: float = 0.02  # Maximum macro force (2%)

    # Lagged effects
    spx_lag_minutes: int = 5  # SPX leads BTC by this amount
    dxy_lag_minutes: int = 15  # DXY effects lag more

    # Session weights (macro assets trade specific hours)
    us_session_weight: float = 1.5  # Stronger coupling during US hours
    asia_session_weight: float = 0.5  # Weaker coupling during Asia
    europe_session_weight: float = 1.0  # Moderate during Europe

    # VIX regime thresholds
    vix_low_threshold: float = 15.0  # Low vol regime
    vix_high_threshold: float = 25.0  # High vol regime
    vix_extreme_threshold: float = 35.0  # Extreme vol regime


@dataclass
class MacroAssetData:
    """Data for a macro asset."""
    symbol: str
    prices: pd.Series  # Price series with datetime index
    returns: Optional[pd.Series] = None

    def __post_init__(self):
        if self.returns is None and len(self.prices) > 1:
            self.returns = self.prices.pct_change()


class CrossAssetCoupling:
    """Models cross-asset correlations and their steering effects.

    Bitcoin's correlation with macro assets varies over time:
    - High correlation: BTC trades as risk asset, follows SPX
    - Decoupled: BTC trades on crypto-specific factors
    - Inverse correlation: BTC acts as hedge/alternative

    The coupling creates steering forces when macro assets move.
    """

    SUPPORTED_ASSETS = ['SPX', 'DXY', 'GOLD', 'VIX', 'QQQ', 'TLT']

    def __init__(self, config: Optional[MacroCouplingConfig] = None):
        """Initialize with config.

        Args:
            config: MacroCouplingConfig with hyperparameters
        """
        self.config = config or MacroCouplingConfig()
        self._macro_data: Dict[str, MacroAssetData] = {}
        self._rolling_betas: Dict[str, float] = {}
        self._rolling_correlations: Dict[str, float] = {}

    def update_macro_data(self, asset: str, data: MacroAssetData):
        """Update data for a macro asset.

        Args:
            asset: Asset symbol (SPX, DXY, GOLD, VIX)
            data: MacroAssetData object
        """
        self._macro_data[asset.upper()] = data
        self._rolling_betas = {}  # Invalidate cache
        self._rolling_correlations = {}

    def calculate_rolling_beta(
        self,
        btc_returns: pd.Series,
        macro_returns: pd.Series,
        window: Optional[int] = None,
    ) -> float:
        """Calculate rolling beta coefficient.

        β = Cov(BTC, Macro) / Var(Macro)

        Args:
            btc_returns: BTC return series
            macro_returns: Macro asset return series
            window: Lookback window (uses config default if None)

        Returns:
            Beta coefficient
        """
        window = window or self.config.beta_lookback_hours

        # Align series
        aligned = pd.DataFrame({
            'btc': btc_returns,
            'macro': macro_returns
        }).dropna()

        if len(aligned) < self.config.beta_min_samples:
            return 0.0

        # Use recent window
        recent = aligned.tail(window)

        # Calculate beta
        cov = recent['btc'].cov(recent['macro'])
        var = recent['macro'].var()

        if var == 0 or np.isnan(var):
            return 0.0

        return cov / var

    def calculate_rolling_correlation(
        self,
        btc_returns: pd.Series,
        macro_returns: pd.Series,
        window: Optional[int] = None,
    ) -> float:
        """Calculate rolling correlation.

        Args:
            btc_returns: BTC return series
            macro_returns: Macro asset return series
            window: Lookback window

        Returns:
            Correlation coefficient
        """
        window = window or self.config.beta_lookback_hours

        aligned = pd.DataFrame({
            'btc': btc_returns,
            'macro': macro_returns
        }).dropna()

        if len(aligned) < self.config.beta_min_samples:
            return 0.0

        recent = aligned.tail(window)
        corr = recent['btc'].corr(recent['macro'])

        return corr if not np.isnan(corr) else 0.0

    def get_session_weight(self, current_time: datetime) -> float:
        """Get session-based coupling weight.

        Args:
            current_time: Current datetime (UTC)

        Returns:
            Session weight multiplier
        """
        hour = current_time.hour

        # US session: 13:30-21:00 UTC (9:30 AM - 4 PM ET)
        if 13 <= hour < 21:
            return self.config.us_session_weight

        # Europe session: 7:00-15:30 UTC
        if 7 <= hour < 16:
            return self.config.europe_session_weight

        # Asia session: 0:00-8:00 UTC
        return self.config.asia_session_weight

    def get_vix_regime(self, vix_level: float) -> str:
        """Determine VIX regime.

        Args:
            vix_level: Current VIX level

        Returns:
            Regime string
        """
        if vix_level >= self.config.vix_extreme_threshold:
            return 'extreme'
        elif vix_level >= self.config.vix_high_threshold:
            return 'high'
        elif vix_level <= self.config.vix_low_threshold:
            return 'low'
        else:
            return 'normal'

    def calculate_macro_drift(
        self,
        btc_returns: pd.Series,
        current_time: datetime,
        regime: str = 'ranging',
    ) -> Dict[str, float]:
        """Calculate drift contribution from macro assets.

        Args:
            btc_returns: Recent BTC returns for beta calculation
            current_time: Current datetime
            regime: Current BTC market regime

        Returns:
            Dict with drift components
        """
        if not self._macro_data:
            return {
                'macro_drift': 0.0,
                'spx_contribution': 0.0,
                'dxy_contribution': 0.0,
                'gold_contribution': 0.0,
                'vix_contribution': 0.0,
                'correlation_regime': 'unknown',
            }

        # Get regime multiplier
        regime_mult = {
            'trending': self.config.beta_mult_trending,
            'ranging': self.config.beta_mult_ranging,
            'volatile': self.config.beta_mult_volatile,
        }.get(regime, 1.0)

        # Get session weight
        session_weight = self.get_session_weight(current_time)

        total_drift = 0.0
        contributions = {}
        correlations = {}

        # Process each macro asset
        asset_configs = {
            'SPX': ('beta_spx', self.config.spx_lag_minutes),
            'DXY': ('beta_dxy', self.config.dxy_lag_minutes),
            'GOLD': ('beta_gold', 0),
            'VIX': ('beta_vix', 0),
        }

        for asset, (beta_attr, lag_minutes) in asset_configs.items():
            if asset not in self._macro_data:
                contributions[f'{asset.lower()}_contribution'] = 0.0
                continue

            macro_data = self._macro_data[asset]
            if macro_data.returns is None or len(macro_data.returns) < 2:
                contributions[f'{asset.lower()}_contribution'] = 0.0
                continue

            # Calculate rolling beta if not cached
            if asset not in self._rolling_betas:
                self._rolling_betas[asset] = self.calculate_rolling_beta(
                    btc_returns, macro_data.returns
                )
                self._rolling_correlations[asset] = self.calculate_rolling_correlation(
                    btc_returns, macro_data.returns
                )

            # Get recent macro return (with lag if applicable)
            if lag_minutes > 0 and len(macro_data.returns) > lag_minutes:
                macro_return = macro_data.returns.iloc[-(1 + lag_minutes)]
            else:
                macro_return = macro_data.returns.iloc[-1]

            if pd.isna(macro_return):
                macro_return = 0.0

            # Get base beta from config
            base_beta = getattr(self.config, beta_attr)

            # Blend with rolling beta (adaptive)
            rolling_beta = self._rolling_betas[asset]
            blended_beta = 0.7 * base_beta + 0.3 * rolling_beta

            # Calculate contribution
            contribution = blended_beta * macro_return * regime_mult * session_weight
            contributions[f'{asset.lower()}_contribution'] = contribution
            correlations[asset] = self._rolling_correlations[asset]

            total_drift += contribution

        # Determine correlation regime
        avg_correlation = np.mean([abs(c) for c in correlations.values()]) if correlations else 0
        if avg_correlation >= self.config.correlation_threshold_high:
            corr_regime = 'coupled'
        elif avg_correlation <= self.config.correlation_threshold_low:
            corr_regime = 'decoupled'
        else:
            corr_regime = 'moderate'

        # Apply overall strength and cap
        total_drift *= self.config.macro_force_strength
        total_drift = np.clip(total_drift, -self.config.macro_force_max, self.config.macro_force_max)

        result = {
            'macro_drift': total_drift,
            'correlation_regime': corr_regime,
            'session_weight': session_weight,
            **contributions,
        }

        # Add correlations
        for asset, corr in correlations.items():
            result[f'{asset.lower()}_correlation'] = corr

        # Add VIX regime if available
        if 'VIX' in self._macro_data:
            vix_data = self._macro_data['VIX']
            if len(vix_data.prices) > 0:
                vix_level = vix_data.prices.iloc[-1]
                result['vix_level'] = vix_level
                result['vix_regime'] = self.get_vix_regime(vix_level)

        return result

    def get_macro_features(
        self,
        btc_returns: pd.Series,
        current_time: datetime,
        regime: str = 'ranging',
    ) -> Dict[str, float]:
        """Get macro-related features for ML model."""
        drift_result = self.calculate_macro_drift(btc_returns, current_time, regime)

        features = {
            'macro_drift': drift_result['macro_drift'],
            'macro_spx_contrib': drift_result.get('spx_contribution', 0),
            'macro_dxy_contrib': drift_result.get('dxy_contribution', 0),
            'macro_gold_contrib': drift_result.get('gold_contribution', 0),
            'macro_vix_contrib': drift_result.get('vix_contribution', 0),
            'macro_session_weight': drift_result.get('session_weight', 1.0),
            'macro_coupled': 1.0 if drift_result.get('correlation_regime') == 'coupled' else 0.0,
            'macro_decoupled': 1.0 if drift_result.get('correlation_regime') == 'decoupled' else 0.0,
        }

        # Add correlations
        for asset in ['spx', 'dxy', 'gold', 'vix']:
            features[f'macro_{asset}_corr'] = drift_result.get(f'{asset}_correlation', 0)

        # Add VIX features
        if 'vix_level' in drift_result:
            features['macro_vix_level'] = drift_result['vix_level']
            vix_regime = drift_result.get('vix_regime', 'normal')
            features['macro_vix_high'] = 1.0 if vix_regime in ['high', 'extreme'] else 0.0
            features['macro_vix_low'] = 1.0 if vix_regime == 'low' else 0.0
        else:
            features['macro_vix_level'] = 20.0  # Default
            features['macro_vix_high'] = 0.0
            features['macro_vix_low'] = 0.0

        # Add rolling betas
        for asset in ['SPX', 'DXY', 'GOLD', 'VIX']:
            features[f'macro_{asset.lower()}_beta'] = self._rolling_betas.get(asset, 0)

        return features

    def simulate_macro_data(
        self,
        btc_prices: pd.Series,
        correlation_spx: float = 0.5,
        correlation_dxy: float = -0.3,
    ):
        """Simulate macro data for testing when real data unavailable.

        Creates synthetic macro asset prices that have specified
        correlations with BTC.

        Args:
            btc_prices: BTC price series
            correlation_spx: Target correlation with SPX
            correlation_dxy: Target correlation with DXY
        """
        btc_returns = btc_prices.pct_change().dropna()
        n = len(btc_returns)

        if n < 10:
            return

        # Generate correlated noise
        def generate_correlated_returns(target_corr: float, base_level: float = 1.0):
            noise = np.random.normal(0, 0.01, n)
            correlated = target_corr * btc_returns.values + np.sqrt(1 - target_corr**2) * noise
            prices = base_level * np.cumprod(1 + correlated)
            return pd.Series(prices, index=btc_returns.index)

        # SPX (starts around 5000)
        spx_prices = generate_correlated_returns(correlation_spx, 5000)
        self._macro_data['SPX'] = MacroAssetData(
            symbol='SPX',
            prices=spx_prices,
        )

        # DXY (starts around 105)
        dxy_prices = generate_correlated_returns(correlation_dxy, 105)
        self._macro_data['DXY'] = MacroAssetData(
            symbol='DXY',
            prices=dxy_prices,
        )

        # GOLD (starts around 2000)
        gold_prices = generate_correlated_returns(0.2, 2000)
        self._macro_data['GOLD'] = MacroAssetData(
            symbol='GOLD',
            prices=gold_prices,
        )

        # VIX (mean-reverting around 18)
        vix_base = 18 + np.cumsum(np.random.normal(0, 0.5, n))
        vix_prices = pd.Series(np.clip(vix_base, 10, 50), index=btc_returns.index)
        self._macro_data['VIX'] = MacroAssetData(
            symbol='VIX',
            prices=vix_prices,
        )

    def clear(self):
        """Clear all macro data."""
        self._macro_data = {}
        self._rolling_betas = {}
        self._rolling_correlations = {}


def get_macro_hyperparameters() -> List[Dict]:
    """Get hyperparameter definitions for improvement loop."""
    return [
        {
            'name': 'macro_beta_spx',
            'path': 'particle.macro.beta_spx',
            'current_value': 0.3,
            'candidates': [0.1, 0.2, 0.3, 0.4, 0.5],
            'variable_type': 'continuous',
        },
        {
            'name': 'macro_beta_dxy',
            'path': 'particle.macro.beta_dxy',
            'current_value': -0.2,
            'candidates': [-0.4, -0.3, -0.2, -0.1, 0.0],
            'variable_type': 'continuous',
        },
        {
            'name': 'macro_beta_gold',
            'path': 'particle.macro.beta_gold',
            'current_value': 0.1,
            'candidates': [0.0, 0.1, 0.2, 0.3],
            'variable_type': 'continuous',
        },
        {
            'name': 'macro_beta_vix',
            'path': 'particle.macro.beta_vix',
            'current_value': -0.15,
            'candidates': [-0.3, -0.2, -0.15, -0.1, 0.0],
            'variable_type': 'continuous',
        },
        {
            'name': 'macro_force_strength',
            'path': 'particle.macro.macro_force_strength',
            'current_value': 0.5,
            'candidates': [0.2, 0.3, 0.5, 0.7, 1.0],
            'variable_type': 'continuous',
        },
        {
            'name': 'macro_beta_lookback_hours',
            'path': 'particle.macro.beta_lookback_hours',
            'current_value': 168,
            'candidates': [72, 120, 168, 336],
            'variable_type': 'discrete',
        },
        {
            'name': 'macro_us_session_weight',
            'path': 'particle.macro.us_session_weight',
            'current_value': 1.5,
            'candidates': [1.0, 1.25, 1.5, 1.75, 2.0],
            'variable_type': 'continuous',
        },
        {
            'name': 'macro_correlation_threshold_high',
            'path': 'particle.macro.correlation_threshold_high',
            'current_value': 0.6,
            'candidates': [0.4, 0.5, 0.6, 0.7],
            'variable_type': 'continuous',
        },
    ]
