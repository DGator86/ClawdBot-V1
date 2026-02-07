"""Crypto Price-as-a-Particle: Strengthened Indicator Integration.

Treats log-price x_tau = log(p_tau) as a particle in event-time tau,
evolving under the SDE:

    dx_tau = mu_tau * dtau - grad(V(x_tau)) * dtau + sigma_tau * dW_tau + J_tau

where:
    V = V_liqbook + V_AVWAP + V_MA + V_liq   (total potential)
    sigma_tau = f(BB_width, realized_vol, spread, depth)  (diffusion)
    mu_tau = mu(MA_slope, regime, funding)  (drift)
    lambda_tau = lambda(delta_OI, liq_density, squeeze_state)  (jump hazard)

Components (in implementation priority order):
1. AVWAP anchor set (impulse start + major flush + weekly open)
2. Bollinger-driven sigma_tau + squeeze detector
3. MA-well + slope drift
4. Funding + OI hazard multiplier
5. RSI throttle (impulse efficiency)
6. Ichimoku regime gate (diffusive/ballistic topology)
7. CVD (cumulative volume delta) / aggressive flow

All indicators are reinterpreted as physics rather than superstition:
- AVWAP = mass anchors / fair-value wells
- EMA/SMA = drift surfaces / restoring forces
- Bollinger = volatility curvature / entropy pressure
- RSI = momentum saturation / impulse efficiency
- Ichimoku = regime topology / allowed zones
- Funding = continuous external force
- OI = stored leverage energy
- Liquidation density = potential cliffs
- CVD = impulse directionality
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# 1. AVWAP Anchor Set -- Mass Anchors + Fair-Value Wells (Tier 1)
# ---------------------------------------------------------------------------

class AnchorType(Enum):
    """Types of anchored VWAP."""
    IMPULSE_START = "impulse_start"
    MAJOR_FLUSH = "major_flush"
    FUNDING_FLIP = "funding_flip"
    WEEKLY_OPEN = "weekly_open"
    DAILY_OPEN = "daily_open"
    CUSTOM = "custom"


@dataclass
class AVWAPAnchor:
    """A single anchored VWAP reference."""
    anchor_type: AnchorType
    anchor_bar_idx: int
    avwap_value: float
    cumulative_volume: float
    alpha: float = 1.0  # Strength multiplier (increases for high-leverage flushes)


@dataclass
class AVWAPConfig:
    """Configuration for AVWAP anchor detection and well strength."""
    # Impulse detection
    impulse_threshold_sigma: float = 2.0  # Std devs for impulse detection
    impulse_lookback: int = 20  # Bars to compute vol for impulse threshold
    # Flush detection
    flush_volume_mult: float = 3.0  # Volume multiple for flush detection
    flush_return_sigma: float = 2.5  # Return threshold for flush
    # Well strength
    base_alpha: float = 1.0
    flush_alpha_boost: float = 2.0  # Extra alpha for flush anchors
    volume_decay_halflife: int = 200  # Bars for anchor relevance decay
    # Max anchors
    max_anchors: int = 8


class AVWAPAnchorSet:
    """Manages anchored VWAP wells for crypto price-as-particle model.

    Each AVWAP is a center-of-mass well:
        V_AVWAP(x) = sum_i alpha_i * vol_i * |x - log(AVWAP_i)|

    alpha_i increases when the anchor corresponds to a high-leverage flush
    or major OI reset. Multiple AVWAPs competing create orbital trapping
    between wells.

    Use-case: "Where is fair value migrating?" and "Which well dominates?"
    """

    def __init__(self, config: Optional[AVWAPConfig] = None):
        self.config = config or AVWAPConfig()
        self._anchors: List[AVWAPAnchor] = []

    @property
    def anchors(self) -> List[AVWAPAnchor]:
        return list(self._anchors)

    def detect_and_compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anchor points and compute AVWAPs for each bar.

        Args:
            df: DataFrame with columns: close, high, low, volume, returns.
                Must be single-symbol, sorted by bar_idx.

        Returns:
            DataFrame with AVWAP features appended.
        """
        result = df.copy()
        n = len(result)
        if n < 10:
            result["avwap_dominant"] = result["close"]
            result["avwap_distance"] = 0.0
            result["avwap_well_strength"] = 0.0
            result["avwap_n_competing"] = 0
            result["avwap_potential"] = 0.0
            return result

        close = result["close"].values
        volume = result["volume"].values
        returns = result["returns"].values if "returns" in result.columns else np.diff(np.log(close + 1e-9), prepend=np.log(close[0] + 1e-9))

        # Compute rolling vol for impulse detection
        vol_series = pd.Series(returns).rolling(
            self.config.impulse_lookback, min_periods=5
        ).std().values

        # Detect anchor points
        self._anchors = []
        self._detect_impulse_starts(close, returns, volume, vol_series)
        self._detect_major_flushes(close, returns, volume, vol_series)

        # Add weekly/daily open anchors (use bar index heuristic)
        self._detect_periodic_opens(close, volume, n)

        # Limit anchors
        if len(self._anchors) > self.config.max_anchors:
            # Keep strongest anchors
            self._anchors.sort(key=lambda a: a.alpha * a.cumulative_volume, reverse=True)
            self._anchors = self._anchors[:self.config.max_anchors]

        # Compute running AVWAPs from each anchor point
        avwap_values = np.full((len(self._anchors), n), np.nan)
        for i, anchor in enumerate(self._anchors):
            cum_pv = 0.0
            cum_v = 0.0
            for j in range(anchor.anchor_bar_idx, n):
                cum_pv += close[j] * volume[j]
                cum_v += volume[j]
                if cum_v > 0:
                    avwap_values[i, j] = cum_pv / cum_v
            anchor.cumulative_volume = cum_v

        # Compute per-bar features
        dominant_avwap = np.full(n, np.nan)
        avwap_dist = np.zeros(n)
        well_strength = np.zeros(n)
        n_competing = np.zeros(n, dtype=int)
        potential = np.zeros(n)

        for j in range(n):
            valid_anchors = []
            for i, anchor in enumerate(self._anchors):
                if not np.isnan(avwap_values[i, j]):
                    bars_since = j - anchor.anchor_bar_idx
                    decay = 0.5 ** (bars_since / max(self.config.volume_decay_halflife, 1))
                    weight = anchor.alpha * decay
                    valid_anchors.append((avwap_values[i, j], weight))

            if not valid_anchors:
                dominant_avwap[j] = close[j]
                continue

            # Find dominant well (highest weight)
            best_avwap, best_weight = max(valid_anchors, key=lambda x: x[1])
            dominant_avwap[j] = best_avwap

            # Distance from dominant AVWAP
            log_price = np.log(close[j] + 1e-9)
            log_avwap = np.log(best_avwap + 1e-9)
            avwap_dist[j] = log_price - log_avwap
            well_strength[j] = best_weight

            # Count competing wells (within 1% of each other)
            competing = 0
            for av, w in valid_anchors:
                if abs(np.log(av + 1e-9) - log_avwap) < 0.01 and w > 0.3 * best_weight:
                    competing += 1
            n_competing[j] = competing

            # Total potential from all wells
            for av, w in valid_anchors:
                dist = abs(log_price - np.log(av + 1e-9))
                potential[j] += w * dist

        result["avwap_dominant"] = dominant_avwap
        result["avwap_distance"] = avwap_dist
        result["avwap_well_strength"] = well_strength
        result["avwap_n_competing"] = n_competing
        result["avwap_potential"] = potential

        return result

    def _detect_impulse_starts(
        self, close: np.ndarray, returns: np.ndarray,
        volume: np.ndarray, vol_series: np.ndarray,
    ):
        """Detect bars where impulse moves begin."""
        n = len(close)
        threshold = self.config.impulse_threshold_sigma

        for i in range(self.config.impulse_lookback + 1, n - 1):
            if np.isnan(vol_series[i]) or vol_series[i] < 1e-9:
                continue
            # Impulse: large return followed by continuation
            if abs(returns[i]) > threshold * vol_series[i]:
                # Check if next bar continues the direction
                if i + 1 < n and np.sign(returns[i + 1]) == np.sign(returns[i]):
                    self._anchors.append(AVWAPAnchor(
                        anchor_type=AnchorType.IMPULSE_START,
                        anchor_bar_idx=max(0, i - 1),  # Anchor at bar before impulse
                        avwap_value=close[max(0, i - 1)],
                        cumulative_volume=volume[i],
                        alpha=self.config.base_alpha,
                    ))

    def _detect_major_flushes(
        self, close: np.ndarray, returns: np.ndarray,
        volume: np.ndarray, vol_series: np.ndarray,
    ):
        """Detect high-volume liquidation flush events."""
        n = len(close)
        lookback = self.config.impulse_lookback

        vol_ma = pd.Series(volume).rolling(lookback, min_periods=5).mean().values

        for i in range(lookback + 1, n):
            if np.isnan(vol_series[i]) or vol_series[i] < 1e-9 or np.isnan(vol_ma[i]):
                continue
            # Flush: extreme return + extreme volume
            is_extreme_return = abs(returns[i]) > self.config.flush_return_sigma * vol_series[i]
            is_extreme_volume = volume[i] > self.config.flush_volume_mult * vol_ma[i]

            if is_extreme_return and is_extreme_volume:
                self._anchors.append(AVWAPAnchor(
                    anchor_type=AnchorType.MAJOR_FLUSH,
                    anchor_bar_idx=i,
                    avwap_value=close[i],
                    cumulative_volume=volume[i],
                    alpha=self.config.base_alpha + self.config.flush_alpha_boost,
                ))

    def _detect_periodic_opens(self, close: np.ndarray, volume: np.ndarray, n: int):
        """Add periodic open anchors (weekly-ish and daily-ish based on bar count)."""
        # Heuristic: assume ~200 bars/day for D0 domain. Weekly ~ 1400 bars.
        # We just place anchors at regular intervals as structural reference.
        daily_approx = max(n // 7, 50)  # Rough daily interval
        weekly_approx = daily_approx * 7

        # Weekly opens
        for i in range(0, n, max(weekly_approx, 1)):
            self._anchors.append(AVWAPAnchor(
                anchor_type=AnchorType.WEEKLY_OPEN,
                anchor_bar_idx=i,
                avwap_value=close[i],
                cumulative_volume=volume[i] if i < len(volume) else 0,
                alpha=self.config.base_alpha * 0.8,
            ))

        # Daily opens (lower weight)
        for i in range(0, n, max(daily_approx, 1)):
            if i % max(weekly_approx, 1) == 0:
                continue  # Skip weekly opens
            self._anchors.append(AVWAPAnchor(
                anchor_type=AnchorType.DAILY_OPEN,
                anchor_bar_idx=i,
                avwap_value=close[i],
                cumulative_volume=volume[i] if i < len(volume) else 0,
                alpha=self.config.base_alpha * 0.4,
            ))


# ---------------------------------------------------------------------------
# 2. Bollinger Bands -- Volatility Curvature + Entropy Pressure (Tier 1)
# ---------------------------------------------------------------------------

@dataclass
class BollingerConfig:
    """Configuration for Bollinger-based diffusion geometry."""
    bb_period: int = 20
    bb_std_mult: float = 2.0
    squeeze_threshold: float = 0.4  # Normalized BB width below this = squeeze
    squeeze_energy_scale: float = 2.0  # How much potential energy a squeeze stores
    expansion_decay: float = 0.8  # How fast expansion energy dissipates


class BollingerDiffusion:
    """Bollinger Bands as volatility curvature and entropy pressure.

    Do NOT treat as "breakout direction." Treat as diffusion geometry:
    - Band width = local sigma_tau
    - Squeeze = stored potential energy
    - Expansion = energy release already underway

    sigma_tau = f(BB_width_tau)

    Bands are soft barriers:
    - Near upper band in squeeze -> higher boundary test probability
    - Escape probability depends on liquidity curvature + liquidation density
    """

    def __init__(self, config: Optional[BollingerConfig] = None):
        self.config = config or BollingerConfig()

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Bollinger-derived diffusion features.

        Args:
            df: DataFrame with close column (single symbol, sorted).

        Returns:
            DataFrame with Bollinger diffusion features.
        """
        result = df.copy()
        n = len(result)
        period = self.config.bb_period

        if n < period:
            result["bb_width"] = 0.0
            result["bb_width_norm"] = 0.5
            result["bb_sigma_tau"] = 0.01
            result["bb_squeeze"] = False
            result["bb_squeeze_energy"] = 0.0
            result["bb_position"] = 0.5
            result["bb_expansion_rate"] = 0.0
            return result

        close = result["close"]

        # Core Bollinger
        sma = close.rolling(period, min_periods=period).mean()
        std = close.rolling(period, min_periods=period).std()
        upper = sma + self.config.bb_std_mult * std
        lower = sma - self.config.bb_std_mult * std

        result["bb_upper"] = upper
        result["bb_lower"] = lower
        result["bb_mid"] = sma

        # Band width (absolute)
        width = upper - lower
        result["bb_width"] = width

        # Normalized band width (relative to price)
        result["bb_width_norm"] = width / (close + 1e-9)

        # Percentile rank of width over longer lookback -> squeeze detection
        width_pctile = result["bb_width_norm"].rolling(
            period * 5, min_periods=period
        ).rank(pct=True)
        result["bb_width_pctile"] = width_pctile

        # sigma_tau derived from BB width
        result["bb_sigma_tau"] = std / (close + 1e-9)

        # Squeeze detection (low percentile of width)
        result["bb_squeeze"] = width_pctile < self.config.squeeze_threshold

        # Squeeze energy (how long squeeze has persisted * compression depth)
        squeeze_bars = result["bb_squeeze"].astype(float).rolling(
            period * 3, min_periods=1
        ).sum()
        compression = (1.0 - width_pctile.clip(lower=0.01))
        result["bb_squeeze_energy"] = (
            squeeze_bars * compression * self.config.squeeze_energy_scale
        )

        # Position within bands (0=lower, 0.5=mid, 1=upper)
        result["bb_position"] = (close - lower) / (width + 1e-9)

        # Expansion rate (dwidth/dt)
        result["bb_expansion_rate"] = result["bb_width_norm"].diff()

        return result


# ---------------------------------------------------------------------------
# 3. MA Wells -- Drift Surfaces + Restoring Forces (Tier 1)
# ---------------------------------------------------------------------------

@dataclass
class MAWellConfig:
    """Configuration for moving average well fields."""
    ema_spans: List[int] = field(default_factory=lambda: [9, 21, 50, 200])
    spring_constant_base: float = 1.0
    slope_drift_scale: float = 1.0
    cluster_threshold: float = 0.005  # When MAs are within 0.5%, they cluster


class MAWellField:
    """EMA/SMA as equilibrium surfaces creating restoring forces.

    V_MA(x) = k_l * |x - log(EMA_l)|

    mu_tau ~ d/dtau * log(EMA_l)

    Key crypto twist: MA clustering on higher timeframe + perps leverage
    = "loaded spring."

    Crypto trends hard and mean-reverts violently. MAs are good if you
    treat them as equilibrium surfaces, not trading signals.
    """

    def __init__(self, config: Optional[MAWellConfig] = None):
        self.config = config or MAWellConfig()

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute MA well features.

        Args:
            df: DataFrame with close (single symbol, sorted).

        Returns:
            DataFrame with MA well features.
        """
        result = df.copy()
        close = result["close"]
        n = len(close)

        if n < max(self.config.ema_spans):
            for span in self.config.ema_spans:
                result[f"ma_well_{span}_dist"] = 0.0
                result[f"ma_well_{span}_force"] = 0.0
                result[f"ma_well_{span}_slope"] = 0.0
            result["ma_well_total_potential"] = 0.0
            result["ma_well_net_drift"] = 0.0
            result["ma_well_cluster"] = False
            result["ma_well_cluster_energy"] = 0.0
            return result

        log_price = np.log(close.values + 1e-9)
        total_potential = np.zeros(n)
        net_drift = np.zeros(n)
        ema_values = {}

        for span in self.config.ema_spans:
            ema = close.ewm(span=span, adjust=False).mean()
            log_ema = np.log(ema.values + 1e-9)
            ema_values[span] = log_ema

            # Distance to EMA (in log space)
            dist = log_price - log_ema
            result[f"ma_well_{span}_dist"] = dist

            # Restoring force: F = -k * (x - x_eq)
            k = self.config.spring_constant_base / np.sqrt(span)  # Weaker for longer MAs
            force = -k * dist
            result[f"ma_well_{span}_force"] = force

            # EMA slope -> background drift direction
            slope = pd.Series(log_ema).diff().values
            result[f"ma_well_{span}_slope"] = slope * self.config.slope_drift_scale

            # Accumulate potential and drift
            total_potential += k * np.abs(dist)
            net_drift += slope * self.config.slope_drift_scale / len(self.config.ema_spans)

        result["ma_well_total_potential"] = total_potential
        result["ma_well_net_drift"] = net_drift

        # MA clustering detection (loaded spring)
        spans = self.config.ema_spans
        if len(spans) >= 2:
            max_spread = np.zeros(n)
            for i in range(len(spans)):
                for j in range(i + 1, len(spans)):
                    spread = np.abs(ema_values[spans[i]] - ema_values[spans[j]])
                    max_spread = np.maximum(max_spread, spread)
            result["ma_well_cluster"] = max_spread < self.config.cluster_threshold
            # Cluster energy: how long MAs have been clustered * compression
            cluster_float = result["ma_well_cluster"].astype(float)
            cluster_duration = cluster_float.rolling(50, min_periods=1).sum()
            result["ma_well_cluster_energy"] = cluster_duration * np.clip(
                self.config.cluster_threshold - max_spread, 0, None
            )
        else:
            result["ma_well_cluster"] = False
            result["ma_well_cluster_energy"] = 0.0

        return result


# ---------------------------------------------------------------------------
# 4. OI Hazard Multiplier -- Stored Leverage Energy
# ---------------------------------------------------------------------------

@dataclass
class OIHazardConfig:
    """Configuration for open interest hazard model."""
    lambda_base: float = 0.02  # Base jump intensity
    oi_change_scale: float = 1.0  # How much OI change amplifies jump hazard
    oi_level_scale: float = 0.5  # How much absolute OI amplifies hazard
    oi_lookback: int = 20  # Bars for OI change computation
    hazard_floor: float = 0.005  # Minimum hazard rate
    hazard_ceiling: float = 0.15  # Maximum hazard rate


class OIHazardModel:
    """Open Interest as stored leverage energy -> jump hazard multiplier.

    Rising OI during a move = leverage piling in -> higher jump risk.

    lambda_tau = lambda_0 * h(delta_OI_tau, OI_tau)

    The hazard function h captures:
    - Rising OI with directional move: unstable leverage buildup
    - Falling OI with move: deleveraging (less jump risk)
    - High absolute OI: larger potential cascade magnitude
    """

    def __init__(self, config: Optional[OIHazardConfig] = None):
        self.config = config or OIHazardConfig()

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute OI hazard features.

        Args:
            df: DataFrame with open_interest column (or zeros if unavailable).

        Returns:
            DataFrame with OI hazard features.
        """
        result = df.copy()
        n = len(result)

        if "open_interest" not in result.columns:
            result["open_interest"] = 0.0

        oi = result["open_interest"].values.astype(float)
        returns = result["returns"].values if "returns" in result.columns else np.zeros(n)

        # OI change
        lookback = self.config.oi_lookback
        oi_change = np.zeros(n)
        oi_change_pct = np.zeros(n)
        for i in range(lookback, n):
            base_oi = oi[i - lookback]
            if base_oi > 0:
                oi_change[i] = oi[i] - base_oi
                oi_change_pct[i] = (oi[i] - base_oi) / base_oi

        result["oi_change"] = oi_change
        result["oi_change_pct"] = oi_change_pct

        # OI percentile (relative to recent history)
        oi_pctile = pd.Series(oi).rolling(lookback * 5, min_periods=lookback).rank(pct=True).values
        result["oi_percentile"] = np.nan_to_num(oi_pctile, nan=0.5)

        # Directional OI buildup (OI rising during directional move)
        cum_returns = pd.Series(returns).rolling(lookback, min_periods=1).sum().values
        directional_buildup = oi_change_pct * np.abs(cum_returns)
        result["oi_directional_buildup"] = directional_buildup

        # Hazard multiplier
        h = np.ones(n)
        # Component 1: OI change amplification
        h += self.config.oi_change_scale * np.abs(oi_change_pct)
        # Component 2: Absolute OI level
        h += self.config.oi_level_scale * np.nan_to_num(oi_pctile, nan=0.0)
        # Component 3: Directional buildup is especially dangerous
        h += np.abs(directional_buildup) * 2.0

        lambda_tau = self.config.lambda_base * h
        lambda_tau = np.clip(lambda_tau, self.config.hazard_floor, self.config.hazard_ceiling)
        result["oi_hazard_rate"] = lambda_tau

        return result


# ---------------------------------------------------------------------------
# 5. RSI Throttle -- Momentum Saturation / Impulse Efficiency (Tier 2)
# ---------------------------------------------------------------------------

@dataclass
class RSIThrottleConfig:
    """Configuration for RSI impulse efficiency throttle."""
    rsi_period: int = 14
    saturation_upper: float = 80.0  # RSI above this -> full suppression
    saturation_lower: float = 20.0  # RSI below this -> full suppression
    neutral_zone: Tuple[float, float] = (40.0, 60.0)  # Full impulse efficiency
    throttle_power: float = 2.0  # How aggressively to suppress at extremes


class RSIThrottle:
    """RSI as momentum saturation / impulse efficiency throttle.

    RSI is NOT a directional signal. It's a proxy for velocity persistence
    + exhaustion.

    - High RSI -> marginal buy impulses produce less displacement
    - Low RSI -> marginal sell impulses produce less displacement

    Scale impulse response:
        J_tau <- J_tau * g(RSI_tau)

    where g suppresses acceleration when RSI is saturated. This prevents
    the classic crypto failure: "RSI overbought" (wrong) vs "RSI saturated
    so impulse efficiency drops" (right).
    """

    def __init__(self, config: Optional[RSIThrottleConfig] = None):
        self.config = config or RSIThrottleConfig()

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute RSI throttle features.

        Args:
            df: DataFrame with returns (single symbol, sorted).

        Returns:
            DataFrame with RSI throttle features.
        """
        result = df.copy()
        n = len(result)
        period = self.config.rsi_period

        if n < period or "returns" not in result.columns:
            result["rsi"] = 50.0
            result["rsi_throttle"] = 1.0
            result["rsi_saturation"] = 0.0
            result["rsi_impulse_efficiency_buy"] = 1.0
            result["rsi_impulse_efficiency_sell"] = 1.0
            return result

        returns = result["returns"]

        # Standard RSI calculation
        gains = returns.where(returns > 0, 0.0)
        losses = (-returns).where(returns < 0, 0.0)
        avg_gain = gains.ewm(span=period, adjust=False).mean()
        avg_loss = losses.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        result["rsi"] = rsi

        # Throttle function g(RSI)
        # g = 1.0 in neutral zone, drops toward 0 at extremes
        rsi_values = rsi.values
        throttle = np.ones(n)
        buy_efficiency = np.ones(n)
        sell_efficiency = np.ones(n)
        saturation = np.zeros(n)

        upper = self.config.saturation_upper
        lower = self.config.saturation_lower
        neutral_lo, neutral_hi = self.config.neutral_zone
        power = self.config.throttle_power

        for i in range(n):
            r = rsi_values[i]
            if np.isnan(r):
                continue

            # Buy impulse efficiency (drops when RSI is high)
            if r > neutral_hi:
                frac = min((r - neutral_hi) / (upper - neutral_hi + 1e-9), 1.0)
                buy_efficiency[i] = max(1.0 - frac ** power, 0.05)
            # Sell impulse efficiency (drops when RSI is low)
            if r < neutral_lo:
                frac = min((neutral_lo - r) / (neutral_lo - lower + 1e-9), 1.0)
                sell_efficiency[i] = max(1.0 - frac ** power, 0.05)

            # Combined throttle (symmetric)
            if r > neutral_hi:
                throttle[i] = buy_efficiency[i]
                saturation[i] = min((r - neutral_hi) / (upper - neutral_hi + 1e-9), 1.0)
            elif r < neutral_lo:
                throttle[i] = sell_efficiency[i]
                saturation[i] = min((neutral_lo - r) / (neutral_lo - lower + 1e-9), 1.0)

        result["rsi_throttle"] = throttle
        result["rsi_saturation"] = saturation
        result["rsi_impulse_efficiency_buy"] = buy_efficiency
        result["rsi_impulse_efficiency_sell"] = sell_efficiency

        return result


# ---------------------------------------------------------------------------
# 6. Ichimoku Regime Gate -- Regime Topology / Allowed Zones (Tier 2)
# ---------------------------------------------------------------------------

class ManifoldRegime(Enum):
    """Regime classification from Ichimoku topology."""
    DIFFUSIVE = "diffusive"    # Inside cloud -> choppy, mean-reversion dominates
    BALLISTIC = "ballistic"    # Outside cloud -> trend forces dominate
    TRANSITION = "transition"  # Near cloud edge -> regime switching


@dataclass
class IchimokuConfig:
    """Configuration for Ichimoku regime topology."""
    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_b_period: int = 52
    cloud_displacement: int = 26
    # Regime thresholds
    transition_zone_pct: float = 0.005  # 0.5% from cloud edge = transition


class IchimokuRegimeGate:
    """Ichimoku as regime topology / allowed zones.

    NOT for trading crosses. For determining which physics apply:

    Regime_tau in {diffusive, ballistic, transition}

    Switch which forces exist based on regime:
    - diffusive: MA wells stronger, volatility reverts
    - ballistic: MA wells weaken, drift dominates
    - transition: both forces present at reduced strength

    Cloud thickness -> regime stability barrier. Thicker cloud means
    harder to switch regimes.
    """

    def __init__(self, config: Optional[IchimokuConfig] = None):
        self.config = config or IchimokuConfig()

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Ichimoku regime features.

        Args:
            df: DataFrame with high, low, close (single symbol, sorted).

        Returns:
            DataFrame with Ichimoku regime features.
        """
        result = df.copy()
        n = len(result)

        min_required = max(self.config.senkou_b_period, self.config.cloud_displacement) + 5
        if n < min_required:
            result["ichi_regime"] = ManifoldRegime.DIFFUSIVE.value
            result["ichi_regime_code"] = 0
            result["ichi_cloud_thickness"] = 0.0
            result["ichi_cloud_distance"] = 0.0
            result["ichi_regime_stability"] = 0.5
            result["ichi_ma_well_multiplier"] = 1.0
            result["ichi_drift_multiplier"] = 1.0
            return result

        high = result["high"]
        low = result["low"]
        close = result["close"]

        # Tenkan-sen (conversion line)
        tenkan = (
            high.rolling(self.config.tenkan_period).max() +
            low.rolling(self.config.tenkan_period).min()
        ) / 2

        # Kijun-sen (base line)
        kijun = (
            high.rolling(self.config.kijun_period).max() +
            low.rolling(self.config.kijun_period).min()
        ) / 2

        # Senkou Span A (leading span A)
        senkou_a = ((tenkan + kijun) / 2).shift(self.config.cloud_displacement)

        # Senkou Span B (leading span B)
        senkou_b_raw = (
            high.rolling(self.config.senkou_b_period).max() +
            low.rolling(self.config.senkou_b_period).min()
        ) / 2
        senkou_b = senkou_b_raw.shift(self.config.cloud_displacement)

        # Cloud boundaries
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

        # Cloud thickness (normalized)
        cloud_thickness = (cloud_top - cloud_bottom) / (close + 1e-9)
        result["ichi_cloud_thickness"] = cloud_thickness

        # Distance from cloud (positive = above, negative = below)
        cloud_distance = np.zeros(n)
        regime_codes = np.zeros(n)
        regime_labels = []

        close_vals = close.values
        cloud_top_vals = cloud_top.values
        cloud_bottom_vals = cloud_bottom.values

        transition_zone = self.config.transition_zone_pct

        for i in range(n):
            ct = cloud_top_vals[i]
            cb = cloud_bottom_vals[i]
            c = close_vals[i]

            if np.isnan(ct) or np.isnan(cb):
                cloud_distance[i] = 0.0
                regime_codes[i] = 0
                regime_labels.append(ManifoldRegime.DIFFUSIVE.value)
                continue

            if c > ct:
                dist = (c - ct) / (c + 1e-9)
                cloud_distance[i] = dist
                if dist < transition_zone:
                    regime_codes[i] = 2  # transition
                    regime_labels.append(ManifoldRegime.TRANSITION.value)
                else:
                    regime_codes[i] = 1  # ballistic (above)
                    regime_labels.append(ManifoldRegime.BALLISTIC.value)
            elif c < cb:
                dist = (c - cb) / (c + 1e-9)
                cloud_distance[i] = dist
                if abs(dist) < transition_zone:
                    regime_codes[i] = 2
                    regime_labels.append(ManifoldRegime.TRANSITION.value)
                else:
                    regime_codes[i] = -1  # ballistic (below)
                    regime_labels.append(ManifoldRegime.BALLISTIC.value)
            else:
                cloud_distance[i] = 0.0
                regime_codes[i] = 0  # diffusive
                regime_labels.append(ManifoldRegime.DIFFUSIVE.value)

        result["ichi_cloud_distance"] = cloud_distance
        result["ichi_regime"] = regime_labels
        result["ichi_regime_code"] = regime_codes

        # Regime stability (how long in current regime)
        regime_change = pd.Series(regime_codes).diff().abs()
        stability = 1.0 - regime_change.rolling(20, min_periods=1).mean()
        result["ichi_regime_stability"] = stability.values

        # Force multipliers based on regime
        # Diffusive: MA wells stronger, drift weaker
        # Ballistic: MA wells weaker, drift stronger
        ma_mult = np.ones(n)
        drift_mult = np.ones(n)
        for i in range(n):
            rc = regime_codes[i]
            if rc == 0:  # diffusive
                ma_mult[i] = 1.5
                drift_mult[i] = 0.5
            elif abs(rc) == 1:  # ballistic
                ma_mult[i] = 0.5
                drift_mult[i] = 1.5
            else:  # transition
                ma_mult[i] = 1.0
                drift_mult[i] = 1.0

        result["ichi_ma_well_multiplier"] = ma_mult
        result["ichi_drift_multiplier"] = drift_mult

        return result


# ---------------------------------------------------------------------------
# 7. CVD -- Cumulative Volume Delta / Aggressive Flow
# ---------------------------------------------------------------------------

@dataclass
class CVDConfig:
    """Configuration for CVD tracker."""
    ewm_span_fast: int = 10
    ewm_span_slow: int = 50
    divergence_lookback: int = 20
    impulse_momentum_span: int = 5


class CVDTracker:
    """Cumulative Volume Delta for impulse directionality.

    CVD = cumsum(buy_volume - sell_volume)

    This is closest to "prints are truth." CVD informs the sign and
    persistence of impulses J_tau.

    Features:
    - CVD trend (fast vs slow EMA)
    - CVD-price divergence (warning signal)
    - CVD impulse (recent aggressive flow direction)
    """

    def __init__(self, config: Optional[CVDConfig] = None):
        self.config = config or CVDConfig()

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute CVD features.

        Args:
            df: DataFrame with buy_volume, sell_volume (single symbol, sorted).

        Returns:
            DataFrame with CVD features.
        """
        result = df.copy()
        n = len(result)

        if n < 5:
            result["cvd"] = 0.0
            result["cvd_delta"] = 0.0
            result["cvd_fast"] = 0.0
            result["cvd_slow"] = 0.0
            result["cvd_trend"] = 0.0
            result["cvd_divergence"] = 0.0
            result["cvd_impulse_sign"] = 0.0
            return result

        buy_vol = result["buy_volume"].values if "buy_volume" in result.columns else np.zeros(n)
        sell_vol = result["sell_volume"].values if "sell_volume" in result.columns else np.zeros(n)

        # Volume delta per bar
        delta = buy_vol - sell_vol
        result["cvd_delta"] = delta

        # Cumulative volume delta
        cvd = np.cumsum(delta)
        result["cvd"] = cvd

        # Fast and slow EMA of CVD
        cvd_series = pd.Series(cvd)
        fast = cvd_series.ewm(span=self.config.ewm_span_fast, adjust=False).mean()
        slow = cvd_series.ewm(span=self.config.ewm_span_slow, adjust=False).mean()
        result["cvd_fast"] = fast.values
        result["cvd_slow"] = slow.values

        # CVD trend (fast - slow normalized)
        total_vol = buy_vol + sell_vol
        norm = pd.Series(total_vol).rolling(self.config.ewm_span_slow, min_periods=5).mean().values
        norm = np.where(norm < 1e-9, 1.0, norm)
        result["cvd_trend"] = (fast.values - slow.values) / norm

        # CVD-price divergence
        # If price goes up but CVD goes down -> bearish divergence
        close = result["close"].values if "close" in result.columns else np.zeros(n)
        lookback = self.config.divergence_lookback
        divergence = np.zeros(n)
        for i in range(lookback, n):
            price_change = close[i] - close[i - lookback]
            cvd_change = cvd[i] - cvd[i - lookback]
            # Normalize
            price_norm = np.abs(close[i]) + 1e-9
            cvd_norm = np.abs(norm[i]) * lookback + 1e-9
            divergence[i] = (
                np.sign(price_change) * abs(price_change) / price_norm -
                np.sign(cvd_change) * abs(cvd_change) / cvd_norm
            )
        result["cvd_divergence"] = divergence

        # CVD impulse (recent aggressive flow sign and magnitude)
        delta_series = pd.Series(delta)
        impulse = delta_series.ewm(span=self.config.impulse_momentum_span, adjust=False).mean()
        result["cvd_impulse_sign"] = np.sign(impulse.values)

        return result


# ---------------------------------------------------------------------------
# Unified Crypto Particle Potential
# ---------------------------------------------------------------------------

@dataclass
class CryptoParticleConfig:
    """Master configuration for unified crypto particle potential."""
    avwap: AVWAPConfig = field(default_factory=AVWAPConfig)
    bollinger: BollingerConfig = field(default_factory=BollingerConfig)
    ma_well: MAWellConfig = field(default_factory=MAWellConfig)
    oi_hazard: OIHazardConfig = field(default_factory=OIHazardConfig)
    rsi_throttle: RSIThrottleConfig = field(default_factory=RSIThrottleConfig)
    ichimoku: IchimokuConfig = field(default_factory=IchimokuConfig)
    cvd: CVDConfig = field(default_factory=CVDConfig)

    # Potential weights
    w_avwap: float = 1.0
    w_ma_well: float = 1.0
    w_bollinger: float = 1.0
    w_liq: float = 1.0  # Weight for external liquidation potential

    # Drift weights
    w_ma_slope: float = 1.0
    w_funding: float = 1.0
    w_regime: float = 1.0


class CryptoParticlePotential:
    """Unified crypto potential integrating all physics-consistent indicators.

    Total potential:
        V = V_liqbook + V_AVWAP + V_MA + V_liq

    Diffusion:
        sigma_tau = f(BB_width, realized_vol, spread, depth)

    Drift:
        mu_tau = mu(MA_slope, regime, funding)

    Jump hazard:
        lambda_tau = lambda(delta_OI, liq_density, squeeze_state)

    RSI throttles impulse response:
        J_tau <- J_tau * g(RSI_tau)

    Ichimoku gates which regime physics apply:
        diffusive -> MA wells stronger, vol reverts
        ballistic -> MA wells weaker, drift dominates
    """

    def __init__(self, config: Optional[CryptoParticleConfig] = None):
        self.config = config or CryptoParticleConfig()
        self.avwap = AVWAPAnchorSet(self.config.avwap)
        self.bollinger = BollingerDiffusion(self.config.bollinger)
        self.ma_well = MAWellField(self.config.ma_well)
        self.oi_hazard = OIHazardModel(self.config.oi_hazard)
        self.rsi_throttle = RSIThrottle(self.config.rsi_throttle)
        self.ichimoku = IchimokuRegimeGate(self.config.ichimoku)
        self.cvd = CVDTracker(self.config.cvd)

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all crypto particle features for a single symbol.

        Args:
            df: DataFrame with OHLCV + buy_volume/sell_volume + returns.
                Must be single-symbol, sorted by bar_idx.

        Returns:
            DataFrame with all crypto particle features appended.
        """
        result = df.copy()

        # Ensure returns exist
        if "returns" not in result.columns:
            result["returns"] = result.groupby("symbol")["close"].pct_change() if "symbol" in result.columns else result["close"].pct_change()

        # Run per-symbol if multiple symbols present
        if "symbol" in result.columns and result["symbol"].nunique() > 1:
            parts = []
            for sym, grp in result.groupby("symbol"):
                processed = self._compute_single_symbol(grp)
                parts.append(processed)
            return pd.concat(parts, ignore_index=True)
        else:
            return self._compute_single_symbol(result)

    def _compute_single_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for a single symbol."""
        result = df.copy()

        # 1. AVWAP anchor wells (Tier 1)
        result = self.avwap.detect_and_compute(result)

        # 2. Bollinger diffusion geometry (Tier 1)
        result = self.bollinger.compute_features(result)

        # 3. MA well field (Tier 1)
        result = self.ma_well.compute_features(result)

        # 4. OI hazard multiplier
        result = self.oi_hazard.compute_features(result)

        # 5. RSI throttle (Tier 2)
        result = self.rsi_throttle.compute_features(result)

        # 6. Ichimoku regime gate (Tier 2)
        result = self.ichimoku.compute_features(result)

        # 7. CVD impulse tracker
        result = self.cvd.compute_features(result)

        # === Unified potential computation ===
        result = self._compute_unified_potential(result)

        # === Unified drift computation ===
        result = self._compute_unified_drift(result)

        # === Unified diffusion computation ===
        result = self._compute_unified_diffusion(result)

        # === Unified jump hazard ===
        result = self._compute_unified_hazard(result)

        return result

    def _compute_unified_potential(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute V = V_AVWAP + V_MA (+ external V_liq, V_liqbook).

        External potentials (liquidation book, order book) are added
        by the caller via the EnhancedQuantumEngine.
        """
        result = df.copy()

        v_avwap = result.get("avwap_potential", pd.Series(0.0, index=result.index))
        v_ma = result.get("ma_well_total_potential", pd.Series(0.0, index=result.index))

        # Ichimoku modulates MA well strength
        ichi_ma_mult = result.get("ichi_ma_well_multiplier", pd.Series(1.0, index=result.index))
        v_ma_modulated = v_ma * ichi_ma_mult

        result["unified_potential"] = (
            self.config.w_avwap * v_avwap +
            self.config.w_ma_well * v_ma_modulated
        )

        # Potential gradient (restoring force direction)
        avwap_dist = result.get("avwap_distance", pd.Series(0.0, index=result.index))
        ma_net_force = pd.Series(0.0, index=result.index)
        for span in self.config.ma_well.ema_spans:
            col = f"ma_well_{span}_force"
            if col in result.columns:
                ma_net_force = ma_net_force + result[col]

        result["unified_restoring_force"] = (
            -self.config.w_avwap * np.sign(avwap_dist) * v_avwap.clip(lower=0) +
            self.config.w_ma_well * ma_net_force * ichi_ma_mult
        )

        return result

    def _compute_unified_drift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute mu_tau = mu(MA_slope, regime, funding).

        Funding force is added externally by the caller.
        """
        result = df.copy()

        # MA slope drift (modulated by Ichimoku regime)
        ma_drift = result.get("ma_well_net_drift", pd.Series(0.0, index=result.index))
        ichi_drift_mult = result.get("ichi_drift_multiplier", pd.Series(1.0, index=result.index))

        # CVD-informed drift bias
        cvd_trend = result.get("cvd_trend", pd.Series(0.0, index=result.index))

        result["unified_drift"] = (
            self.config.w_ma_slope * ma_drift * ichi_drift_mult +
            cvd_trend * 0.3  # CVD provides directional bias
        )

        return result

    def _compute_unified_diffusion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sigma_tau = f(BB_width, realized_vol).

        Spread and depth can be added externally.
        """
        result = df.copy()

        bb_sigma = result.get("bb_sigma_tau", pd.Series(0.02, index=result.index))
        realized_vol = result.get("realized_vol", pd.Series(0.02, index=result.index))

        # Blend BB-derived and realized vol
        result["unified_sigma"] = 0.6 * bb_sigma + 0.4 * realized_vol.fillna(0.02)

        # Squeeze state amplifies potential energy (stored vol)
        squeeze_energy = result.get("bb_squeeze_energy", pd.Series(0.0, index=result.index))
        result["unified_squeeze_state"] = squeeze_energy

        return result

    def _compute_unified_hazard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute lambda_tau = lambda(delta_OI, liq_density, squeeze_state).

        RSI throttles impulse magnitude when hazard triggers.
        """
        result = df.copy()

        oi_hazard = result.get("oi_hazard_rate", pd.Series(0.02, index=result.index))
        squeeze_energy = result.get("bb_squeeze_energy", pd.Series(0.0, index=result.index))

        # Squeeze state increases jump hazard (compressed vol = stored energy)
        squeeze_boost = 1.0 + 0.1 * squeeze_energy.clip(upper=10)

        result["unified_jump_hazard"] = oi_hazard * squeeze_boost

        # RSI throttle on impulse magnitude
        rsi_throttle = result.get("rsi_throttle", pd.Series(1.0, index=result.index))
        cvd_impulse = result.get("cvd_impulse_sign", pd.Series(0.0, index=result.index))

        # Effective impulse = CVD direction * RSI throttle
        result["unified_impulse_efficiency"] = rsi_throttle
        result["unified_impulse_direction"] = cvd_impulse

        return result

    def get_state_vector(self, df: pd.DataFrame, bar_idx: int = -1) -> Dict[str, float]:
        """Extract the full particle state at a specific bar.

        Returns dict suitable for feeding into QuantumPriceEngine or
        Monte Carlo simulation.
        """
        if bar_idx < 0:
            bar_idx = len(df) + bar_idx

        row = df.iloc[bar_idx]

        def safe_get(col: str, default: float = 0.0) -> float:
            val = row.get(col, default)
            if isinstance(val, (int, float, np.integer, np.floating)):
                return float(val) if not np.isnan(val) else default
            return default

        return {
            # Potential
            "unified_potential": safe_get("unified_potential"),
            "unified_restoring_force": safe_get("unified_restoring_force"),
            "avwap_distance": safe_get("avwap_distance"),
            "avwap_well_strength": safe_get("avwap_well_strength"),
            "ma_well_total_potential": safe_get("ma_well_total_potential"),
            "ma_well_cluster_energy": safe_get("ma_well_cluster_energy"),
            # Drift
            "unified_drift": safe_get("unified_drift"),
            "ma_well_net_drift": safe_get("ma_well_net_drift"),
            "cvd_trend": safe_get("cvd_trend"),
            "cvd_divergence": safe_get("cvd_divergence"),
            # Diffusion
            "unified_sigma": safe_get("unified_sigma", 0.02),
            "bb_sigma_tau": safe_get("bb_sigma_tau", 0.02),
            "bb_squeeze_energy": safe_get("bb_squeeze_energy"),
            "bb_position": safe_get("bb_position", 0.5),
            # Hazard
            "unified_jump_hazard": safe_get("unified_jump_hazard", 0.02),
            "oi_hazard_rate": safe_get("oi_hazard_rate", 0.02),
            "unified_squeeze_state": safe_get("unified_squeeze_state"),
            # Impulse
            "unified_impulse_efficiency": safe_get("unified_impulse_efficiency", 1.0),
            "unified_impulse_direction": safe_get("unified_impulse_direction"),
            "rsi": safe_get("rsi", 50.0),
            "rsi_throttle": safe_get("rsi_throttle", 1.0),
            # Regime
            "ichi_regime_code": safe_get("ichi_regime_code"),
            "ichi_cloud_thickness": safe_get("ichi_cloud_thickness"),
            "ichi_regime_stability": safe_get("ichi_regime_stability", 0.5),
        }


# ---------------------------------------------------------------------------
# Feature name registry
# ---------------------------------------------------------------------------

def get_crypto_potential_feature_names() -> List[str]:
    """Get list of all crypto particle potential feature names."""
    return [
        # AVWAP
        "avwap_dominant", "avwap_distance", "avwap_well_strength",
        "avwap_n_competing", "avwap_potential",
        # Bollinger diffusion
        "bb_upper", "bb_lower", "bb_mid",
        "bb_width", "bb_width_norm", "bb_width_pctile",
        "bb_sigma_tau", "bb_squeeze", "bb_squeeze_energy",
        "bb_position", "bb_expansion_rate",
        # MA wells
        "ma_well_9_dist", "ma_well_9_force", "ma_well_9_slope",
        "ma_well_21_dist", "ma_well_21_force", "ma_well_21_slope",
        "ma_well_50_dist", "ma_well_50_force", "ma_well_50_slope",
        "ma_well_200_dist", "ma_well_200_force", "ma_well_200_slope",
        "ma_well_total_potential", "ma_well_net_drift",
        "ma_well_cluster", "ma_well_cluster_energy",
        # OI hazard
        "oi_change", "oi_change_pct", "oi_percentile",
        "oi_directional_buildup", "oi_hazard_rate",
        # RSI throttle
        "rsi", "rsi_throttle", "rsi_saturation",
        "rsi_impulse_efficiency_buy", "rsi_impulse_efficiency_sell",
        # Ichimoku regime
        "ichi_regime", "ichi_regime_code",
        "ichi_cloud_thickness", "ichi_cloud_distance",
        "ichi_regime_stability",
        "ichi_ma_well_multiplier", "ichi_drift_multiplier",
        # CVD
        "cvd", "cvd_delta", "cvd_fast", "cvd_slow",
        "cvd_trend", "cvd_divergence", "cvd_impulse_sign",
        # Unified
        "unified_potential", "unified_restoring_force",
        "unified_drift", "unified_sigma",
        "unified_squeeze_state", "unified_jump_hazard",
        "unified_impulse_efficiency", "unified_impulse_direction",
    ]


def get_crypto_potential_hyperparameters() -> List[Dict]:
    """Get hyperparameter definitions for the crypto potential module."""
    return [
        # AVWAP
        {
            "name": "avwap_impulse_threshold_sigma",
            "path": "particle.crypto_potential.avwap.impulse_threshold_sigma",
            "current_value": 2.0,
            "candidates": [1.5, 2.0, 2.5, 3.0],
            "variable_type": "continuous",
        },
        {
            "name": "avwap_flush_volume_mult",
            "path": "particle.crypto_potential.avwap.flush_volume_mult",
            "current_value": 3.0,
            "candidates": [2.0, 3.0, 4.0, 5.0],
            "variable_type": "continuous",
        },
        # Bollinger
        {
            "name": "bb_period",
            "path": "particle.crypto_potential.bollinger.bb_period",
            "current_value": 20,
            "candidates": [10, 15, 20, 30],
            "variable_type": "discrete",
        },
        {
            "name": "bb_squeeze_threshold",
            "path": "particle.crypto_potential.bollinger.squeeze_threshold",
            "current_value": 0.4,
            "candidates": [0.2, 0.3, 0.4, 0.5],
            "variable_type": "continuous",
        },
        # MA wells
        {
            "name": "ma_spring_constant_base",
            "path": "particle.crypto_potential.ma_well.spring_constant_base",
            "current_value": 1.0,
            "candidates": [0.5, 0.75, 1.0, 1.5, 2.0],
            "variable_type": "continuous",
        },
        {
            "name": "ma_slope_drift_scale",
            "path": "particle.crypto_potential.ma_well.slope_drift_scale",
            "current_value": 1.0,
            "candidates": [0.5, 0.75, 1.0, 1.5, 2.0],
            "variable_type": "continuous",
        },
        # OI hazard
        {
            "name": "oi_lambda_base",
            "path": "particle.crypto_potential.oi_hazard.lambda_base",
            "current_value": 0.02,
            "candidates": [0.01, 0.02, 0.04, 0.06],
            "variable_type": "continuous",
        },
        {
            "name": "oi_change_scale",
            "path": "particle.crypto_potential.oi_hazard.oi_change_scale",
            "current_value": 1.0,
            "candidates": [0.5, 1.0, 1.5, 2.0],
            "variable_type": "continuous",
        },
        # RSI throttle
        {
            "name": "rsi_period",
            "path": "particle.crypto_potential.rsi_throttle.rsi_period",
            "current_value": 14,
            "candidates": [7, 10, 14, 21],
            "variable_type": "discrete",
        },
        {
            "name": "rsi_throttle_power",
            "path": "particle.crypto_potential.rsi_throttle.throttle_power",
            "current_value": 2.0,
            "candidates": [1.0, 1.5, 2.0, 3.0],
            "variable_type": "continuous",
        },
        # Ichimoku
        {
            "name": "ichi_transition_zone_pct",
            "path": "particle.crypto_potential.ichimoku.transition_zone_pct",
            "current_value": 0.005,
            "candidates": [0.002, 0.005, 0.01, 0.02],
            "variable_type": "continuous",
        },
        # CVD
        {
            "name": "cvd_ewm_span_fast",
            "path": "particle.crypto_potential.cvd.ewm_span_fast",
            "current_value": 10,
            "candidates": [5, 10, 15, 20],
            "variable_type": "discrete",
        },
        # Unified weights
        {
            "name": "w_avwap",
            "path": "particle.crypto_potential.w_avwap",
            "current_value": 1.0,
            "candidates": [0.5, 0.75, 1.0, 1.25, 1.5],
            "variable_type": "continuous",
        },
        {
            "name": "w_ma_well",
            "path": "particle.crypto_potential.w_ma_well",
            "current_value": 1.0,
            "candidates": [0.5, 0.75, 1.0, 1.25, 1.5],
            "variable_type": "continuous",
        },
        {
            "name": "w_ma_slope",
            "path": "particle.crypto_potential.w_ma_slope",
            "current_value": 1.0,
            "candidates": [0.5, 0.75, 1.0, 1.25, 1.5],
            "variable_type": "continuous",
        },
    ]
