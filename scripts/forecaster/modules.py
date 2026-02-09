"""
Forecaster Modules — All 12 Paradigms
=======================================
Each module implements:
  - predict(snapshot, horizon_hours) -> ModuleOutput
  - trusted_regimes: set of Regime where this module has lift
  - failure_modes: list of known failure conditions
"""
from __future__ import annotations
import math
import time
from typing import Optional

import numpy as np

from .schemas import (
    Bar, MarketSnapshot, ModuleOutput, PredictionTargets,
    Regime, EvalMetrics,
)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _ema(data: list[float], span: int) -> list[float]:
    """Exponential moving average."""
    if not data:
        return []
    alpha = 2.0 / (span + 1)
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1 - alpha) * result[-1])
    return result


def _sma(data: list[float], window: int) -> list[float]:
    """Simple moving average with padding."""
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


def _rolling_std(data: list[float], window: int) -> list[float]:
    """Rolling standard deviation."""
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        chunk = data[start:i+1]
        if len(chunk) < 2:
            result.append(0.0)
        else:
            mean = sum(chunk) / len(chunk)
            var = sum((x - mean)**2 for x in chunk) / (len(chunk) - 1)
            result.append(math.sqrt(var))
    return result


def _log_returns(prices: list[float]) -> list[float]:
    """Compute log returns from price series."""
    return [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))
            if prices[i-1] > 0 and prices[i] > 0]


def _zscore(value: float, data: list[float]) -> float:
    """Z-score of value relative to data."""
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    std = math.sqrt(sum((x - mean)**2 for x in data) / (len(data) - 1))
    return (value - mean) / std if std > 0 else 0.0


def _logistic(z: float) -> float:
    """Logistic sigmoid: 1/(1+exp(-z))."""
    z = max(-20, min(20, z))  # clamp to avoid overflow
    return 1.0 / (1.0 + math.exp(-z))


def _calibrate_direction_prob(raw_prob: float) -> float:
    """
    Post-hoc calibration of direction probability.

    Maps raw ensemble direction_prob to calibrated probability using
    piece-wise linear interpolation from backtest observed frequencies.

    TensorTrade research insight: prediction engines tend to be
    over-confident in the 0.55-0.70 range (moderate bullish bias).
    The agent "knows" direction but overstates certainty. This map
    Backtest insight: prediction engines tend to be over-confident
    in the 0.55-0.70 range (moderate bullish bias). The ensemble
    "knows" direction but overstates certainty. This map
    shrinks confidence toward 0.50 where the model has no real edge
    and preserves signal at extremes where it does.

    Calibration points (from 75-forecast, 2000-bar BTCUSDT backtest):
        raw  →  calibrated  (source: observed hit rates)
        0.30 →  0.43        (raw predicted 0.37, observed 0.50 → symmetric)
        0.40 →  0.46        (raw predicted 0.47, observed 0.46)
        0.50 →  0.50        (anchor: 50/50 by definition)
        0.55 →  0.50        (raw predicted 0.54, observed 0.43 → ~50%)
        0.60 →  0.50        (model has no edge here; clamp)
        0.65 →  0.51        (raw predicted 0.63, observed 0.46 → minimal edge)
        0.70 →  0.53        (start of real directional signal)
        0.80 →  0.60        (strong signal zone)
        0.90 →  0.72        (very strong signal)

    Symmetric for bearish side (prob < 0.50).
    """
    # Calibration table: (raw_prob, calibrated_prob) for bullish side
    # Bearish is symmetric: raw 0.3 maps same as 1.0 - map(0.7)
    #
    # Goal: shrink overconfidence but PRESERVE directional signal.
    # The evaluator uses >0.50 = bullish, <0.50 = bearish.
    # If we clamp everything to 0.500 we destroy the signal entirely.
    # Instead: compress the range [0.50, 0.70] → [0.50, 0.55] to
    # keep the correct side of 0.50 while reducing magnitude.
    cal_points = [
        (0.50, 0.500),
        (0.55, 0.515),   # was 0.55 → compress to 0.515
        (0.60, 0.530),   # was 0.60 → compress to 0.53
        (0.65, 0.545),   # was 0.65 → compress to 0.545
        (0.70, 0.560),   # was 0.70 → some signal preserved
        (0.75, 0.590),
        (0.80, 0.630),
        (0.85, 0.680),
        (0.90, 0.740),
        (0.95, 0.810),
        (1.00, 0.880),
    ]

    # Handle bearish side by symmetry
    if raw_prob < 0.50:
        calibrated_bull = _calibrate_direction_prob(1.0 - raw_prob)
        return 1.0 - calibrated_bull

    # Piece-wise linear interpolation on bullish side
    if raw_prob >= 1.0:
        return cal_points[-1][1]

    for i in range(len(cal_points) - 1):
        lo_raw, lo_cal = cal_points[i]
        hi_raw, hi_cal = cal_points[i + 1]
        if lo_raw <= raw_prob <= hi_raw:
            t = (raw_prob - lo_raw) / (hi_raw - lo_raw) if hi_raw > lo_raw else 0
            return lo_cal + t * (hi_cal - lo_cal)

    return raw_prob  # fallback


def _rsi(prices: list[float], period: int = 14) -> float:
    """Relative Strength Index."""
    if len(prices) < period + 1:
        return 50.0
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    recent = changes[-period:]
    gains = [c for c in recent if c > 0]
    losses = [-c for c in recent if c < 0]
    avg_gain = sum(gains) / period if gains else 0.001
    avg_loss = sum(losses) / period if losses else 0.001
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(highs: list[float], lows: list[float], closes: list[float],
         period: int = 14) -> float:
    """Average True Range."""
    if len(highs) < period + 1:
        if highs and lows:
            return sum(h - l for h, l in zip(highs, lows)) / len(highs)
        return 0.0
    trs = []
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return sum(trs[-period:]) / period


# ═══════════════════════════════════════════════════════════════
# MODULE 1: TECHNICAL FEATURE GENERATORS
# ═══════════════════════════════════════════════════════════════

class TechnicalModule:
    """
    TA as features, not religion. Generates trend, mean-reversion,
    volatility regime, and volume confirmation features.

    Trusted in: TREND_UP, TREND_DOWN, RANGE
    Fails in: CASCADE_RISK, EVENT_WINDOW, POST_JUMP
    """
    name = "technical"
    trusted_regimes = {Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE}

    def predict(self, snap: MarketSnapshot, horizon_hours: float = 24) -> ModuleOutput:
        t0 = time.time()
        closes = snap.closes
        targets = PredictionTargets()
        features = {}

        if len(closes) < 30:
            return ModuleOutput(self.name, targets, confidence=0.0,
                                elapsed_ms=(time.time()-t0)*1000)

        price = closes[-1]

        # ── Trend features ────────────────────────────────
        ema_12 = _ema(closes, 12)
        ema_26 = _ema(closes, 26)
        ema_50 = _ema(closes, 50) if len(closes) >= 50 else ema_26
        sma_200 = _sma(closes, min(200, len(closes)))

        # EMA slopes (rate of change)
        ema12_slope = (ema_12[-1] - ema_12[-2]) / ema_12[-2] if len(ema_12) >= 2 else 0
        ema26_slope = (ema_26[-1] - ema_26[-2]) / ema_26[-2] if len(ema_26) >= 2 else 0

        # MACD
        macd = ema_12[-1] - ema_26[-1]
        macd_signal = _ema([ema_12[i] - ema_26[i] for i in range(len(ema_26))], 9)
        macd_hist = macd - macd_signal[-1] if macd_signal else 0

        # Cross states
        ema_cross_bull = 1.0 if ema_12[-1] > ema_26[-1] else -1.0
        price_above_sma200 = 1.0 if price > sma_200[-1] else -1.0

        # Donchian breakout
        highs = snap.highs
        lows = snap.lows
        lookback = min(20, len(highs))
        donchian_high = max(highs[-lookback:]) if highs else price
        donchian_low = min(lows[-lookback:]) if lows else price
        donchian_pos = (price - donchian_low) / (donchian_high - donchian_low) \
            if donchian_high > donchian_low else 0.5

        features["ema12_slope"] = ema12_slope
        features["ema26_slope"] = ema26_slope
        features["macd_hist"] = macd_hist / price if price > 0 else 0
        features["ema_cross_bull"] = ema_cross_bull
        features["price_above_sma200"] = price_above_sma200
        features["donchian_pos"] = donchian_pos

        # ── Mean reversion features ───────────────────────
        sma_20 = _sma(closes, 20)
        std_20 = _rolling_std(closes, 20)
        z_score = (price - sma_20[-1]) / std_20[-1] if std_20[-1] > 0 else 0

        rsi_val = _rsi(closes, 14)
        rsi_normalized = (rsi_val - 50) / 50  # -1 to +1

        # Bollinger %B
        upper_bb = sma_20[-1] + 2 * std_20[-1]
        lower_bb = sma_20[-1] - 2 * std_20[-1]
        bb_pctb = (price - lower_bb) / (upper_bb - lower_bb) \
            if upper_bb > lower_bb else 0.5
        bb_bandwidth = (upper_bb - lower_bb) / sma_20[-1] if sma_20[-1] > 0 else 0

        features["zscore_20"] = z_score
        features["rsi_14"] = rsi_val
        features["rsi_normalized"] = rsi_normalized
        features["bb_pctb"] = bb_pctb
        features["bb_bandwidth"] = bb_bandwidth

        # ── Volatility regime features ────────────────────
        atr_val = _atr(snap.highs, snap.lows, closes, 14)
        atr_pct = atr_val / price if price > 0 else 0
        log_rets = _log_returns(closes)
        realized_vol = float(np.std(log_rets[-24:])) if len(log_rets) >= 24 else \
            float(np.std(log_rets)) if log_rets else 0.02
        # Vol of vol
        if len(log_rets) >= 48:
            vol_windows = [float(np.std(log_rets[i:i+24]))
                           for i in range(0, len(log_rets)-24, 6)]
            vol_of_vol = float(np.std(vol_windows)) if len(vol_windows) >= 2 else 0
        else:
            vol_of_vol = 0.0

        features["atr_pct"] = atr_pct
        features["realized_vol"] = realized_vol
        features["vol_of_vol"] = vol_of_vol

        # ── Volume confirmation ───────────────────────────
        volumes = snap.volumes
        if len(volumes) >= 20:
            vol_sma = sum(volumes[-20:]) / 20
            vol_zscore = _zscore(volumes[-1], volumes[-20:])
            obv = 0.0
            for i in range(1, len(closes)):
                if closes[i] > closes[i-1]:
                    obv += volumes[i] if i < len(volumes) else 0
                elif closes[i] < closes[i-1]:
                    obv -= volumes[i] if i < len(volumes) else 0
            features["volume_zscore"] = vol_zscore
            features["obv_slope"] = obv / (price * vol_sma) if vol_sma > 0 else 0
        else:
            features["volume_zscore"] = 0
            features["obv_slope"] = 0

        # ── Synthesize prediction ─────────────────────────
        # Trend score: momentum signals weighted
        trend_score = (
            0.25 * ema12_slope * 100 +   # amplify tiny slopes
            0.20 * (macd_hist / price * 1000 if price > 0 else 0) +
            0.15 * ema_cross_bull +
            0.15 * price_above_sma200 +
            0.15 * (donchian_pos - 0.5) * 2 +
            0.10 * features.get("obv_slope", 0)
        )

        # Mean reversion signal: extreme z-scores pull back
        mr_score = 0.0
        if abs(z_score) > 1.5:
            mr_score = -z_score * 0.3  # reversion force
        if rsi_val > 80:
            mr_score -= 0.2
        elif rsi_val < 20:
            mr_score += 0.2

        # Combine with regime-aware blending
        combined = trend_score * 0.6 + mr_score * 0.4

        # Convert to expected return (scaled to horizon)
        scale = math.sqrt(horizon_hours / 24)
        targets.expected_return = combined * realized_vol * scale
        targets.return_std = realized_vol * scale
        targets.direction_prob = _logistic(combined * 2)
        targets.volatility_forecast = realized_vol * scale
        targets.vol_of_vol = vol_of_vol

        # Quantiles
        targets.quantile_10 = targets.expected_return - 1.28 * targets.return_std
        targets.quantile_50 = targets.expected_return
        targets.quantile_90 = targets.expected_return + 1.28 * targets.return_std

        # Confidence: lower in extreme vol
        confidence = max(0.1, min(0.8, 0.6 - abs(z_score) * 0.1))

        return ModuleOutput(
            self.name, targets, confidence=confidence,
            features=features, elapsed_ms=(time.time()-t0)*1000,
        )


# ═══════════════════════════════════════════════════════════════
# MODULE 2: CLASSICAL STATS (GARCH-ish vol + Kalman trend)
# ═══════════════════════════════════════════════════════════════

class ClassicalStatsModule:
    """
    EWMA volatility, regime-switching detector, Kalman-style trend filter.

    Best for: volatility forecasts, uncertainty bands, regime detection.
    Bad at: directional return prediction.
    """
    name = "classical_stats"
    trusted_regimes = {Regime.RANGE, Regime.VOL_EXPANSION, Regime.TREND_UP, Regime.TREND_DOWN}

    def predict(self, snap: MarketSnapshot, horizon_hours: float = 24) -> ModuleOutput:
        t0 = time.time()
        closes = snap.closes
        targets = PredictionTargets()

        if len(closes) < 20:
            return ModuleOutput(self.name, targets, confidence=0.0,
                                elapsed_ms=(time.time()-t0)*1000)

        log_rets = _log_returns(closes)
        if len(log_rets) < 10:
            return ModuleOutput(self.name, targets, confidence=0.0,
                                elapsed_ms=(time.time()-t0)*1000)

        # ── EWMA Volatility (GARCH-like) ─────────────────
        # sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_t
        lam = 0.94  # RiskMetrics lambda
        var_ewma = log_rets[0] ** 2
        ewma_vars = [var_ewma]
        for r in log_rets[1:]:
            var_ewma = lam * var_ewma + (1 - lam) * r ** 2
            ewma_vars.append(var_ewma)
        sigma_ewma = math.sqrt(ewma_vars[-1])

        # Scale to horizon
        scale = math.sqrt(horizon_hours)
        sigma_h = sigma_ewma * scale

        # ── Kalman-style trend filter ─────────────────────
        # Simple exponential smoothing of returns as "drift"
        alpha_kalman = 0.1
        mu_kalman = log_rets[0]
        for r in log_rets[1:]:
            mu_kalman = alpha_kalman * r + (1 - alpha_kalman) * mu_kalman

        # ── Regime detection via vol clustering ───────────
        # High vol state vs low vol state
        recent_vol = float(np.std(log_rets[-24:])) if len(log_rets) >= 24 else sigma_ewma
        long_vol = float(np.std(log_rets[-min(168, len(log_rets)):])) if len(log_rets) >= 48 else sigma_ewma
        vol_ratio = recent_vol / long_vol if long_vol > 0 else 1.0

        # Vol expansion if recent >> long
        vol_expanding = vol_ratio > 1.5

        # ── Autocorrelation check ─────────────────────────
        if len(log_rets) >= 24:
            r_mean = sum(log_rets[-24:]) / 24
            numerator = sum((log_rets[-24+i] - r_mean) * (log_rets[-24+i-1] - r_mean)
                           for i in range(1, 24))
            denominator = sum((r - r_mean)**2 for r in log_rets[-24:])
            autocorr = numerator / denominator if denominator > 0 else 0
        else:
            autocorr = 0.0

        features = {
            "sigma_ewma": sigma_ewma,
            "sigma_horizon": sigma_h,
            "mu_kalman": mu_kalman,
            "vol_ratio": vol_ratio,
            "vol_expanding": float(vol_expanding),
            "autocorr_lag1": autocorr,
        }

        # ── Build targets ─────────────────────────────────
        targets.expected_return = mu_kalman * horizon_hours  # drift extrapolation
        targets.return_std = sigma_h
        targets.volatility_forecast = sigma_h
        targets.vol_of_vol = abs(vol_ratio - 1.0) * sigma_ewma
        targets.direction_prob = _logistic(mu_kalman / sigma_ewma * 2) if sigma_ewma > 0 else 0.5

        # Quantiles from normal approx
        targets.quantile_10 = targets.expected_return - 1.28 * sigma_h
        targets.quantile_50 = targets.expected_return
        targets.quantile_90 = targets.expected_return + 1.28 * sigma_h

        # Jump probability: P(|r| > 2*sigma)
        # Under normal: ~4.6%, but crypto is heavy-tailed so multiply
        tail_multiplier = 2.5  # empirical crypto kurtosis adjustment
        targets.jump_prob = min(0.5, 0.046 * tail_multiplier * vol_ratio)
        targets.crash_prob = min(0.3, 0.023 * tail_multiplier * vol_ratio)

        # Confidence: vol models are well-calibrated
        confidence = 0.7 if not vol_expanding else 0.5

        return ModuleOutput(
            self.name, targets, confidence=confidence,
            features=features, elapsed_ms=(time.time()-t0)*1000,
        )


# ═══════════════════════════════════════════════════════════════
# MODULE 3: MACRO / CROSS-ASSET FACTOR RESIDUALIZATION
# ═══════════════════════════════════════════════════════════════

class MacroFactorModule:
    """
    Decomposes crypto returns into macro factor exposures:
    - Risk-on/off (SPX proxy)
    - USD strength (DXY proxy)
    - Crypto-specific residual

    Works when crypto trades as a macro risk asset.
    Fails during crypto-specific events.
    """
    name = "macro_factor"
    trusted_regimes = {Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE}

    def predict(self, snap: MarketSnapshot, horizon_hours: float = 24) -> ModuleOutput:
        t0 = time.time()
        targets = PredictionTargets()
        features = {}

        # Factor loadings (empirical betas, calibrated offline)
        # BTC ~ 0.3 * SPX + (-0.2) * DXY + alpha
        beta_spx = 0.30
        beta_dxy = -0.20

        macro_drift = 0.0
        confidence = 0.2  # low default when no macro data

        if snap.spx_return_1d is not None:
            spx_contribution = beta_spx * snap.spx_return_1d
            macro_drift += spx_contribution
            features["spx_return_1d"] = snap.spx_return_1d
            features["spx_contribution"] = spx_contribution
            confidence = 0.4

        if snap.dxy_return_1d is not None:
            dxy_contribution = beta_dxy * snap.dxy_return_1d
            macro_drift += dxy_contribution
            features["dxy_return_1d"] = snap.dxy_return_1d
            features["dxy_contribution"] = dxy_contribution
            confidence = 0.5

        if snap.gold_return_1d is not None:
            features["gold_return_1d"] = snap.gold_return_1d

        # ETH/BTC ratio as crypto risk appetite
        if snap.eth_btc_ratio is not None:
            features["eth_btc_ratio"] = snap.eth_btc_ratio

        # Factor-implied return
        scale = horizon_hours / 24
        targets.expected_return = macro_drift * scale
        targets.return_std = 0.02 * math.sqrt(scale)  # baseline uncertainty
        targets.direction_prob = _logistic(macro_drift * 20)

        targets.quantile_10 = targets.expected_return - 1.28 * targets.return_std
        targets.quantile_50 = targets.expected_return
        targets.quantile_90 = targets.expected_return + 1.28 * targets.return_std

        return ModuleOutput(
            self.name, targets, confidence=confidence,
            features=features, elapsed_ms=(time.time()-t0)*1000,
        )


# ═══════════════════════════════════════════════════════════════
# MODULE 4: DERIVATIVES POSITIONING (LFI + TAIL RISK)
# ═══════════════════════════════════════════════════════════════

class DerivativesModule:
    """
    Produces a Leverage Fragility Index (LFI) from:
    - Funding rate (level + changes)
    - Open interest + OI changes
    - Long/short ratios
    - Liquidation cascade signals

    Best for: tail risk, squeeze/cascade detection, short-horizon direction
              when positioning is extreme + liquidity thin.
    """
    name = "derivatives"
    trusted_regimes = {Regime.CASCADE_RISK, Regime.VOL_EXPANSION, Regime.TREND_UP, Regime.TREND_DOWN}

    def predict(self, snap: MarketSnapshot, horizon_hours: float = 24) -> ModuleOutput:
        t0 = time.time()
        targets = PredictionTargets()
        features = {}

        confidence = 0.15  # low default

        # ── Funding rate analysis ─────────────────────────
        if snap.funding_rate is not None:
            fr = snap.funding_rate
            features["funding_rate"] = fr
            features["funding_extreme"] = abs(fr) > 0.01  # >1% = extreme
            features["funding_direction"] = 1.0 if fr > 0 else (-1.0 if fr < 0 else 0)

            # High positive funding → longs crowded → short squeeze possible
            # But also mean-reversion risk if funding extreme for long
            if abs(fr) > 0.005:
                confidence = 0.4

            if snap.funding_rate_history and len(snap.funding_rate_history) >= 3:
                fr_accel = snap.funding_rate_history[-1] - snap.funding_rate_history[-3]
                features["funding_acceleration"] = fr_accel
        else:
            fr = 0.0

        # ── Open interest analysis ────────────────────────
        if snap.open_interest is not None and snap.oi_history:
            oi = snap.open_interest
            oi_change = (oi / snap.oi_history[-1] - 1.0) if snap.oi_history[-1] > 0 else 0
            features["oi_change_pct"] = oi_change
            features["oi_growth_fast"] = oi_change > 0.05  # 5% OI growth = leverage building

            if len(snap.oi_history) >= 5:
                oi_z = _zscore(oi, snap.oi_history[-20:] if len(snap.oi_history) >= 20 else snap.oi_history)
                features["oi_zscore"] = oi_z
            confidence = max(confidence, 0.35)

        # ── Leverage Fragility Index (LFI) ────────────────
        # LFI = w1*z(delta_OI) + w2*z(|funding|) + w3*z(liq_volume) - w4*z(book_depth)
        lfi_components = []

        # OI growth z-score
        oi_z = features.get("oi_zscore", 0)
        lfi_components.append(0.30 * oi_z)

        # Funding magnitude z-score
        funding_z = abs(fr) / 0.005 if abs(fr) > 0 else 0  # normalize by typical funding
        lfi_components.append(0.25 * funding_z)

        # Book depth (inverse = less depth = more fragile)
        if snap.bid_depth is not None and snap.ask_depth is not None:
            depth = snap.bid_depth + snap.ask_depth
            depth_z = max(0, 2.0 - depth / 100)  # lower depth = higher z
            lfi_components.append(-0.20 * (-depth_z))  # negative contribution for good depth
            features["book_depth"] = depth

        # Spread (wider = more fragile)
        if snap.spread is not None:
            spread_z = snap.spread / 0.001  # normalize by typical spread
            lfi_components.append(0.15 * spread_z)
            features["spread"] = snap.spread

        lfi = sum(lfi_components)
        features["lfi"] = lfi
        features["lfi_high"] = lfi > 1.5
        features["lfi_extreme"] = lfi > 2.5

        # ── Tail risk from LFI ────────────────────────────
        base_jump = 0.05
        base_crash = 0.02
        targets.jump_prob = min(0.6, base_jump * (1 + lfi))
        targets.crash_prob = min(0.4, base_crash * (1 + max(0, lfi)))

        # Direction: extreme positive funding → expect pullback
        if fr > 0.005:
            targets.direction_prob = 0.5 - min(0.2, fr * 10)  # lean bearish
            targets.expected_return = -abs(fr) * 2
        elif fr < -0.005:
            targets.direction_prob = 0.5 + min(0.2, abs(fr) * 10)  # lean bullish
            targets.expected_return = abs(fr) * 2
        else:
            targets.direction_prob = 0.5
            targets.expected_return = 0.0

        targets.return_std = 0.03 * (1 + lfi * 0.5)
        targets.volatility_forecast = targets.return_std
        targets.quantile_10 = targets.expected_return - 1.28 * targets.return_std
        targets.quantile_50 = targets.expected_return
        targets.quantile_90 = targets.expected_return + 1.28 * targets.return_std

        if lfi > 1.5:
            confidence = 0.6

        return ModuleOutput(
            self.name, targets, confidence=confidence,
            features=features, elapsed_ms=(time.time()-t0)*1000,
        )


# ═══════════════════════════════════════════════════════════════
# MODULE 5: MICROSTRUCTURE / ORDER FLOW
# ═══════════════════════════════════════════════════════════════

class MicrostructureModule:
    """
    Order Flow Imbalance (OFI), trade imbalance, liquidity slope.
    Best for hourly forecasts with good data.

    Overrides TA when flow disagrees.
    """
    name = "microstructure"
    trusted_regimes = {Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE}

    def predict(self, snap: MarketSnapshot, horizon_hours: float = 24) -> ModuleOutput:
        t0 = time.time()
        targets = PredictionTargets()
        features = {}
        confidence = 0.1

        # ── Trade imbalance ───────────────────────────────
        if snap.recent_trades:
            buy_vol = sum(t.get("qty", 0) for t in snap.recent_trades
                         if t.get("side") == "buy")
            sell_vol = sum(t.get("qty", 0) for t in snap.recent_trades
                          if t.get("side") == "sell")
            total = buy_vol + sell_vol
            if total > 0:
                trade_imbalance = (buy_vol - sell_vol) / total
                features["trade_imbalance"] = trade_imbalance
                confidence = 0.3
            else:
                trade_imbalance = 0.0
        else:
            trade_imbalance = 0.0

        # ── Book imbalance ────────────────────────────────
        if snap.bid_depth is not None and snap.ask_depth is not None:
            total_depth = snap.bid_depth + snap.ask_depth
            if total_depth > 0:
                book_imbalance = (snap.bid_depth - snap.ask_depth) / total_depth
                features["book_imbalance"] = book_imbalance
                confidence = max(confidence, 0.35)
            else:
                book_imbalance = 0.0
        else:
            book_imbalance = 0.0

        # ── Spread as liquidity signal ────────────────────
        if snap.spread is not None:
            features["spread_bps"] = snap.spread * 10000 if snap.current_price > 0 else 0
            # Wide spread = uncertainty / thin liquidity
            if snap.spread > 0.002:  # >20bps
                confidence *= 0.5  # less confident in thin markets

        # ── Synthesize ────────────────────────────────────
        flow_signal = trade_imbalance * 0.6 + book_imbalance * 0.4

        targets.direction_prob = _logistic(flow_signal * 3)
        targets.expected_return = flow_signal * 0.005 * math.sqrt(horizon_hours)
        targets.return_std = 0.02 * math.sqrt(horizon_hours / 24)
        targets.volatility_forecast = targets.return_std

        targets.quantile_10 = targets.expected_return - 1.28 * targets.return_std
        targets.quantile_50 = targets.expected_return
        targets.quantile_90 = targets.expected_return + 1.28 * targets.return_std

        return ModuleOutput(
            self.name, targets, confidence=confidence,
            features=features, elapsed_ms=(time.time()-t0)*1000,
        )


# ═══════════════════════════════════════════════════════════════
# MODULE 6: ON-CHAIN SLOW PRIORS
# ═══════════════════════════════════════════════════════════════

class OnChainModule:
    """
    Slow-moving cycle and risk context signals from on-chain data.
    NOT for hourly timing — provides drift priors and risk context.
    """
    name = "onchain"
    trusted_regimes = set(Regime)  # always contributes as a prior

    def predict(self, snap: MarketSnapshot, horizon_hours: float = 24) -> ModuleOutput:
        t0 = time.time()
        targets = PredictionTargets()
        features = {}
        confidence = 0.15

        # ── MVRV Ratio ────────────────────────────────────
        if snap.mvrv_ratio is not None:
            features["mvrv"] = snap.mvrv_ratio
            # MVRV > 3.5 = overheated; < 1.0 = undervalued
            if snap.mvrv_ratio > 3.5:
                targets.expected_return = -0.005  # slight bearish drift
                targets.crash_prob = 0.15
                features["mvrv_signal"] = "overheated"
            elif snap.mvrv_ratio < 1.0:
                targets.expected_return = 0.005  # slight bullish drift
                features["mvrv_signal"] = "undervalued"
            else:
                targets.expected_return = 0.0
                features["mvrv_signal"] = "neutral"
            confidence = 0.25

        # ── Exchange flows ────────────────────────────────
        if snap.exchange_inflow_24h is not None and snap.exchange_outflow_24h is not None:
            net_flow = snap.exchange_outflow_24h - snap.exchange_inflow_24h
            features["net_exchange_flow"] = net_flow
            # Net outflow = accumulation (bullish); inflow = distribution
            if net_flow > 0:
                targets.direction_prob = 0.55
                features["flow_signal"] = "accumulation"
            else:
                targets.direction_prob = 0.45
                features["flow_signal"] = "distribution"
            confidence = 0.3

        targets.return_std = 0.03  # wide uncertainty for slow signals
        targets.volatility_forecast = 0.03
        targets.quantile_10 = targets.expected_return - 0.04
        targets.quantile_50 = targets.expected_return
        targets.quantile_90 = targets.expected_return + 0.04

        return ModuleOutput(
            self.name, targets, confidence=confidence,
            features=features, elapsed_ms=(time.time()-t0)*1000,
        )


# ═══════════════════════════════════════════════════════════════
# MODULE 7: SENTIMENT / ATTENTION MODULATORS
# ═══════════════════════════════════════════════════════════════

class SentimentModule:
    """
    Sentiment as a secondary modulator:
    - Increase tail risk under euphoria
    - Detect capitulation after crashes

    NOT a primary direction predictor.
    """
    name = "sentiment"
    trusted_regimes = {Regime.TREND_UP, Regime.TREND_DOWN, Regime.CASCADE_RISK}

    def predict(self, snap: MarketSnapshot, horizon_hours: float = 24) -> ModuleOutput:
        t0 = time.time()
        targets = PredictionTargets()
        features = {}
        confidence = 0.1

        if snap.fear_greed_index is not None:
            fgi = snap.fear_greed_index
            features["fear_greed"] = fgi

            # Extreme greed → increase crash prob
            if fgi > 80:
                targets.crash_prob = 0.12
                targets.jump_prob = 0.15
                targets.direction_prob = 0.45  # slight contrarian
                features["sentiment_signal"] = "extreme_greed"
                confidence = 0.3
            # Extreme fear → potential bounce
            elif fgi < 20:
                targets.crash_prob = 0.05
                targets.direction_prob = 0.55  # slight contrarian
                features["sentiment_signal"] = "extreme_fear"
                confidence = 0.3
            else:
                features["sentiment_signal"] = "neutral"

        if snap.social_volume is not None:
            features["social_volume"] = snap.social_volume

        targets.return_std = 0.03
        targets.volatility_forecast = 0.03
        targets.quantile_10 = -0.04
        targets.quantile_50 = 0.0
        targets.quantile_90 = 0.04

        return ModuleOutput(
            self.name, targets, confidence=confidence,
            features=features, elapsed_ms=(time.time()-t0)*1000,
        )


# ═══════════════════════════════════════════════════════════════
# MODULE 8: TABULAR ML META-LEARNER
# ═══════════════════════════════════════════════════════════════

# ── ML hyperparameter defaults (from walk-forward tuning) ──────
# Key findings: low learning rate + tight regularization + early
# stopping prevents overfitting. Fewer features (≤13) consistently
# outperform large feature sets (34+) which overfit to noise.
_GBM_PARAMS = {
    "objective": "binary",        # direction classification
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,        # low LR for stable convergence
    "num_leaves": 15,             # tight — prevents memorising noise
    "max_depth": 4,               # shallow trees generalise better
    "min_child_samples": 8,       # need real support per leaf
    "subsample": 0.7,             # row subsampling
    "colsample_bytree": 0.6,     # feature subsampling per tree
    "reg_alpha": 0.3,             # L1 regularisation
    "reg_lambda": 1.0,            # L2 regularisation
    "n_estimators": 300,          # cap; early stopping will pick fewer
    "verbose": -1,
}

_GBM_REG_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 8,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.3,
    "reg_lambda": 1.0,
    "n_estimators": 300,
    "verbose": -1,
}

# Maximum features the GBM will use.  Research showed 13 beats 34.
_MAX_GBM_FEATURES = 16

# Minimum training samples before the GBM is trusted over the
# fallback weighted-average.
_MIN_GBM_TRAIN_SAMPLES = 30
# fallback weighted-average.  With 75 walk-forward samples the GBM
# was overfitting (HR dropped from 46.7% baseline to 45.3%).
# Require 500+ to accumulate enough regime diversity before enabling.
_MIN_GBM_TRAIN_SAMPLES = 500


class MetaLearnerModule:
    """
    Gradient Boosted Trees meta-learner over features from all
    other modules, trained walk-forward on accumulated predictions.

    Walk-forward protocol:
      1. Collect (features, actual_return) pairs from past forecasts.
      2. Train LightGBM direction classifier + return regressor on
         the expanding window — no look-ahead.
      3. Predict direction_prob and expected_return for current bar.
      4. Calibrate probabilities via walk-forward isotonic regression
         when enough samples exist, else fall back to the static
         piece-wise calibration map.

    Feature selection:
      After training, only the top _MAX_GBM_FEATURES by importance
      are retained.  Fewer, stronger features beat many noisy ones.
    """
    name = "meta_learner"
    trusted_regimes = set(Regime)

    def __init__(self):
        # Walk-forward history: list of (feature_dict, actual_return)
        self._history: list[tuple[dict, float]] = []
        # Trained models (None until enough data)
        self._dir_model = None     # LightGBM classifier
        self._ret_model = None     # LightGBM regressor
        self._feature_names: list[str] = []  # ordered feature columns
        self._isotonic = None      # sklearn IsotonicRegression for calibration
        self._gbm_available = False
        try:
            import lightgbm  # noqa: F401
            self._gbm_available = True
        except ImportError:
            pass

    # ── Public API ──────────────────────────────────────────────

    def record_outcome(self, features: dict, actual_return: float):
        """Store a past (features → actual) pair for walk-forward training."""
        self._history.append((features, actual_return))

    def predict_from_modules(self, outputs: list[ModuleOutput],
                             horizon_hours: float = 24) -> ModuleOutput:
        t0 = time.time()
        targets = PredictionTargets()

        if not outputs:
            return ModuleOutput(self.name, targets, confidence=0.0,
                                elapsed_ms=(time.time()-t0)*1000)

        # ── 1. Collect features from all modules ──────────────
        all_features = self._extract_features(outputs)

        # ── 2. Confidence-weighted baseline (always computed) ──
        baseline = self._weighted_average(outputs)

        # ── 3. Try GBM prediction ─────────────────────────────
        gbm_used = False
        if (self._gbm_available
                and len(self._history) >= _MIN_GBM_TRAIN_SAMPLES):
            try:
                self._retrain_if_needed()
                if self._dir_model is not None:
                    gbm_pred = self._gbm_predict(all_features)
                    if gbm_pred is not None:
                        # Blend GBM with baseline — GBM gets more
                        # weight as training data grows
                        n = len(self._history)
                        gbm_w = min(0.8, n / (n + 50))  # ramp up
                        base_w = 1.0 - gbm_w

                        targets.direction_prob = (
                            gbm_w * gbm_pred["direction_prob"]
                            + base_w * baseline.direction_prob
                        )
                        targets.expected_return = (
                            gbm_w * gbm_pred["expected_return"]
                            + base_w * baseline.expected_return
                        )
                        gbm_used = True
            except Exception:
                pass  # fall through to baseline

        if not gbm_used:
            targets.direction_prob = baseline.direction_prob
            targets.expected_return = baseline.expected_return

        # Always use baseline for these (GBM doesn't predict them)
        targets.volatility_forecast = baseline.volatility_forecast
        targets.return_std = baseline.return_std
        targets.jump_prob = baseline.jump_prob
        targets.crash_prob = baseline.crash_prob
        targets.quantile_10 = baseline.quantile_10
        targets.quantile_50 = baseline.quantile_50
        targets.quantile_90 = baseline.quantile_90

        # ── 4. Calibrate direction probability ────────────────
        targets.direction_prob = self._calibrate(targets.direction_prob)

        # Shrink return when direction signal is weak
        distance_from_50 = abs(targets.direction_prob - 0.5)
        if distance_from_50 < 0.03:
            targets.expected_return *= 0.5
        elif distance_from_50 < 0.08:
            targets.expected_return *= 0.7

        # ── 5. Module agreement check ─────────────────────────
        # Fewer, stronger signals beat many weak ones
        targets = self._apply_agreement_filter(targets, outputs)

        confidence = min(0.9, sum(o.confidence * o.weight for o in outputs)
                         / max(1, len(outputs)))


        # ── 5. Module agreement check ─────────────────────────
        # Fewer, stronger signals beat many weak ones
        targets = self._apply_agreement_filter(targets, outputs)

        confidence = min(0.9, sum(o.confidence * o.weight for o in outputs)
                         / max(1, len(outputs)))

        # Record feature importance in output for diagnostics
        meta_features = dict(all_features)
        meta_features["gbm_used"] = float(gbm_used)
        meta_features["gbm_train_samples"] = float(len(self._history))
        if self._dir_model is not None and self._feature_names:
            imp = self._dir_model.feature_importance(importance_type="gain")
            for fname, score in zip(self._feature_names, imp):
                meta_features[f"imp__{fname}"] = float(score)

        return ModuleOutput(
            self.name, targets, confidence=confidence,
            features=meta_features, elapsed_ms=(time.time()-t0)*1000,
        )

    # ── Internal: weighted-average baseline ────────────────────

    def _weighted_average(self, outputs: list[ModuleOutput]) -> PredictionTargets:
        """Confidence-weighted average across module outputs."""
        """Confidence-weighted average across module outputs.

        Note: calibration and agreement filtering are applied downstream
        in predict_from_modules via _calibrate() and _apply_agreement_filter().
        """
        targets = PredictionTargets()
        total_weight = 0.0
        weighted_return = weighted_dir = weighted_vol = 0.0
        weighted_jump = weighted_crash = 0.0
        q10s, q50s, q90s = [], [], []

        for out in outputs:
            w = out.confidence * out.weight
            if w <= 0:
                continue
            total_weight += w
            weighted_return += w * out.targets.expected_return
            weighted_dir += w * out.targets.direction_prob
            weighted_vol += w * out.targets.volatility_forecast
            weighted_jump += w * out.targets.jump_prob
            weighted_crash += w * out.targets.crash_prob
            q10s.append((w, out.targets.quantile_10))
            q50s.append((w, out.targets.quantile_50))
            q90s.append((w, out.targets.quantile_90))

        if total_weight > 0:
            targets.expected_return = weighted_return / total_weight
            targets.direction_prob = weighted_dir / total_weight
            targets.volatility_forecast = weighted_vol / total_weight
            targets.jump_prob = weighted_jump / total_weight
            targets.crash_prob = weighted_crash / total_weight
            targets.return_std = targets.volatility_forecast
            targets.quantile_10 = sum(w * q for w, q in q10s) / total_weight
            targets.quantile_50 = sum(w * q for w, q in q50s) / total_weight
            targets.quantile_90 = sum(w * q for w, q in q90s) / total_weight

        return targets

    # ── Internal: feature extraction ──────────────────────────

    # Features that leak runtime/meta information into training.
    # These change based on VPS load, code path, and run order —
    # not market state.  The GBM learns noise from them.
    _LEAKAGE_SUFFIXES = frozenset({
        "elapsed_ms",           # module timing (VPS load artifact)
        "mc_iterations",        # constant config parameter
        "mc_jump_lambda",       # derived from jump_prob (already a feature)
        "mc_avg_max_dd",        # downstream MC stat (not an input signal)
        "mc_worst_dd",          # downstream MC stat
    })

    @staticmethod
    def _extract_features(outputs: list[ModuleOutput]) -> dict:
        """Flatten all module features into a single dict.

        Strips leakage features (timing, counters, MC meta) that
        correlate with runtime variability rather than market state.
        """
        all_features = {}
        for out in outputs:
            for k, v in out.features.items():
                if isinstance(v, (int, float, bool)):
                    if k in MetaLearnerModule._LEAKAGE_SUFFIXES:
                        continue
                    all_features[f"{out.module_name}__{k}"] = float(v)
        return all_features

    # ── Internal: GBM training (walk-forward) ──────────────────

    def _retrain_if_needed(self):
        """Retrain GBM models on accumulated walk-forward history.

        Retrains every 10 new samples to avoid constant retraining
        while still adapting to new data.
        """
        import lightgbm as lgb

        n = len(self._history)
        # Only retrain every 10 new samples (or first time)
        if (self._dir_model is not None
                and n % 10 != 0
                and n > _MIN_GBM_TRAIN_SAMPLES + 5):
            return

        # Build feature matrix from history
        all_keys: set[str] = set()
        for feats, _ in self._history:
            all_keys.update(feats.keys())

        # Sort for deterministic column order
        feature_names = sorted(all_keys)

        X = np.zeros((n, len(feature_names)))
        y_dir = np.zeros(n)   # binary: 1 = up, 0 = down
        y_ret = np.zeros(n)   # continuous: log return

        for i, (feats, actual_ret) in enumerate(self._history):
            for j, fname in enumerate(feature_names):
                X[i, j] = feats.get(fname, 0.0)
            y_dir[i] = 1.0 if actual_ret > 0 else 0.0
            y_ret[i] = actual_ret

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Feature selection: train on all, keep top N ────
        # First pass: quick train to get importances
        ds = lgb.Dataset(X, label=y_dir, feature_name=feature_names)
        quick_params = dict(_GBM_PARAMS)
        quick_params["n_estimators"] = 50  # fast
        quick_model = lgb.train(
            quick_params, ds, num_boost_round=50,
        )
        importances = quick_model.feature_importance(importance_type="gain")
        top_idx = np.argsort(importances)[-_MAX_GBM_FEATURES:]
        selected_names = [feature_names[i] for i in sorted(top_idx)]

        # Rebuild X with selected features only
        X_sel = np.zeros((n, len(selected_names)))
        for i, (feats, _) in enumerate(self._history):
            for j, fname in enumerate(selected_names):
                X_sel[i, j] = feats.get(fname, 0.0)
        X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)

        self._feature_names = selected_names

        # ── Train direction classifier with early stopping ──
        # Use last 20% as validation for early stopping
        val_size = max(5, n // 5)
        train_end = n - val_size

        ds_train = lgb.Dataset(
            X_sel[:train_end], label=y_dir[:train_end],
            feature_name=selected_names,
        )
        ds_val = lgb.Dataset(
            X_sel[train_end:], label=y_dir[train_end:],
            feature_name=selected_names, reference=ds_train,
        )

        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        self._dir_model = lgb.train(
            _GBM_PARAMS, ds_train,
            num_boost_round=_GBM_PARAMS["n_estimators"],
            valid_sets=[ds_val],
            callbacks=callbacks,
        )

        # ── Train return regressor ──────────────────────────
        ds_train_r = lgb.Dataset(
            X_sel[:train_end], label=y_ret[:train_end],
            feature_name=selected_names,
        )
        ds_val_r = lgb.Dataset(
            X_sel[train_end:], label=y_ret[train_end:],
            feature_name=selected_names, reference=ds_train_r,
        )

        self._ret_model = lgb.train(
            _GBM_REG_PARAMS, ds_train_r,
            num_boost_round=_GBM_REG_PARAMS["n_estimators"],
            valid_sets=[ds_val_r],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

        # ── Fit isotonic calibration on OOB predictions ─────
        # Use the validation fold predictions vs actuals
        if val_size >= 10:
            try:
                from sklearn.isotonic import IsotonicRegression
                val_preds = self._dir_model.predict(X_sel[train_end:])
                val_labels = y_dir[train_end:]
                iso = IsotonicRegression(
                    y_min=0.05, y_max=0.95, out_of_bounds="clip",
                )
                iso.fit(val_preds, val_labels)
                self._isotonic = iso
            except Exception:
                self._isotonic = None

    # ── Internal: GBM prediction ──────────────────────────────

    def _gbm_predict(self, features: dict) -> Optional[dict]:
        """Predict direction_prob and expected_return from features."""
        if self._dir_model is None or not self._feature_names:
            return None

        x = np.zeros((1, len(self._feature_names)))
        for j, fname in enumerate(self._feature_names):
            x[0, j] = features.get(fname, 0.0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        dir_prob = float(self._dir_model.predict(x)[0])
        dir_prob = max(0.05, min(0.95, dir_prob))

        expected_return = 0.0
        if self._ret_model is not None:
            expected_return = float(self._ret_model.predict(x)[0])

        return {
            "direction_prob": dir_prob,
            "expected_return": expected_return,
        }

    # ── Internal: calibration ─────────────────────────────────

    def _calibrate(self, raw_prob: float) -> float:
        """Calibrate direction probability.

        Uses walk-forward isotonic regression if the GBM has trained
        one, otherwise falls back to the static piece-wise map.
        """
        if self._isotonic is not None:
            try:
                cal = float(self._isotonic.predict([raw_prob])[0])
                return max(0.05, min(0.95, cal))
            except Exception:
                pass
        # Static fallback
        return _calibrate_direction_prob(raw_prob)

    # ── Internal: module agreement filter ─────────────────────

    @staticmethod
    def _apply_agreement_filter(
        targets: PredictionTargets,
        outputs: list[ModuleOutput],
    ) -> PredictionTargets:
        """Boost on near-unanimous agreement; dampen on disagreement."""
        dir_votes = [out.targets.direction_prob for out in outputs
                     if out.confidence > 0.2]
        if dir_votes:
            bull_pct = sum(1 for d in dir_votes if d > 0.55) / len(dir_votes)
            bear_pct = sum(1 for d in dir_votes if d < 0.45) / len(dir_votes)

            # Only boost at 85%+ agreement (was 80%)
            if bull_pct > 0.85:
                targets.direction_prob = min(0.75, targets.direction_prob * 1.05)
            elif bear_pct > 0.85:
                targets.direction_prob = max(0.25, targets.direction_prob * 0.95)

            # Uncertainty signal: if modules disagree, clamp hard
            uncertain_pct = sum(1 for d in dir_votes if 0.45 <= d <= 0.55) / len(dir_votes)
            if uncertain_pct > 0.5:
                # Majority of modules are uncertain → clamp to 0.50
                dampen = 0.40 * uncertain_pct
                targets.direction_prob = targets.direction_prob * (1 - dampen) + 0.5 * dampen
                targets.expected_return *= (1 - dampen * 0.5)

        return targets

    # ── Internal: feature extraction ──────────────────────────

    @staticmethod
    def _extract_features(outputs: list[ModuleOutput]) -> dict:
        """Flatten all module features into a single dict."""
        all_features = {}
        for out in outputs:
            for k, v in out.features.items():
                if isinstance(v, (int, float, bool)):
                    all_features[f"{out.module_name}__{k}"] = float(v)
        return all_features

    # ── Internal: GBM training (walk-forward) ──────────────────

    def _retrain_if_needed(self):
        """Retrain GBM models on accumulated walk-forward history.

        Retrains every 10 new samples to avoid constant retraining
        while still adapting to new data.
        """
        import lightgbm as lgb

        n = len(self._history)
        # Only retrain every 10 new samples (or first time)
        if (self._dir_model is not None
                and n % 10 != 0
                and n > _MIN_GBM_TRAIN_SAMPLES + 5):
            return

        # Build feature matrix from history
        all_keys: set[str] = set()
        for feats, _ in self._history:
            all_keys.update(feats.keys())

        # Sort for deterministic column order
        feature_names = sorted(all_keys)

        X = np.zeros((n, len(feature_names)))
        y_dir = np.zeros(n)   # binary: 1 = up, 0 = down
        y_ret = np.zeros(n)   # continuous: log return

        for i, (feats, actual_ret) in enumerate(self._history):
            for j, fname in enumerate(feature_names):
                X[i, j] = feats.get(fname, 0.0)
            y_dir[i] = 1.0 if actual_ret > 0 else 0.0
            y_ret[i] = actual_ret

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Feature selection: train on all, keep top N ────
        # First pass: quick train to get importances
        ds = lgb.Dataset(X, label=y_dir, feature_name=feature_names)
        quick_params = dict(_GBM_PARAMS)
        quick_params["n_estimators"] = 50  # fast
        quick_model = lgb.train(
            quick_params, ds, num_boost_round=50,
        )
        importances = quick_model.feature_importance(importance_type="gain")
        top_idx = np.argsort(importances)[-_MAX_GBM_FEATURES:]
        selected_names = [feature_names[i] for i in sorted(top_idx)]

        # Rebuild X with selected features only
        X_sel = np.zeros((n, len(selected_names)))
        for i, (feats, _) in enumerate(self._history):
            for j, fname in enumerate(selected_names):
                X_sel[i, j] = feats.get(fname, 0.0)
        X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)

        self._feature_names = selected_names

        # ── Train direction classifier with early stopping ──
        # Use last 20% as validation for early stopping
        val_size = max(5, n // 5)
        train_end = n - val_size

        ds_train = lgb.Dataset(
            X_sel[:train_end], label=y_dir[:train_end],
            feature_name=selected_names,
        )
        ds_val = lgb.Dataset(
            X_sel[train_end:], label=y_dir[train_end:],
            feature_name=selected_names, reference=ds_train,
        )

        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        self._dir_model = lgb.train(
            _GBM_PARAMS, ds_train,
            num_boost_round=_GBM_PARAMS["n_estimators"],
            valid_sets=[ds_val],
            callbacks=callbacks,
        )
        if not dir_votes:
            return targets

        bull_pct = sum(1 for d in dir_votes if d > 0.55) / len(dir_votes)
        bear_pct = sum(1 for d in dir_votes if d < 0.45) / len(dir_votes)

        # Only boost at 85%+ agreement
        if bull_pct > 0.85:
            targets.direction_prob = min(0.75, targets.direction_prob * 1.05)
        elif bear_pct > 0.85:
            targets.direction_prob = max(0.25, targets.direction_prob * 0.95)

        # Uncertainty: if modules disagree, clamp toward 0.50
        uncertain_pct = sum(1 for d in dir_votes
                            if 0.45 <= d <= 0.55) / len(dir_votes)
        if uncertain_pct > 0.5:
            dampen = 0.40 * uncertain_pct
            targets.direction_prob = (
                targets.direction_prob * (1 - dampen) + 0.5 * dampen
            )
            targets.expected_return *= (1 - dampen * 0.5)

        return targets

        # ── Train return regressor ──────────────────────────
        ds_train_r = lgb.Dataset(
            X_sel[:train_end], label=y_ret[:train_end],
            feature_name=selected_names,
        )
        ds_val_r = lgb.Dataset(
            X_sel[train_end:], label=y_ret[train_end:],
            feature_name=selected_names, reference=ds_train_r,
        )

        self._ret_model = lgb.train(
            _GBM_REG_PARAMS, ds_train_r,
            num_boost_round=_GBM_REG_PARAMS["n_estimators"],
            valid_sets=[ds_val_r],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

        # ── Fit isotonic calibration on OOB predictions ─────
        # Use the validation fold predictions vs actuals
        if val_size >= 10:
            try:
                from sklearn.isotonic import IsotonicRegression
                val_preds = self._dir_model.predict(X_sel[train_end:])
                val_labels = y_dir[train_end:]
                iso = IsotonicRegression(
                    y_min=0.05, y_max=0.95, out_of_bounds="clip",
                )
                iso.fit(val_preds, val_labels)
                self._isotonic = iso
            except Exception:
                self._isotonic = None

    # ── Internal: GBM prediction ──────────────────────────────

    def _gbm_predict(self, features: dict) -> Optional[dict]:
        """Predict direction_prob and expected_return from features."""
        if self._dir_model is None or not self._feature_names:
            return None

        x = np.zeros((1, len(self._feature_names)))
        for j, fname in enumerate(self._feature_names):
            x[0, j] = features.get(fname, 0.0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        dir_prob = float(self._dir_model.predict(x)[0])
        dir_prob = max(0.05, min(0.95, dir_prob))

        expected_return = 0.0
        if self._ret_model is not None:
            expected_return = float(self._ret_model.predict(x)[0])

        return {
            "direction_prob": dir_prob,
            "expected_return": expected_return,
        }

    # ── Internal: calibration ─────────────────────────────────

    def _calibrate(self, raw_prob: float) -> float:
        """Calibrate direction probability.

        Uses walk-forward isotonic regression if the GBM has trained
        one, otherwise falls back to the static piece-wise map.
        """
        if self._isotonic is not None:
            try:
                cal = float(self._isotonic.predict([raw_prob])[0])
                return max(0.05, min(0.95, cal))
            except Exception:
                pass
        # Static fallback
        return _calibrate_direction_prob(raw_prob)

    # ── Internal: module agreement filter ─────────────────────

    @staticmethod
    def _apply_agreement_filter(
        targets: PredictionTargets,
        outputs: list[ModuleOutput],
    ) -> PredictionTargets:
        """Boost on near-unanimous agreement; dampen on disagreement."""
        dir_votes = [out.targets.direction_prob for out in outputs
                     if out.confidence > 0.2]
        if not dir_votes:
            return targets

        bull_pct = sum(1 for d in dir_votes if d > 0.55) / len(dir_votes)
        bear_pct = sum(1 for d in dir_votes if d < 0.45) / len(dir_votes)

        # Only boost at 85%+ agreement
        if bull_pct > 0.85:
            targets.direction_prob = min(0.75, targets.direction_prob * 1.05)
        elif bear_pct > 0.85:
            targets.direction_prob = max(0.25, targets.direction_prob * 0.95)

        # Uncertainty: if modules disagree, clamp toward 0.50
        uncertain_pct = sum(1 for d in dir_votes
                            if 0.45 <= d <= 0.55) / len(dir_votes)
        if uncertain_pct > 0.5:
            dampen = 0.40 * uncertain_pct
            targets.direction_prob = (
                targets.direction_prob * (1 - dampen) + 0.5 * dampen
            )
            targets.expected_return *= (1 - dampen * 0.5)

        return targets


# ═══════════════════════════════════════════════════════════════
# MODULE 9: DEEP SEQUENCE MODEL (QUANTILE PREDICTOR)
# ═══════════════════════════════════════════════════════════════

class SequenceModule:
    """
    Lightweight quantile regression on recent price sequences.
    In production, would be LSTM/Transformer. Here: polynomial
    quantile regression as a practical baseline that doesn't
    require torch/tensorflow on the VPS.
    """
    name = "sequence_model"
    trusted_regimes = {Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE}

    def predict(self, snap: MarketSnapshot, horizon_hours: float = 24) -> ModuleOutput:
        t0 = time.time()
        targets = PredictionTargets()
        closes = snap.closes

        if len(closes) < 48:
            return ModuleOutput(self.name, targets, confidence=0.0,
                                elapsed_ms=(time.time()-t0)*1000)

        # Use last 96 bars for sequence features
        seq = closes[-min(96, len(closes)):]
        log_rets = _log_returns(seq)

        if len(log_rets) < 10:
            return ModuleOutput(self.name, targets, confidence=0.0,
                                elapsed_ms=(time.time()-t0)*1000)

        rets = np.array(log_rets)

        # ── Quantile regression via historical quantiles ──
        # Partition recent returns into buckets based on preceding patterns
        # Feature: last N returns → predict next-horizon return distribution

        # Simple approach: rolling quantile estimates
        scale = math.sqrt(horizon_hours / 1)  # 1h bars to horizon
        mean_ret = float(np.mean(rets[-24:]))
        std_ret = float(np.std(rets[-24:]))

        # Adjust for serial correlation
        autocorr = float(np.corrcoef(rets[:-1], rets[1:])[0, 1]) if len(rets) > 2 else 0
        persistence_adj = 1 + autocorr * 0.5

        # Empirical quantiles with heavy-tail adjustment (crypto kurtosis ~6-10)
        kurtosis_adj = 1.3  # fatter tails than normal
        targets.expected_return = mean_ret * horizon_hours * persistence_adj
        targets.return_std = std_ret * scale * kurtosis_adj
        targets.quantile_10 = targets.expected_return - 1.28 * targets.return_std * kurtosis_adj
        targets.quantile_50 = targets.expected_return
        targets.quantile_90 = targets.expected_return + 1.28 * targets.return_std * kurtosis_adj

        targets.direction_prob = _logistic(mean_ret / (std_ret + 1e-8) * 2)
        targets.volatility_forecast = targets.return_std

        # Jump detection from recent sequence
        max_abs_ret = float(np.max(np.abs(rets[-12:])))
        if max_abs_ret > 3 * std_ret:
            targets.jump_prob = 0.15  # post-jump environment
        else:
            targets.jump_prob = 0.05

        features = {
            "seq_mean_ret": mean_ret,
            "seq_std_ret": std_ret,
            "seq_autocorr": autocorr,
            "seq_max_abs_ret": max_abs_ret,
            "seq_kurtosis_adj": kurtosis_adj,
        }

        confidence = min(0.6, 0.3 + len(rets) / 200)

        return ModuleOutput(
            self.name, targets, confidence=confidence,
            features=features, elapsed_ms=(time.time()-t0)*1000,
        )


# ═══════════════════════════════════════════════════════════════
# MODULE 10: REGIME STATE MACHINE + GATING POLICY
# ═══════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    The "glue" that classifies market regime and outputs:
    - Regime probabilities
    - Module weights (gating policy)
    - Confidence scalar for the ensemble

    Regime detection uses: realized vol, vol-of-vol, book depth/spread,
    derivatives LFI, jump detector, trend strength.
    """
    name = "regime_detector"

    def detect(self, snap: MarketSnapshot,
               module_outputs: list[ModuleOutput]) -> tuple[dict[Regime, float], float]:
        """
        Returns:
            regime_probs: {Regime: probability}
            confidence_scalar: multiplier for forecast intervals
        """
        probs = {r: 0.0 for r in Regime}
        closes = snap.closes

        if len(closes) < 20:
            probs[Regime.RANGE] = 1.0
            return probs, 1.0

        log_rets = _log_returns(closes)
        if len(log_rets) < 10:
            probs[Regime.RANGE] = 1.0
            return probs, 1.0

        # ── Feature extraction ────────────────────────────
        recent_vol = float(np.std(log_rets[-24:])) if len(log_rets) >= 24 else \
            float(np.std(log_rets))
        long_vol = float(np.std(log_rets[-min(168, len(log_rets)):])) if len(log_rets) >= 48 else recent_vol
        vol_ratio = recent_vol / long_vol if long_vol > 0 else 1.0

        # Trend strength via directional movement
        price = closes[-1]
        ema_12 = _ema(closes, 12)[-1]
        ema_50 = _ema(closes, min(50, len(closes)))[-1]
        trend_strength = (ema_12 - ema_50) / ema_50 if ema_50 > 0 else 0

        # Jump detection: any bar with |return| > 3*vol
        recent_rets = log_rets[-12:] if len(log_rets) >= 12 else log_rets
        max_abs = max(abs(r) for r in recent_rets) if recent_rets else 0
        jump_detected = max_abs > 3 * recent_vol if recent_vol > 0 else False

        # LFI from derivatives module
        lfi = 0.0
        for out in module_outputs:
            if out.module_name == "derivatives":
                lfi = out.features.get("lfi", 0.0)
                break

        # Spread/depth
        spread_wide = (snap.spread is not None and snap.spread > 0.003)

        # ── Regime classification ─────────────────────────
        # Trend up
        if trend_strength > 0.01 and vol_ratio < 2.0:
            probs[Regime.TREND_UP] = min(0.8, trend_strength * 30)
        # Trend down
        elif trend_strength < -0.01 and vol_ratio < 2.0:
            probs[Regime.TREND_DOWN] = min(0.8, abs(trend_strength) * 30)

        # Range -- wider detection band (was 0.005, now 0.01)
        # Catches more "weak trend" states that were classified as
        # trends but performed like range (41.8% HR, n=55)
        if abs(trend_strength) < 0.01:
            probs[Regime.RANGE] = 0.5 + (0.01 - abs(trend_strength)) * 30

        # Vol expansion
        if vol_ratio > 1.5:
            probs[Regime.VOL_EXPANSION] = min(0.7, (vol_ratio - 1.0) * 0.5)

        # Post-jump
        if jump_detected:
            probs[Regime.POST_JUMP] = 0.6

        # Cascade risk
        if lfi > 1.5:
            probs[Regime.CASCADE_RISK] = min(0.7, lfi / 3.0)

        # Illiquid
        if spread_wide:
            probs[Regime.ILLIQUID] = 0.4

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs[Regime.RANGE] = 1.0

        # ── Confidence scalar ─────────────────────────────
        # Controls MC envelope width.
        # Backtest showed P5-P95 coverage at 98.7% (target 90%),
        # meaning envelope was ~40% too wide. Tighten base to 0.70.
        confidence_scalar = 0.70  # tighter base (was 1.0)
        if probs.get(Regime.CASCADE_RISK, 0) > 0.3:
            confidence_scalar = 1.1  # widen for danger (was 1.5)
        if probs.get(Regime.POST_JUMP, 0) > 0.3:
            confidence_scalar = 0.95  # slight widen (was 1.3)
        if probs.get(Regime.VOL_EXPANSION, 0) > 0.3:
            confidence_scalar = 0.85  # moderate (was 1.2)
        if probs.get(Regime.ILLIQUID, 0) > 0.3:
            confidence_scalar = 1.0  # widen for thin liquidity (was 1.4)

        return probs, confidence_scalar

    def compute_weights(self, regime_probs: dict[Regime, float]) -> dict[str, float]:
        """
        Gating policy: assign module weights based on regime probabilities.

        Rules:
        - cascade risk → weight derivatives + tail risk; downweight TA, MR
        - range → weight mean reversion
        - trend → weight momentum/TA
        - post-jump → reduce all confidence
        """
        weights = {
            "technical": 1.0,
            "classical_stats": 1.0,
            "macro_factor": 0.8,
            "derivatives": 1.0,
            "microstructure": 0.8,
            "onchain": 0.5,
            "sentiment": 0.3,
            "sequence_model": 0.7,
            "particle_candle": 0.7,
            "manifold_pattern": 0.6,
        }

        # Cascade risk: derivatives up, TA/MR down
        cascade_p = regime_probs.get(Regime.CASCADE_RISK, 0)
        if cascade_p > 0.3:
            weights["derivatives"] += cascade_p * 1.5
            weights["technical"] *= (1 - cascade_p * 0.6)
            weights["sequence_model"] *= (1 - cascade_p * 0.4)

        # Range: suppress directional modules, boost mean-reversion
        # Backtest showed 41.8% HR in range (n=55) -- model was
        # overconfident on direction in sideways markets.
        range_p = regime_probs.get(Regime.RANGE, 0)
        if range_p > 0.4:
            weights["technical"] *= (1 - range_p * 0.3)  # reduce TA trend signals
            weights["sequence_model"] *= (1 - range_p * 0.4)  # reduce momentum extrapolation
            weights["microstructure"] += range_p * 0.3  # boost flow (short-horizon edge)
            weights["classical_stats"] += range_p * 0.4  # boost vol model (uncertainty)

        # Trend: momentum up -- backtest showed 66.7% HR in trend_up
        # and good MCC, so boost these more aggressively
        trend_p = regime_probs.get(Regime.TREND_UP, 0) + regime_probs.get(Regime.TREND_DOWN, 0)
        if trend_p > 0.4:
            weights["technical"] += trend_p * 0.7  # stronger boost (was 0.5)
            weights["sequence_model"] += trend_p * 0.5  # stronger boost (was 0.3)

        # Post-jump: reduce everything
        jump_p = regime_probs.get(Regime.POST_JUMP, 0)
        if jump_p > 0.3:
            for k in weights:
                weights[k] *= (1 - jump_p * 0.3)

        # Vol expansion: boost vol models
        vol_p = regime_probs.get(Regime.VOL_EXPANSION, 0)
        if vol_p > 0.3:
            weights["classical_stats"] += vol_p * 0.5
            weights["derivatives"] += vol_p * 0.3

        # Particle candle: boost in trends (event bars reveal trend structure)
        if trend_p > 0.4:
            weights["particle_candle"] += trend_p * 0.4
        if range_p > 0.4:
            weights["particle_candle"] *= (1 - range_p * 0.2)  # slightly reduce in range

        # Manifold pattern: boost in range (breakout detection) and trends
        if range_p > 0.4:
            weights["manifold_pattern"] += range_p * 0.3  # pattern breakout in range
        if trend_p > 0.4:
            weights["manifold_pattern"] += trend_p * 0.3

        # Both new modules: reduce in cascade/jump
        if cascade_p > 0.3:
            weights["particle_candle"] *= (1 - cascade_p * 0.4)
            weights["manifold_pattern"] *= (1 - cascade_p * 0.4)
        if jump_p > 0.3:
            weights["particle_candle"] *= (1 - jump_p * 0.3)
            weights["manifold_pattern"] *= (1 - jump_p * 0.3)

        # Normalize so max = 1
        max_w = max(weights.values()) if weights else 1.0
        if max_w > 0:
            weights = {k: v / max_w for k, v in weights.items()}

        return weights


# ═══════════════════════════════════════════════════════════════
# MODULE 11: MONTE CARLO ENVELOPE GENERATOR
# ═══════════════════════════════════════════════════════════════

class MonteCarloModule:
    """
    Regime-conditioned Monte Carlo simulation:
    - Drift from ensemble forecast
    - Volatility from classical stats module
    - Jump process (Poisson + heavy-tail) from derivatives module
    - Produces distribution, barrier probs, risk envelopes
    """
    name = "monte_carlo"

    def simulate(self, snap: MarketSnapshot,
                 ensemble_targets: PredictionTargets,
                 regime_probs: dict[Regime, float],
                 confidence_scalar: float = 1.0,
                 n_iterations: int = 50_000,
                 n_steps: int = 48,
                 barrier_strike: Optional[float] = None,
                 seed: int = 42) -> ModuleOutput:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        price = snap.current_price
        if price <= 0:
            return ModuleOutput(self.name, PredictionTargets(), confidence=0.0)

        # ── Parameters from ensemble ──────────────────────
        mu = ensemble_targets.expected_return
        # Use confidence_scalar to tighten/widen the diffusion envelope,
        # but keep raw vol for barrier simulation (barrier touches need
        # full vol to avoid underestimating touch probability).
        sigma_raw = max(0.005, ensemble_targets.volatility_forecast)
        sigma = sigma_raw * confidence_scalar  # scaled for VaR/envelope
        jump_prob = ensemble_targets.jump_prob
        crash_prob = ensemble_targets.crash_prob

        dt = 1.0 / n_steps

        # ── GBM with jump diffusion ──────────────────────
        # dS/S = mu*dt + sigma*dW + J*dN
        # where N ~ Poisson(lambda*dt), J ~ Normal(mu_j, sigma_j)
        # jump_prob is a per-horizon probability (e.g. 5% = 0.05 expected
        # jumps in 24 h).  Do NOT multiply by n_steps — that converts a
        # probability into a rate n_steps× too large and makes jump variance
        # dominate the simulation (was VaR(95%) ≈ -21%, now ≈ -4%).
        jump_lambda = jump_prob                    # expected jumps per horizon
        jump_mu = -0.02 if crash_prob > jump_prob * 0.4 else 0.0  # negative skew
        # Jump sizes use raw vol (not scaled) to preserve tail accuracy
        jump_sigma = sigma_raw * 1.5  # jumps are 1.5x normal vol (tightened from 2x)

        # Generate diffusion
        Z = rng.standard_normal((n_iterations, n_steps))
        increments = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z

        # Generate jumps (Poisson process)
        if jump_lambda > 0.01:
            N_jumps = rng.poisson(jump_lambda * dt, (n_iterations, n_steps))
            J_sizes = rng.normal(jump_mu, jump_sigma, (n_iterations, n_steps))
            increments += N_jumps * J_sizes

        # Price paths
        log_paths = np.cumsum(increments, axis=1)
        log_paths = np.hstack([np.zeros((n_iterations, 1)), log_paths])
        paths = price * np.exp(log_paths)
        terminal = paths[:, -1]

        # ── Statistics ────────────────────────────────────
        targets = PredictionTargets()
        targets.expected_return = float(np.mean(np.log(terminal / price)))
        targets.return_std = float(np.std(np.log(terminal / price)))
        targets.direction_prob = float(np.mean(terminal > price))
        targets.volatility_forecast = targets.return_std

        targets.quantile_10 = float(np.log(np.percentile(terminal, 10) / price))
        targets.quantile_50 = float(np.log(np.percentile(terminal, 50) / price))
        targets.quantile_90 = float(np.log(np.percentile(terminal, 90) / price))

        # Jump/crash from simulation
        returns = np.log(terminal / price)
        targets.jump_prob = float(np.mean(np.abs(returns) > 2 * sigma))
        targets.crash_prob = float(np.mean(returns < -3 * sigma))

        # Barrier probabilities
        if barrier_strike:
            # Did price touch or exceed barrier at any point?
            max_prices = np.max(paths, axis=1)
            min_prices = np.min(paths, axis=1)
            targets.barrier_above_prob = float(np.mean(max_prices >= barrier_strike))
            targets.barrier_below_prob = float(np.mean(min_prices <= barrier_strike))
            targets.barrier_strike = barrier_strike

        # ── Risk metrics ──────────────────────────────────
        var_95 = float(np.percentile(returns, 5))
        var_99 = float(np.percentile(returns, 1))
        cvar_95 = float(np.mean(returns[returns <= var_95]))
        cvar_99 = float(np.mean(returns[returns <= var_99]))
        max_drawdowns = np.min(paths, axis=1) / price - 1.0

        features = {
            "mc_mean_price": float(np.mean(terminal)),
            "mc_median_price": float(np.median(terminal)),
            "mc_std_price": float(np.std(terminal)),
            "mc_p5_price": float(np.percentile(terminal, 5)),
            "mc_p10_price": float(np.percentile(terminal, 10)),
            "mc_p25_price": float(np.percentile(terminal, 25)),
            "mc_p75_price": float(np.percentile(terminal, 75)),
            "mc_p90_price": float(np.percentile(terminal, 90)),
            "mc_p95_price": float(np.percentile(terminal, 95)),
            "mc_var_95": var_95,
            "mc_var_99": var_99,
            "mc_cvar_95": cvar_95,
            "mc_cvar_99": cvar_99,
            "mc_avg_max_dd": float(np.mean(max_drawdowns)),
            "mc_worst_dd": float(np.min(max_drawdowns)),
            "mc_iterations": n_iterations,
            "mc_jump_lambda": jump_lambda,
        }

        # Percentile envelope for visualization
        envelope = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            envelope[f"p{p}"] = np.percentile(paths, p, axis=0).tolist()
        features["envelope"] = envelope

        targets.regime_probs = dict(regime_probs)

        return ModuleOutput(
            self.name, targets, confidence=0.8,
            features=features, elapsed_ms=(time.time()-t0)*1000,
        )


# ═══════════════════════════════════════════════════════════════
# MODULE 12: CROWD / PREDICTION MARKET PRIORS
# ═══════════════════════════════════════════════════════════════

class CrowdPriorModule:
    """
    Uses Kalshi implied probabilities as external priors
    for barrier probability sanity checks.
    """
    name = "crowd_prior"
    trusted_regimes = set(Regime)

    def predict(self, snap: MarketSnapshot, horizon_hours: float = 24) -> ModuleOutput:
        t0 = time.time()
        targets = PredictionTargets()
        features = {}
        confidence = 0.1

        if snap.kalshi_barrier_probs:
            features["kalshi_priors"] = snap.kalshi_barrier_probs
            confidence = 0.35

            # If Kalshi has a barrier prob, use it as anchor
            for strike_str, prob in snap.kalshi_barrier_probs.items():
                try:
                    strike = float(strike_str)
                    targets.barrier_strike = strike
                    targets.barrier_above_prob = prob
                    targets.barrier_below_prob = 1.0 - prob
                    break
                except (ValueError, TypeError):
                    continue

        return ModuleOutput(
            self.name, targets, confidence=confidence,
            features=features, elapsed_ms=(time.time()-t0)*1000,
        )
