"""
Particle Candles — Event-Quantized Bar Construction
=====================================================
Replaces fixed-time candles with event-quantized aggregates that are
invariant to market speed and regime.

Core ideas:
  1. Event-quantized bars: aggregate by N-trades, Δ-volume, or Δ-entropy
     instead of fixed clock intervals.
  2. Simplex geometry: map each candle onto the (B, W_u, W_l) simplex where
     B = body ratio, W_u = upper wick ratio, W_l = lower wick ratio.
     Constraint: B + W_u + W_l = 1 (all points on a 2-simplex).
  3. Per-bar microstructure features: OFI, realized vol, VWAP displacement,
     entropy gradient — unified feature vector X_j for pattern detection.
  4. Integration with existing forecaster as ParticleCandleModule.

Event rules (choose 2–3):
  - TRADE_COUNT: fixed N trades per bar
  - VOLUME_DELTA: fixed Δ volume per bar
  - ENTROPY_INCREMENT: fixed entropy gain per bar

Usage:
    from scripts.forecaster.particle_candles import (
        ParticleCandleBuilder,
        ParticleCandleModule,
        EventBar,
    )
    builder = ParticleCandleBuilder(rule="volume_delta", threshold=1000.0)
    bars = builder.from_ohlcv(snapshot.bars_1h)
    features = builder.compute_features(bars)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .schemas import (
    Bar, MarketSnapshot, ModuleOutput, PredictionTargets, Regime,
)


# ═══════════════════════════════════════════════════════════════
# EVENT BAR DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════

@dataclass
class EventBar:
    """
    Single event-quantized bar with full microstructure state.

    Each bar j aggregates a bucket E_j of market events defined
    by the chosen quantization rule (trade count, volume, entropy).
    """
    # Identity
    index: int = 0
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    n_events: int = 0          # trades / ticks aggregated

    # OHLCV
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0

    # Simplex geometry (B + W_u + W_l = 1)
    body_ratio: float = 0.333   # B = |C - O| / R
    wick_upper: float = 0.333   # W_u = (H - max(O,C)) / R
    wick_lower: float = 0.333   # W_l = (min(O,C) - L) / R

    # Direction
    is_bullish: bool = True     # C >= O

    # Microstructure
    ofi: float = 0.0            # order flow imbalance
    delta_volume: float = 0.0   # Δ volume from prior bar
    delta_liquidity: float = 0.0  # Δ liquidity
    realized_vol: float = 0.0   # realized volatility within bar
    vwap: float = 0.0           # volume-weighted average price
    d_vwap: float = 0.0         # displacement from VWAP
    entropy: float = 0.0        # information entropy of returns
    entropy_gradient: float = 0.0  # Δ entropy from prior bar

    # Derived
    range_pct: float = 0.0      # (H-L)/O as percentage
    body_direction: float = 0.0  # signed: (C-O)/R


@dataclass
class EventBarSequence:
    """Rolling window of m event bars for motif detection."""
    bars: list[EventBar] = field(default_factory=list)
    symbol: str = ""

    @property
    def n_bars(self) -> int:
        return len(self.bars)

    @property
    def simplex_coords(self) -> np.ndarray:
        """Return (n, 3) array of [B, W_u, W_l] for all bars."""
        if not self.bars:
            return np.zeros((0, 3))
        return np.array([
            [b.body_ratio, b.wick_upper, b.wick_lower]
            for b in self.bars
        ])

    @property
    def feature_matrix(self) -> np.ndarray:
        """
        Return (n, 9) feature matrix X_j per bar.
        Columns: B, W_u, W_l, ΔV, Δℓ, OFI, realized_vol, d_VWAP, entropy_gradient
        """
        if not self.bars:
            return np.zeros((0, 9))
        return np.array([
            [
                b.body_ratio,
                b.wick_upper,
                b.wick_lower,
                b.delta_volume,
                b.delta_liquidity,
                b.ofi,
                b.realized_vol,
                b.d_vwap,
                b.entropy_gradient,
            ]
            for b in self.bars
        ])

    def envelope_metrics(self) -> dict:
        """
        Compute envelope/motion metrics for the sequence.

        Returns slopes, contraction rates, wick asymmetry,
        OFI trend, and curvature for motif classification.
        """
        if self.n_bars < 3:
            return {}

        highs = np.array([b.high for b in self.bars])
        lows = np.array([b.low for b in self.bars])
        bodies_top = np.array([max(b.open, b.close) for b in self.bars])
        bodies_bot = np.array([min(b.open, b.close) for b in self.bars])
        ofis = np.array([b.ofi for b in self.bars])
        wicks_u = np.array([b.wick_upper for b in self.bars])
        wicks_l = np.array([b.wick_lower for b in self.bars])

        t = np.arange(self.n_bars, dtype=float)

        metrics = {}

        # Envelope slopes (linear regression on highs/lows)
        if self.n_bars >= 3:
            try:
                b_h = float(np.polyfit(t, highs, 1)[0])
                b_l = float(np.polyfit(t, lows, 1)[0])
                metrics["slope_high"] = b_h
                metrics["slope_low"] = b_l
                metrics["slope_mid"] = (b_h + b_l) / 2.0
            except (np.linalg.LinAlgError, ValueError):
                metrics["slope_high"] = 0.0
                metrics["slope_low"] = 0.0
                metrics["slope_mid"] = 0.0

        # Contraction rate: how fast the range narrows
        ranges = highs - lows
        if len(ranges) >= 3:
            range_slope = float(np.polyfit(t, ranges, 1)[0])
            avg_range = float(np.mean(ranges))
            metrics["contraction_rate"] = (
                range_slope / avg_range if avg_range > 0 else 0.0
            )
            metrics["range_slope"] = range_slope
            metrics["avg_range"] = avg_range
        else:
            metrics["contraction_rate"] = 0.0
            metrics["range_slope"] = 0.0
            metrics["avg_range"] = 0.0

        # Wick asymmetry: mean(W_u) - mean(W_l)
        metrics["wick_asymmetry"] = float(np.mean(wicks_u) - np.mean(wicks_l))

        # OFI trend (slope of OFI)
        if self.n_bars >= 3:
            try:
                metrics["ofi_trend"] = float(np.polyfit(t, ofis, 1)[0])
            except (np.linalg.LinAlgError, ValueError):
                metrics["ofi_trend"] = 0.0
        else:
            metrics["ofi_trend"] = 0.0

        # Curvature (2nd derivative of midprice)
        mid = (highs + lows) / 2.0
        if self.n_bars >= 4:
            try:
                poly = np.polyfit(t, mid, 2)
                metrics["curvature"] = float(2 * poly[0])
            except (np.linalg.LinAlgError, ValueError):
                metrics["curvature"] = 0.0
        else:
            metrics["curvature"] = 0.0

        # Body convergence: are bodies getting smaller?
        body_sizes = bodies_top - bodies_bot
        if len(body_sizes) >= 3:
            try:
                metrics["body_convergence"] = float(
                    np.polyfit(t, body_sizes, 1)[0]
                )
            except (np.linalg.LinAlgError, ValueError):
                metrics["body_convergence"] = 0.0
        else:
            metrics["body_convergence"] = 0.0

        # Entropy trend
        entropies = np.array([b.entropy for b in self.bars])
        if len(entropies) >= 3:
            try:
                metrics["entropy_trend"] = float(
                    np.polyfit(t, entropies, 1)[0]
                )
            except (np.linalg.LinAlgError, ValueError):
                metrics["entropy_trend"] = 0.0
        else:
            metrics["entropy_trend"] = 0.0

        return metrics


# ═══════════════════════════════════════════════════════════════
# EVENT-QUANTIZED BAR BUILDER
# ═══════════════════════════════════════════════════════════════

class ParticleCandleBuilder:
    """
    Builds event-quantized bars from standard OHLCV candles.

    Rules:
      - volume_delta:  aggregate until cumulative volume >= threshold
      - trade_count:   aggregate until N bars consumed (proxy for N trades)
      - entropy_incr:  aggregate until cumulative entropy >= threshold
      - adaptive:      volume_delta with regime-adaptive threshold

    When working from 1h OHLCV bars (no tick data), we approximate:
      - N trades ≈ volume / avg_trade_size
      - OFI ≈ sign(close - open) * volume
      - Entropy from return distribution within bar
    """

    RULES = ("volume_delta", "trade_count", "entropy_incr", "adaptive")

    def __init__(self,
                 rule: str = "volume_delta",
                 threshold: float = 0.0,
                 adaptive_lookback: int = 20):
        if rule not in self.RULES:
            raise ValueError(f"Unknown rule: {rule}. Choose from {self.RULES}")
        self.rule = rule
        self._base_threshold = threshold
        self.threshold = threshold
        self.adaptive_lookback = adaptive_lookback

    def from_ohlcv(self, bars: list[Bar],
                    recent_trades: list[dict] = None) -> list[EventBar]:
        """
        Convert standard OHLCV bars into event-quantized bars.

        Auto-calibrates threshold if not set (median volume / 2
        for volume_delta, or 3 bars for trade_count).
        """
        if not bars or len(bars) < 2:
            return []

        # Auto-calibrate threshold
        if self.threshold <= 0:
            self._auto_threshold(bars)

        if self.rule == "volume_delta":
            return self._build_volume_bars(bars)
        elif self.rule == "trade_count":
            return self._build_trade_count_bars(bars)
        elif self.rule == "entropy_incr":
            return self._build_entropy_bars(bars)
        elif self.rule == "adaptive":
            return self._build_adaptive_bars(bars)
        return []

    def _auto_threshold(self, bars: list[Bar]):
        """Auto-calibrate threshold based on data statistics."""
        vols = [b.volume for b in bars if b.volume > 0]
        if not vols:
            self.threshold = 1.0
            return

        if self.rule in ("volume_delta", "adaptive"):
            # Target: ~2-3 source bars per event bar
            median_vol = float(np.median(vols))
            self.threshold = median_vol * 2.5
        elif self.rule == "trade_count":
            self.threshold = 3.0  # Aggregate every 3 source bars
        elif self.rule == "entropy_incr":
            # Estimate per-bar entropy from the data itself
            # For typical hourly crypto, entropy per bar is small (~0.01-0.05 nats)
            # Auto-calibrate so we get ~20-40 event bars from input
            sample_bars = bars[:min(20, len(bars))]
            entropy_per_bar = []
            for b in sample_bars:
                if b.open > 0 and b.close > 0:
                    ret = abs(math.log(b.close / b.open))
                    if ret > 1e-10:
                        p = min(0.999, max(0.001, ret))
                        entropy_per_bar.append(-p * math.log(p))
            if entropy_per_bar:
                avg_entropy = float(np.mean(entropy_per_bar))
                # Accumulate ~3-5 bars worth of entropy per event bar
                self.threshold = max(0.001, avg_entropy * 3.0)
            else:
                self.threshold = 0.01

    def _build_volume_bars(self, bars: list[Bar]) -> list[EventBar]:
        """Aggregate bars until cumulative volume >= threshold."""
        event_bars = []
        accum = _BarAccumulator()

        for bar in bars:
            accum.add(bar)
            if accum.volume >= self.threshold:
                eb = accum.finalize(len(event_bars))
                event_bars.append(eb)
                accum = _BarAccumulator()

        # Final partial bar (if significant)
        if accum.count > 0 and accum.volume > self.threshold * 0.3:
            event_bars.append(accum.finalize(len(event_bars)))

        self._compute_inter_bar_features(event_bars)
        return event_bars

    def _build_trade_count_bars(self, bars: list[Bar]) -> list[EventBar]:
        """Aggregate every N source bars into one event bar."""
        n = max(1, int(self.threshold))
        event_bars = []
        accum = _BarAccumulator()

        for bar in bars:
            accum.add(bar)
            if accum.count >= n:
                event_bars.append(accum.finalize(len(event_bars)))
                accum = _BarAccumulator()

        if accum.count > 0:
            event_bars.append(accum.finalize(len(event_bars)))

        self._compute_inter_bar_features(event_bars)
        return event_bars

    def _build_entropy_bars(self, bars: list[Bar]) -> list[EventBar]:
        """Aggregate bars until cumulative entropy >= threshold."""
        event_bars = []
        accum = _BarAccumulator()

        for bar in bars:
            accum.add(bar)
            if accum.entropy >= self.threshold:
                event_bars.append(accum.finalize(len(event_bars)))
                accum = _BarAccumulator()

        if accum.count > 0 and accum.entropy > self.threshold * 0.3:
            event_bars.append(accum.finalize(len(event_bars)))

        self._compute_inter_bar_features(event_bars)
        return event_bars

    def _build_adaptive_bars(self, bars: list[Bar]) -> list[EventBar]:
        """
        Volume-delta with regime-adaptive threshold.

        In high-volatility regimes, use smaller threshold (more granular bars).
        In low-volatility, use larger threshold (fewer, smoother bars).
        """
        event_bars = []
        accum = _BarAccumulator()
        recent_vols: list[float] = []

        for bar in bars:
            accum.add(bar)

            # Adaptive threshold from recent bar ranges
            if bar.high > 0 and bar.low > 0:
                range_pct = (bar.high - bar.low) / bar.low
                recent_vols.append(range_pct)
                if len(recent_vols) > self.adaptive_lookback:
                    recent_vols.pop(0)

            current_threshold = self.threshold
            if len(recent_vols) >= 5:
                avg_range = float(np.mean(recent_vols))
                # More bars when volatile, fewer when calm
                vol_scale = max(0.5, min(2.0,
                    float(np.median(recent_vols)) / avg_range
                    if avg_range > 0 else 1.0
                ))
                current_threshold = self.threshold * vol_scale

            if accum.volume >= current_threshold:
                event_bars.append(accum.finalize(len(event_bars)))
                accum = _BarAccumulator()

        if accum.count > 0 and accum.volume > self.threshold * 0.2:
            event_bars.append(accum.finalize(len(event_bars)))

        self._compute_inter_bar_features(event_bars)
        return event_bars

    def _compute_inter_bar_features(self, event_bars: list[EventBar]):
        """Compute features that depend on neighboring bars."""
        for i in range(len(event_bars)):
            eb = event_bars[i]
            if i > 0:
                prev = event_bars[i - 1]
                eb.delta_volume = eb.volume - prev.volume
                eb.delta_liquidity = (
                    (eb.volume / max(1, eb.n_events))
                    - (prev.volume / max(1, prev.n_events))
                )
                eb.entropy_gradient = eb.entropy - prev.entropy


    def compute_features(self, event_bars: list[EventBar]) -> dict[str, float]:
        """
        Compute aggregate features from event bar sequence.

        Returns a flat dict suitable for the meta-learner.
        """
        if not event_bars:
            return {}

        feats = {}
        n = len(event_bars)

        # Simplex statistics
        bodies = [b.body_ratio for b in event_bars]
        wicks_u = [b.wick_upper for b in event_bars]
        wicks_l = [b.wick_lower for b in event_bars]

        feats["pc__n_event_bars"] = float(n)
        feats["pc__mean_body"] = float(np.mean(bodies))
        feats["pc__std_body"] = float(np.std(bodies)) if n > 1 else 0.0
        feats["pc__mean_wick_upper"] = float(np.mean(wicks_u))
        feats["pc__mean_wick_lower"] = float(np.mean(wicks_l))
        feats["pc__wick_asymmetry"] = feats["pc__mean_wick_upper"] - feats["pc__mean_wick_lower"]

        # Direction statistics
        bullish_pct = sum(1 for b in event_bars if b.is_bullish) / n
        feats["pc__bullish_pct"] = bullish_pct

        # Consecutive direction streaks
        max_streak = _max_streak(event_bars)
        feats["pc__max_consec_dir"] = float(max_streak)

        # OFI statistics
        ofis = [b.ofi for b in event_bars]
        feats["pc__mean_ofi"] = float(np.mean(ofis))
        feats["pc__ofi_momentum"] = float(np.sum(ofis[-5:])) if n >= 5 else 0.0

        # Volatility (realized vol across event bars)
        rvols = [b.realized_vol for b in event_bars if b.realized_vol > 0]
        if rvols:
            feats["pc__mean_rvol"] = float(np.mean(rvols))
            feats["pc__rvol_trend"] = (
                float(np.mean(rvols[-3:])) - float(np.mean(rvols[:3]))
                if n >= 6 else 0.0
            )

        # Entropy statistics
        entropies = [b.entropy for b in event_bars]
        feats["pc__mean_entropy"] = float(np.mean(entropies))
        feats["pc__entropy_accel"] = (
            float(np.mean([b.entropy_gradient for b in event_bars[-3:]]))
            if n >= 3 else 0.0
        )

        # Range contraction/expansion
        ranges = [b.range_pct for b in event_bars]
        if n >= 4:
            t = np.arange(n, dtype=float)
            try:
                feats["pc__range_slope"] = float(np.polyfit(t, ranges, 1)[0])
            except (np.linalg.LinAlgError, ValueError):
                feats["pc__range_slope"] = 0.0
        else:
            feats["pc__range_slope"] = 0.0

        # VWAP displacement trend
        d_vwaps = [b.d_vwap for b in event_bars]
        feats["pc__mean_d_vwap"] = float(np.mean(d_vwaps))

        return feats


# ═══════════════════════════════════════════════════════════════
# BAR ACCUMULATOR (internal)
# ═══════════════════════════════════════════════════════════════

class _BarAccumulator:
    """Accumulates source bars into a single event bar."""

    def __init__(self):
        self.count = 0
        self.open = 0.0
        self.high = -math.inf
        self.low = math.inf
        self.close = 0.0
        self.volume = 0.0
        self.timestamp_start = 0.0
        self.timestamp_end = 0.0
        self._prices: list[float] = []
        self._volumes: list[float] = []
        self.entropy = 0.0

    def add(self, bar: Bar):
        """Add a source OHLCV bar to the accumulator."""
        if self.count == 0:
            self.open = bar.open
            self.timestamp_start = bar.timestamp
        self.high = max(self.high, bar.high)
        self.low = min(self.low, bar.low)
        self.close = bar.close
        self.volume += bar.volume
        self.timestamp_end = bar.timestamp
        self.count += 1

        # Track prices for intra-bar features
        self._prices.extend([bar.open, bar.high, bar.low, bar.close])
        self._volumes.append(bar.volume)

        # Accumulate entropy
        if len(self._prices) >= 4:
            ret = math.log(bar.close / bar.open) if bar.open > 0 and bar.close > 0 else 0.0
            if abs(ret) > 1e-10:
                p = min(0.999, max(0.001, abs(ret)))
                self.entropy += -p * math.log(p)

    def finalize(self, index: int) -> EventBar:
        """Produce an EventBar from accumulated data."""
        eb = EventBar(
            index=index,
            timestamp_start=self.timestamp_start,
            timestamp_end=self.timestamp_end,
            n_events=self.count,
            open=self.open,
            high=self.high if self.high > -math.inf else self.open,
            low=self.low if self.low < math.inf else self.open,
            close=self.close,
            volume=self.volume,
        )

        # Simplex geometry
        R = eb.high - eb.low
        if R > 0:
            body = abs(eb.close - eb.open)
            top = max(eb.open, eb.close)
            bot = min(eb.open, eb.close)

            eb.body_ratio = body / R
            eb.wick_upper = (eb.high - top) / R
            eb.wick_lower = (bot - eb.low) / R

            # Ensure simplex constraint (numerical safety)
            total = eb.body_ratio + eb.wick_upper + eb.wick_lower
            if total > 0:
                eb.body_ratio /= total
                eb.wick_upper /= total
                eb.wick_lower /= total
        else:
            eb.body_ratio = 0.0
            eb.wick_upper = 0.5
            eb.wick_lower = 0.5

        eb.is_bullish = eb.close >= eb.open
        eb.body_direction = (eb.close - eb.open) / R if R > 0 else 0.0
        eb.range_pct = R / eb.open if eb.open > 0 else 0.0
        eb.entropy = self.entropy

        # OFI approximation from OHLCV
        # sign(close - open) * volume is a crude but functional proxy
        if eb.close > eb.open:
            eb.ofi = eb.volume
        elif eb.close < eb.open:
            eb.ofi = -eb.volume
        else:
            eb.ofi = 0.0

        # Normalize OFI by range for comparability
        if R > 0 and eb.open > 0:
            eb.ofi = eb.ofi * (R / eb.open)

        # Realized vol within bar (Parkinson estimator)
        if eb.high > 0 and eb.low > 0 and eb.high >= eb.low:
            eb.realized_vol = math.log(eb.high / eb.low) / (2 * math.sqrt(math.log(2)))

        # VWAP approximation
        if self._volumes and sum(self._volumes) > 0:
            total_vol = sum(self._volumes)
            # Approximate VWAP as volume-weighted mid
            vwap_sum = 0.0
            for i, v in enumerate(self._volumes):
                # Use the corresponding close prices
                idx = min(i * 4 + 3, len(self._prices) - 1)
                vwap_sum += self._prices[idx] * v
            eb.vwap = vwap_sum / total_vol if total_vol > 0 else eb.close
            eb.d_vwap = (eb.close - eb.vwap) / eb.vwap if eb.vwap > 0 else 0.0
        else:
            eb.vwap = eb.close
            eb.d_vwap = 0.0

        return eb


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _max_streak(bars: list[EventBar]) -> int:
    """Find max consecutive same-direction streak."""
    if not bars:
        return 0
    max_s = 1
    current = 1
    for i in range(1, len(bars)):
        if bars[i].is_bullish == bars[i - 1].is_bullish:
            current += 1
            max_s = max(max_s, current)
        else:
            current = 1
    return max_s


def _simplex_distance(a: tuple, b: tuple) -> float:
    """Euclidean distance on the simplex."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ═══════════════════════════════════════════════════════════════
# PARTICLE CANDLE FORECASTER MODULE
# ═══════════════════════════════════════════════════════════════

class ParticleCandleModule:
    """
    Forecaster module: event-quantized candle analysis.

    Converts standard bars to event-quantized bars, then computes
    simplex geometry, microstructure features, and envelope metrics
    that are used as additional inputs to the meta-learner and
    regime gate.

    Implements the standard predict(snapshot, horizon_hours) interface.
    """

    name = "particle_candle"
    trusted_regimes = {Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE, Regime.POST_JUMP}
    failure_modes = ["insufficient_bars", "zero_volume"]

    def __init__(self,
                 rule: str = "adaptive",
                 threshold: float = 0.0,
                 window_sizes: tuple = (20, 40)):
        self.builder = ParticleCandleBuilder(rule=rule, threshold=threshold)
        self.window_sizes = window_sizes

    def predict(self, snap: MarketSnapshot,
                horizon_hours: float = 24.0) -> ModuleOutput:
        """
        Build event bars from snapshot and produce features + prediction.
        """
        t0 = time.time()
        features = {}
        targets = PredictionTargets()

        bars = snap.bars_1h
        if len(bars) < 10:
            return ModuleOutput(
                module_name=self.name,
                targets=targets,
                confidence=0.0,
                features=features,
                metadata={"reason": "insufficient bars"},
                elapsed_ms=(time.time() - t0) * 1000,
            )

        # Build event bars
        try:
            event_bars = self.builder.from_ohlcv(bars)
        except Exception:
            event_bars = []

        if len(event_bars) < 5:
            return ModuleOutput(
                module_name=self.name,
                targets=targets,
                confidence=0.0,
                features=features,
                metadata={"reason": "too_few_event_bars"},
                elapsed_ms=(time.time() - t0) * 1000,
            )

        # Compute per-bar features
        bar_features = self.builder.compute_features(event_bars)
        features.update(bar_features)

        # Compute envelope metrics for multiple windows
        for w in self.window_sizes:
            if len(event_bars) >= w:
                seq = EventBarSequence(bars=event_bars[-w:], symbol=snap.symbol)
                env = seq.envelope_metrics()
                for k, v in env.items():
                    features[f"pc__env_{w}__{k}"] = v

        # Use the shortest available window
        if len(event_bars) >= min(self.window_sizes):
            w = min(w_ for w_ in self.window_sizes if len(event_bars) >= w_)
            seq = EventBarSequence(bars=event_bars[-w:], symbol=snap.symbol)
            env = seq.envelope_metrics()
        else:
            seq = EventBarSequence(bars=event_bars, symbol=snap.symbol)
            env = seq.envelope_metrics()

        # Direction signal from particle candle analysis
        dir_signals = []

        # Signal 1: OFI momentum
        ofi_mom = features.get("pc__ofi_momentum", 0.0)
        if abs(ofi_mom) > 0:
            dir_signals.append(("ofi", 0.50 + 0.10 * np.sign(ofi_mom), 0.3))

        # Signal 2: Wick asymmetry (upper wicks = selling pressure)
        wick_asym = features.get("pc__wick_asymmetry", 0.0)
        if abs(wick_asym) > 0.05:
            # More upper wicks → bearish
            dir_signals.append(("wick", 0.50 - 0.15 * np.sign(wick_asym), 0.2))

        # Signal 3: Body ratio trend (larger bodies = stronger trend)
        body_std = features.get("pc__std_body", 0.0)
        bullish_pct = features.get("pc__bullish_pct", 0.5)
        if body_std > 0.05:
            dir_signals.append(("body", 0.50 + 0.15 * (bullish_pct - 0.5) * 2, 0.25))

        # Signal 4: Envelope slope
        slope_mid = env.get("slope_mid", 0.0)
        price = snap.current_price
        if price > 0 and abs(slope_mid) > 0:
            norm_slope = slope_mid / price
            dir_signals.append(("slope", 0.50 + min(0.15, max(-0.15,
                norm_slope * 500)), 0.25))

        # Signal 5: Range contraction (breakout imminent?)
        contraction = env.get("contraction_rate", 0.0)
        if contraction < -0.03:
            # Range narrowing → breakout expected, amplify direction
            features["pc__breakout_signal"] = 1.0
        else:
            features["pc__breakout_signal"] = 0.0

        # Combine signals
        if dir_signals:
            total_w = sum(w for _, _, w in dir_signals)
            dir_prob = sum(p * w for _, p, w in dir_signals) / total_w if total_w > 0 else 0.5
            dir_prob = max(0.30, min(0.70, dir_prob))
        else:
            dir_prob = 0.5

        targets.direction_prob = dir_prob

        # Expected return from envelope slope
        if price > 0:
            targets.expected_return = slope_mid / price * horizon_hours if slope_mid else 0.0
            targets.expected_return = max(-0.05, min(0.05, targets.expected_return))

        # Volatility from realized vol
        mean_rvol = features.get("pc__mean_rvol", 0.0)
        targets.volatility_forecast = mean_rvol * math.sqrt(horizon_hours)

        # Confidence based on data quality
        n_event = len(event_bars)
        confidence = min(0.75, 0.20 + n_event / (n_event + 30))

        elapsed = (time.time() - t0) * 1000

        return ModuleOutput(
            module_name=self.name,
            targets=targets,
            confidence=confidence,
            features=features,
            metadata={
                "n_event_bars": n_event,
                "rule": self.builder.rule,
                "threshold": self.builder.threshold,
                "envelope_metrics": env,
            },
            elapsed_ms=elapsed,
        )
