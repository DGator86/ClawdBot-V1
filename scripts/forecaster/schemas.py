"""
Forecaster Schemas — Module Interfaces & Data Types
=====================================================
Every module conforms to these interfaces. The ensemble orchestrator
consumes ModuleOutput from each paradigm and produces a ForecastResult.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


# ── Regime Taxonomy ─────────────────────────────────────────
class Regime(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    VOL_EXPANSION = "vol_expansion"
    POST_JUMP = "post_jump"
    CASCADE_RISK = "cascade_risk"
    EVENT_WINDOW = "event_window"
    ILLIQUID = "illiquid"


# ── Prediction Targets ──────────────────────────────────────
@dataclass
class PredictionTargets:
    """All prediction targets produced by the ensemble."""
    # A1) Next-horizon return: y_{t,h} = log(P_{t+h}/P_t)
    expected_return: float = 0.0          # log-return forecast
    return_std: float = 0.0               # uncertainty on return

    # A2) Direction
    direction_prob: float = 0.5           # P(return > 0)

    # A3) Distribution / quantiles
    quantile_10: float = 0.0             # 10th percentile of return
    quantile_50: float = 0.0             # median return
    quantile_90: float = 0.0             # 90th percentile of return

    # A4) Volatility
    volatility_forecast: float = 0.0      # forecasted realized vol
    vol_of_vol: float = 0.0               # uncertainty on vol

    # A5) Tail risk / jump probability
    jump_prob: float = 0.0                # P(|y| > 2*sigma)
    crash_prob: float = 0.0               # P(y < -3*sigma)

    # A6) Regime
    regime: Regime = Regime.RANGE
    regime_probs: dict = field(default_factory=dict)

    # A7) Barrier probability (Kalshi-specific)
    barrier_above_prob: float = 0.5       # P(price >= K)
    barrier_below_prob: float = 0.5       # P(price <= K)
    barrier_strike: float = 0.0


# ── Module Output ───────────────────────────────────────────
@dataclass
class ModuleOutput:
    """Standard output from any forecasting module."""
    module_name: str
    targets: PredictionTargets
    confidence: float = 0.5               # 0-1, how much this module trusts itself
    weight: float = 1.0                   # gating weight (set by regime machine)
    features: dict = field(default_factory=dict)  # raw feature values for meta-learner
    metadata: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0


# ── OHLCV Bar ───────────────────────────────────────────────
@dataclass
class Bar:
    """Single OHLCV bar."""
    timestamp: float   # unix epoch
    open: float
    high: float
    low: float
    close: float
    volume: float


# ── Market Data Snapshot ────────────────────────────────────
@dataclass
class MarketSnapshot:
    """All data available at forecast time."""
    symbol: str
    bars_1h: list[Bar] = field(default_factory=list)    # hourly OHLCV
    bars_4h: list[Bar] = field(default_factory=list)    # 4h aggregates
    bars_1d: list[Bar] = field(default_factory=list)    # daily aggregates

    # Derivatives
    funding_rate: Optional[float] = None
    funding_rate_history: list[float] = field(default_factory=list)
    open_interest: Optional[float] = None
    oi_history: list[float] = field(default_factory=list)
    long_short_ratio: Optional[float] = None

    # Order book / microstructure
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_depth: Optional[float] = None     # total bid volume within 0.5%
    ask_depth: Optional[float] = None
    spread: Optional[float] = None
    recent_trades: list[dict] = field(default_factory=list)

    # Cross-asset
    spx_return_1d: Optional[float] = None
    dxy_return_1d: Optional[float] = None
    gold_return_1d: Optional[float] = None
    eth_btc_ratio: Optional[float] = None

    # On-chain (slow)
    exchange_inflow_24h: Optional[float] = None
    exchange_outflow_24h: Optional[float] = None
    mvrv_ratio: Optional[float] = None

    # Sentiment
    fear_greed_index: Optional[float] = None
    social_volume: Optional[float] = None

    # Crowd / prediction market
    kalshi_barrier_probs: dict = field(default_factory=dict)

    # Meta
    timestamp: float = field(default_factory=time.time)

    @property
    def current_price(self) -> float:
        if self.bars_1h:
            return self.bars_1h[-1].close
        return 0.0

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.bars_1h]

    @property
    def highs(self) -> list[float]:
        return [b.high for b in self.bars_1h]

    @property
    def lows(self) -> list[float]:
        return [b.low for b in self.bars_1h]

    @property
    def volumes(self) -> list[float]:
        return [b.volume for b in self.bars_1h]


# ── Evaluation Metrics ──────────────────────────────────────
@dataclass
class EvalMetrics:
    """Walk-forward evaluation results."""
    # Direction
    hit_rate: float = 0.0
    mcc: float = 0.0                      # Matthews correlation
    hit_rate_by_regime: dict = field(default_factory=dict)

    # Distribution
    pinball_loss_10: float = 0.0
    pinball_loss_50: float = 0.0
    pinball_loss_90: float = 0.0
    crps: float = 0.0                     # continuous ranked probability score

    # Tail risk
    brier_score_jump: float = 0.0
    brier_score_crash: float = 0.0

    # Return magnitude
    mae: float = 0.0
    ic: float = 0.0                       # information coefficient (rank corr)

    # Calibration
    calibration_bins: dict = field(default_factory=dict)

    # Monte Carlo full-pipeline metrics
    mc_metrics: dict = field(default_factory=dict)

    # Per-regime breakdown
    metrics_by_regime: dict = field(default_factory=dict)
    metrics_by_vol_bucket: dict = field(default_factory=dict)
