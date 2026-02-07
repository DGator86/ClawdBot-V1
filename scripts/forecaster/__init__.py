"""
Crypto Forecasting Engine — 14 Paradigm Ensemble
=================================================
Modular ensemble forecaster implementing all major crypto prediction paradigms:

1. Technical feature generators (trend, mean-reversion, vol regime, volume)
2. Classical stats (GARCH vol, Kalman trend, regime-switching)
3. Macro/cross-asset factor residualization
4. Derivatives positioning (LFI, funding, OI, tail risk)
5. Microstructure/order flow (OFI, trade imbalance, liquidity)
6. On-chain slow priors (cycle/risk context)
7. Sentiment/attention modulators
8. Tabular ML meta-learner (walk-forward GBM)
9. Deep sequence model (quantile predictor)
10. Regime state machine + gating policy
11. Monte Carlo envelope generator (regime-conditioned)
12. Crowd/prediction-market implied priors
13. Particle candle analysis (event-quantized bars + simplex geometry)
14. Manifold pattern detection (motif clustering + classical mapping)

Usage:
    from scripts.forecaster import Forecaster
    fc = Forecaster()
    result = fc.forecast("BTCUSDT", horizon_hours=24)
"""
from .engine import Forecaster, ForecastResult
from .rl_env import (
    ForecastTradingEnv,
    evaluate_forecaster_as_trader,
    commission_sweep,
)
from .ml_models import HybridPredictor, TemporalFeatureExtractor
from .regime_gate import RegimeGate, ArbitrageDetector
from .auto_fix import AutoFixPipeline, CalibrationSuite, HealthMonitor
from .particle_candles import (
    ParticleCandleModule,
    ParticleCandleBuilder,
    EventBar,
    EventBarSequence,
)
from .manifold_patterns import (
    ManifoldPatternModule,
    ManifoldPatternDetector,
    PatternDetection,
)

__all__ = [
    "Forecaster",
    "ForecastResult",
    "ForecastTradingEnv",
    "evaluate_forecaster_as_trader",
    "commission_sweep",
    "HybridPredictor",
    "TemporalFeatureExtractor",
    "RegimeGate",
    "ArbitrageDetector",
    "AutoFixPipeline",
    "CalibrationSuite",
    "HealthMonitor",
    "ParticleCandleModule",
    "ParticleCandleBuilder",
    "EventBar",
    "EventBarSequence",
    "ManifoldPatternModule",
    "ManifoldPatternDetector",
    "PatternDetection",
]
