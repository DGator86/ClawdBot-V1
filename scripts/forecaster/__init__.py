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
# Lazy imports to avoid the runpy RuntimeWarning when running
# ``python -m scripts.forecaster.engine``.  Eager import of .engine
# in __init__.py puts the module in sys.modules *before* runpy
# executes it as __main__, which triggers the warning.
#
# With lazy imports, symbols are only loaded when accessed, not at
# package import time.

import importlib as _importlib

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

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Forecaster":                  (".engine",            "Forecaster"),
    "ForecastResult":              (".engine",            "ForecastResult"),
    "ForecastTradingEnv":          (".rl_env",            "ForecastTradingEnv"),
    "evaluate_forecaster_as_trader": (".rl_env",          "evaluate_forecaster_as_trader"),
    "commission_sweep":            (".rl_env",            "commission_sweep"),
    "HybridPredictor":             (".ml_models",         "HybridPredictor"),
    "TemporalFeatureExtractor":    (".ml_models",         "TemporalFeatureExtractor"),
    "RegimeGate":                  (".regime_gate",       "RegimeGate"),
    "ArbitrageDetector":           (".regime_gate",       "ArbitrageDetector"),
    "AutoFixPipeline":             (".auto_fix",          "AutoFixPipeline"),
    "CalibrationSuite":            (".auto_fix",          "CalibrationSuite"),
    "HealthMonitor":               (".auto_fix",          "HealthMonitor"),
    "ParticleCandleModule":        (".particle_candles",  "ParticleCandleModule"),
    "ParticleCandleBuilder":       (".particle_candles",  "ParticleCandleBuilder"),
    "EventBar":                    (".particle_candles",  "EventBar"),
    "EventBarSequence":            (".particle_candles",  "EventBarSequence"),
    "ManifoldPatternModule":       (".manifold_patterns", "ManifoldPatternModule"),
    "ManifoldPatternDetector":     (".manifold_patterns", "ManifoldPatternDetector"),
    "PatternDetection":            (".manifold_patterns", "PatternDetection"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path, __package__)
        val = getattr(mod, attr)
        # Cache on the module so __getattr__ isn't called again
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
