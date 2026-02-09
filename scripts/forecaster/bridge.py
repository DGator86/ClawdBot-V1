"""
Forecaster-Scanner Bridge
===========================
Provides the model_prob function used by kalshi-edge-scanner.py,
replacing the simple price-distance logistic with the full
12-paradigm ensemble forecast.

Usage in edge scanner:
    from scripts.forecaster.bridge import get_ensemble_model_prob

    model_prob, model_source, forecast_meta = get_ensemble_model_prob(
        symbol="BTCUSDT",
        strike=65000.0,
        current_price=63000.0,
        horizon_hours=24.0,
    )

The bridge is designed to fail gracefully: if the ensemble is unavailable
or errors out, it returns None so the scanner can fall back to its
built-in price-distance model.
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

# Ensure the parent directory is in the path for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_scripts_dir = os.path.dirname(_this_dir)
_repo_dir = os.path.dirname(_scripts_dir)
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)


# Singleton forecaster instance (lazy-loaded)
_forecaster = None
_last_snapshot = {}        # symbol -> (MarketSnapshot, timestamp)
_snapshot_ttl = 60         # reuse snapshot for 60 seconds


def _get_forecaster():
    """Lazy-init the Forecaster singleton."""
    global _forecaster
    if _forecaster is None:
        from scripts.forecaster.engine import Forecaster
        _forecaster = Forecaster(
            mc_iterations=20_000,   # faster for scanner loop
            mc_steps=48,
            enable_mc=True,
        )
    return _forecaster


def _get_snapshot(symbol: str):
    """Get a MarketSnapshot, reusing a cached one if fresh enough."""
    now = time.time()
    if symbol in _last_snapshot:
        snap, ts = _last_snapshot[symbol]
        if now - ts < _snapshot_ttl:
            return snap

    from scripts.forecaster.data import fetch_market_snapshot
    snap = fetch_market_snapshot(symbol, bars_limit=200)
    _last_snapshot[symbol] = (snap, now)
    return snap


def get_ensemble_model_prob(
    symbol: str,
    strike: float,
    current_price: float,
    horizon_hours: float = 24.0,
) -> tuple[Optional[float], str, dict]:
    """
    Compute model probability for a Kalshi barrier contract using
    the 12-paradigm ensemble forecaster.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        strike: Barrier strike price
        current_price: Current price (for sanity check)
        horizon_hours: Hours until contract expiry

    Returns:
        (model_prob, model_source, meta_dict)
        - model_prob: P(price >= strike at expiry), or None on failure
        - model_source: "ensemble-12" or fallback identifier
        - meta_dict: additional forecast metadata for logging
    """
    try:
        fc = _get_forecaster()
        snap = _get_snapshot(symbol)

        # Sanity check: if snapshot price is wildly different from
        # current_price, the data might be stale
        if snap.current_price > 0 and current_price > 0:
            price_diff = abs(snap.current_price - current_price) / current_price
            if price_diff > 0.05:  # >5% difference
                return None, "snapshot_stale", {"price_diff": price_diff}

        result = fc.forecast_from_snapshot(
            snap, horizon_hours, barrier_strike=strike
        )

        # The barrier probability we want: P(price >= strike)
        model_prob = result.barrier_above_prob

        # Additional quality check: if MC didn't run or barrier wasn't computed,
        # use the distribution-based estimate
        if model_prob == 0.5 and result.predicted_price > 0:
            # Fall back to ensemble distribution estimate
            # Use quantile-based interpolation
            sigma = result.volatility * math.sqrt(horizon_hours / 24)
            mu = result.targets.expected_return
            if sigma > 0:
                z = (math.log(strike / current_price) - mu) / sigma
                model_prob = 1.0 - (1.0 / (1.0 + math.exp(-1.7 * z)))
            else:
                model_prob = 0.5

        meta = {
            "predicted_price": result.predicted_price,
            "direction": result.direction,
            "confidence": result.confidence,
            "regime": result.regime,
            "volatility": result.volatility,
            "jump_prob": result.jump_prob,
            "crash_prob": result.crash_prob,
            "modules_run": result.modules_run,
            "elapsed_ms": result.elapsed_ms,
            "var_95": result.var_95,
        }

        return model_prob, "ensemble-12", meta

    except Exception as e:
        return None, f"ensemble_error: {e}", {}


def get_ensemble_forecast(
    symbol: str = "BTCUSDT",
    horizon_hours: float = 24.0,
    barrier_strike: Optional[float] = None,
) -> Optional[dict]:
    """
    Run a full ensemble forecast and return a dict compatible with
    the Monte Carlo simulation.py PREDICTION format.

    Returns None on failure.
    """
    try:
        fc = _get_forecaster()
        snap = _get_snapshot(symbol)
        result = fc.forecast_from_snapshot(
            snap, horizon_hours, barrier_strike=barrier_strike
        )
        return result.to_prediction_dict()
    except Exception as e:
        print(f"Ensemble forecast error: {e}")
        return None
