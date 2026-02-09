"""
Forecaster Engine -- Ensemble Orchestrator (Ultimate Enhanced)
Wires all 12 modules through the regime gate, produces a unified
Forecaster Engine -- Ensemble Orchestrator (Ultimate Enhanced + Particle)
Wires all 14 modules through the regime gate, produces a unified
ForecastResult, and exposes a simple `forecast()` API.

Ultimate-fix enhancements:
  - Hybrid ML: LightGBM + temporal features (replaces plain GBM)
  - Regime gating: blocks anti-predictive regimes (range HR=41.8%)
  - Auto-calibration: isotonic + Platt scaling on direction probs
  - Health monitoring: rolling HR tracking with auto-retrain
  - Arbitrage detection: spread + model-edge opportunities

Architecture:
  MarketSnapshot -> [12 Modules] -> RegimeDetector -> GatingPolicy
                                  -> HybridPredictor -> RegimeGate
                                  -> MonteCarloModule -> AutoFix
                                  -> ForecastResult
Particle candle enhancements:
  - Event-quantized bars: aggregate by volume/trades/entropy (not clock time)
  - Simplex geometry: B + W_u + W_l = 1 on the candle manifold
  - Manifold patterns: GMM clustering + classical pattern template matching
  - Forward distributions: conditional P(Δp | pattern, regime) per cluster

Architecture:
  MarketSnapshot -> [9 Base Modules] -> RegimeDetector -> GatingPolicy
                 -> ParticleCandleModule -> ManifoldPatternModule
                 -> HybridPredictor -> RegimeGate
                 -> MonteCarloModule -> AutoFix
                 -> ForecastResult

Usage:
    from scripts.forecaster.engine import Forecaster
    fc = Forecaster()
    result = fc.forecast("BTCUSDT", horizon_hours=24)
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .schemas import (
    Bar, MarketSnapshot, ModuleOutput, PredictionTargets,
    Regime, EvalMetrics,
)
from .modules import (
    TechnicalModule,
    ClassicalStatsModule,
    MacroFactorModule,
    DerivativesModule,
    MicrostructureModule,
    OnChainModule,
    SentimentModule,
    MetaLearnerModule,
    SequenceModule,
    RegimeDetector,
    MonteCarloModule,
    CrowdPriorModule,
)

# Ultimate-fix imports
from .ml_models import HybridPredictor, TemporalFeatureExtractor
from .regime_gate import RegimeGate, ArbitrageDetector, should_trade, GateDecision
from .auto_fix import AutoFixPipeline, CalibrationSuite

# Particle candle + manifold pattern imports
from .particle_candles import ParticleCandleModule, ParticleCandleBuilder, EventBar, EventBarSequence
from .manifold_patterns import ManifoldPatternModule, ManifoldPatternDetector, PatternDetection


# ═══════════════════════════════════════════════════════════════
# FORECAST RESULT
# ═══════════════════════════════════════════════════════════════

@dataclass
class ForecastResult:
    """Complete output from the ensemble forecaster."""

    # Identity
    symbol: str = ""
    timestamp: str = ""
    horizon_hours: float = 24.0
    current_price: float = 0.0

    # Ensemble targets (weighted average from all modules)
    targets: PredictionTargets = field(default_factory=PredictionTargets)

    # Regime
    regime: str = "range"
    regime_probs: dict = field(default_factory=dict)
    confidence_scalar: float = 1.0

    # Module-level detail
    module_outputs: dict = field(default_factory=dict)   # name -> summary
    gating_weights: dict = field(default_factory=dict)   # name -> weight

    # Monte Carlo results
    mc_summary: dict = field(default_factory=dict)

    # Price-level outputs (for edge scanner / Kalshi)
    predicted_price: float = 0.0
    direction: str = "flat"
    confidence: float = 0.5
    volatility: float = 0.0

    # Quantile prices
    price_q05: float = 0.0
    price_q10: float = 0.0
    price_q25: float = 0.0
    price_q50: float = 0.0
    price_q75: float = 0.0
    price_q90: float = 0.0
    price_q95: float = 0.0

    # Risk
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    jump_prob: float = 0.0
    crash_prob: float = 0.0

    # Barrier (Kalshi)
    barrier_above_prob: float = 0.5
    barrier_below_prob: float = 0.5
    barrier_strike: float = 0.0

    # Performance
    elapsed_ms: float = 0.0
    modules_run: int = 0

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, PredictionTargets):
                d[k] = {
                    "expected_return": v.expected_return,
                    "return_std": v.return_std,
                    "direction_prob": v.direction_prob,
                    "quantile_10": v.quantile_10,
                    "quantile_50": v.quantile_50,
                    "quantile_90": v.quantile_90,
                    "volatility_forecast": v.volatility_forecast,
                    "vol_of_vol": v.vol_of_vol,
                    "jump_prob": v.jump_prob,
                    "crash_prob": v.crash_prob,
                    "regime": v.regime.value if isinstance(v.regime, Regime) else str(v.regime),
                    "regime_probs": {
                        (rk.value if isinstance(rk, Regime) else str(rk)): rv
                        for rk, rv in v.regime_probs.items()
                    },
                    "barrier_above_prob": v.barrier_above_prob,
                    "barrier_below_prob": v.barrier_below_prob,
                    "barrier_strike": v.barrier_strike,
                }
            elif isinstance(v, dict):
                # Convert any Regime keys to strings
                clean = {}
                for dk, dv in v.items():
                    key = dk.value if isinstance(dk, Regime) else str(dk)
                    clean[key] = dv
                d[k] = clean
            else:
                d[k] = v
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_prediction_dict(self) -> dict:
        """
        Return a dict compatible with the existing Monte Carlo simulation.py
        PREDICTION format, so the MC engine can be driven from ensemble output.
        """
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "current_price": self.current_price,
            "predicted_price": self.predicted_price,
            "direction": self.direction,
            "confidence": self.confidence,
            "volatility": self.volatility,
            "quantiles": {
                "q05": self.price_q05,
                "q50": self.price_q50,
                "q95": self.price_q95,
            },
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        arrow = "\u2191" if self.direction == "Up" else "\u2193" if self.direction == "Down" else "\u2194"
        return (
            f"{self.symbol} {arrow} ${self.predicted_price:,.2f} "
            f"({self.confidence*100:.1f}% conf) "
            f"regime={self.regime} vol={self.volatility:.4f} "
            f"jump={self.jump_prob:.2f} crash={self.crash_prob:.2f}"
        )


# ═══════════════════════════════════════════════════════════════
# FORECASTER ENGINE
# ═══════════════════════════════════════════════════════════════

class Forecaster:
    """
    14-paradigm ensemble forecaster with regime gating.

    Paradigms 1-12: original ensemble (technical, classical, macro, etc.)
    Paradigm 13: Particle candle analysis (event-quantized bars + simplex geometry)
    Paradigm 14: Manifold pattern detection (motif clustering + classical mapping)

    Usage:
        fc = Forecaster()
        # With pre-built snapshot:
        result = fc.forecast_from_snapshot(snapshot, horizon_hours=24)

        # With auto-fetch (requires data module):
        result = fc.forecast("BTCUSDT", horizon_hours=24)
    """

    def __init__(self,
                 mc_iterations: int = 50_000,
                 mc_steps: int = 48,
                 mc_seed: int = 42,
                 enable_mc: bool = True,
                 enable_regime_gate: bool = True,
                 enable_hybrid_ml: bool = True,
                 enable_auto_fix: bool = True):
                 enable_auto_fix: bool = True,
                 enable_particle_candles: bool = True,
                 enable_manifold_patterns: bool = True):
        # ── Instantiate all modules ───────────────────────
        self.technical = TechnicalModule()
        self.classical = ClassicalStatsModule()
        self.macro = MacroFactorModule()
        self.derivatives = DerivativesModule()
        self.microstructure = MicrostructureModule()
        self.onchain = OnChainModule()
        self.sentiment = SentimentModule()
        self.sequence = SequenceModule()
        self.crowd = CrowdPriorModule()

        # Meta-modules
        self.regime_detector = RegimeDetector()
        self.meta_learner = MetaLearnerModule()
        self.monte_carlo = MonteCarloModule()

        # ── Ultimate-fix: enhanced modules ─────────────────
        self.hybrid_predictor = HybridPredictor()
        self.regime_gate = RegimeGate()
        self.auto_fix = AutoFixPipeline()
        self.arb_detector = ArbitrageDetector()
        self._temporal_extractor = TemporalFeatureExtractor(lookback=48)

        # ── Particle candle + manifold pattern modules ────
        self.particle_candle = ParticleCandleModule(
            rule="adaptive", window_sizes=(20, 40),
        )
        self.manifold_pattern = ManifoldPatternModule(
            window=30, candle_rule="adaptive",
        )

        # Config
        self.mc_iterations = mc_iterations
        self.mc_steps = mc_steps
        self.mc_seed = mc_seed
        self.enable_mc = enable_mc
        self.enable_regime_gate = enable_regime_gate
        self.enable_hybrid_ml = enable_hybrid_ml
        self.enable_auto_fix = enable_auto_fix
        self.enable_particle_candles = enable_particle_candles
        self.enable_manifold_patterns = enable_manifold_patterns

        # All predictive modules (order matters for feature flow)
        self._modules = [
            self.technical,
            self.classical,
            self.macro,
            self.derivatives,
            self.microstructure,
            self.onchain,
            self.sentiment,
            self.sequence,
            self.crowd,
        ]

    def forecast_from_snapshot(self,
                               snap: MarketSnapshot,
                               horizon_hours: float = 24.0,
                               barrier_strike: Optional[float] = None,
                               ) -> ForecastResult:
        """
        Run the full ensemble pipeline on a MarketSnapshot.

        Pipeline:
        1. Run all 9 base modules
        2. Regime detection + gating weights
        3. Apply weights to module outputs
        4. Meta-learner combines weighted outputs
        5. Monte Carlo simulation with regime-conditioned params
        6. Package into ForecastResult
        """
        t0 = time.time()
        result = ForecastResult(
            symbol=snap.symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            horizon_hours=horizon_hours,
            current_price=snap.current_price,
        )

        # ── Step 1: Run all base modules ──────────────────
        module_outputs: list[ModuleOutput] = []
        for mod in self._modules:
            try:
                out = mod.predict(snap, horizon_hours)
                module_outputs.append(out)
                result.module_outputs[out.module_name] = {
                    "confidence": round(out.confidence, 4),
                    "direction_prob": round(out.targets.direction_prob, 4),
                    "expected_return": round(out.targets.expected_return, 6),
                    "volatility": round(out.targets.volatility_forecast, 6),
                    "jump_prob": round(out.targets.jump_prob, 4),
                    "elapsed_ms": round(out.elapsed_ms, 2),
                }
            except Exception as e:
                result.module_outputs[mod.name] = {
                    "error": str(e), "confidence": 0.0,
                }

        # ── Step 2: Regime detection ──────────────────────
        regime_probs, confidence_scalar = self.regime_detector.detect(
            snap, module_outputs
        )
        result.regime_probs = {
            r.value: round(p, 4)
            for r, p in regime_probs.items()
        }
        result.confidence_scalar = round(confidence_scalar, 4)

        # Dominant regime
        dominant = max(regime_probs, key=regime_probs.get)
        result.regime = dominant.value

        # ── Step 3: Compute and apply gating weights ──────
        gating_weights = self.regime_detector.compute_weights(regime_probs)

        # Zero-out weights for modules with no actual data input.
        # If a module reports very low confidence (≤ 0.15) it means
        # it had no real data to work with (e.g. derivatives: 0/2,
        # macro: 0 items, onchain: empty).  Giving these weight just
        # pollutes the ensemble with priors that look like signal.
        _NO_DATA_THRESHOLD = 0.15
        for out in module_outputs:
            if out.confidence <= _NO_DATA_THRESHOLD:
                gating_weights[out.module_name] = 0.0

        result.gating_weights = {
            k: round(v, 4) for k, v in gating_weights.items()
        }

        for out in module_outputs:
            w = gating_weights.get(out.module_name, 0.5)
            out.weight = w

        # ── Step 4: Meta-learner ──────────────────────────
        try:
            meta_out = self.meta_learner.predict_from_modules(
                module_outputs, horizon_hours
            )
            module_outputs.append(meta_out)
            result.module_outputs[meta_out.module_name] = {
                "confidence": round(meta_out.confidence, 4),
                "direction_prob": round(meta_out.targets.direction_prob, 4),
                "expected_return": round(meta_out.targets.expected_return, 6),
                "volatility": round(meta_out.targets.volatility_forecast, 6),
                "jump_prob": round(meta_out.targets.jump_prob, 4),
                "elapsed_ms": round(meta_out.elapsed_ms, 2),
            }
            ensemble_targets = meta_out.targets
        except Exception as e:
            result.module_outputs["meta_learner"] = {"error": str(e)}
            # Fallback: simple average of module outputs
            ensemble_targets = self._simple_average(module_outputs)

        # Assign regime info to ensemble targets
        ensemble_targets.regime = dominant
        ensemble_targets.regime_probs = regime_probs

        # ── Step 4a2: Particle candle module ────────────────
        if self.enable_particle_candles and len(snap.bars_1h) >= 20:
            try:
                pc_out = self.particle_candle.predict(snap, horizon_hours)
                if pc_out.confidence > 0:
                    module_outputs.append(pc_out)
                    result.module_outputs[pc_out.module_name] = {
                        "confidence": round(pc_out.confidence, 4),
                        "direction_prob": round(pc_out.targets.direction_prob, 4),
                        "expected_return": round(pc_out.targets.expected_return, 6),
                        "volatility": round(pc_out.targets.volatility_forecast, 6),
                        "n_event_bars": pc_out.metadata.get("n_event_bars", 0),
                        "elapsed_ms": round(pc_out.elapsed_ms, 2),
                    }
            except Exception as e:
                result.module_outputs["particle_candle"] = {"error": str(e)}

        # ── Step 4a3: Manifold pattern module ──────────────
        if self.enable_manifold_patterns and len(snap.bars_1h) >= 20:
            try:
                mp_out = self.manifold_pattern.predict(snap, horizon_hours)
                if mp_out.confidence > 0:
                    module_outputs.append(mp_out)
                    result.module_outputs[mp_out.module_name] = {
                        "confidence": round(mp_out.confidence, 4),
                        "direction_prob": round(mp_out.targets.direction_prob, 4),
                        "expected_return": round(mp_out.targets.expected_return, 6),
                        "pattern": mp_out.metadata.get("pattern", "none"),
                        "breakout_bias": mp_out.metadata.get("breakout_bias", "neutral"),
                        "match_score": mp_out.metadata.get("match_score", 0),
                        "elapsed_ms": round(mp_out.elapsed_ms, 2),
                    }
            except Exception as e:
                result.module_outputs["manifold_pattern"] = {"error": str(e)}

        # ── Step 4b: Hybrid ML enhancement ────────────────
        if self.enable_hybrid_ml and snap.closes:
            try:
                temporal_feats = self._temporal_extractor.extract(
                    snap.closes, snap.volumes, snap.highs, snap.lows
                )
                all_features = MetaLearnerModule._extract_features(module_outputs)
                for k, v in temporal_feats.items():
                    all_features[f"temporal__{k}"] = v

                self.hybrid_predictor.maybe_retrain()
                hybrid_pred = self.hybrid_predictor.predict(all_features)
                if hybrid_pred is not None:
                    h_w = 0.40
                    m_w = 1.0 - h_w
                    ensemble_targets.direction_prob = (
                        h_w * hybrid_pred["direction_prob"]
                        + m_w * ensemble_targets.direction_prob
                    )
                    ensemble_targets.expected_return = (
                        h_w * hybrid_pred["expected_return"]
                        + m_w * ensemble_targets.expected_return
                    )
                    result.module_outputs["hybrid_ml"] = {
                        "confidence": round(hybrid_pred["confidence"], 4),
                        "direction_prob": round(hybrid_pred["direction_prob"], 4),
                        "n_train_samples": hybrid_pred["n_train_samples"],
                    }
            except Exception as e:
                result.module_outputs["hybrid_ml"] = {"error": str(e)}

        # ── Step 4c: Regime gating ─────────────────────────
        gate_decision = None
        if self.enable_regime_gate:
            try:
                gate_decision = self.regime_gate.apply(
                    ensemble_targets, regime_probs
                )
                result.module_outputs["regime_gate"] = {
                    "action": gate_decision.action,
                    "tier": gate_decision.tier,
                    "original_dir_prob": round(gate_decision.original_direction_prob, 4),
                    "gated_dir_prob": round(gate_decision.gated_direction_prob, 4),
                    "multiplier": round(gate_decision.confidence_multiplier, 4),
                    "ev_edge": round(gate_decision.ev_edge, 4),
                    "min_ev_required": round(gate_decision.min_ev_required, 4),
                }
            except Exception as e:
                result.module_outputs["regime_gate"] = {"error": str(e)}

        # ── Step 4d: Auto-calibration ──────────────────────
        if self.enable_auto_fix:
            try:
                cal_prob = self.auto_fix.calibrate_prob(
                    ensemble_targets.direction_prob
                )
                if cal_prob != ensemble_targets.direction_prob:
                    ensemble_targets.direction_prob = cal_prob
            except Exception:
                pass

        result.targets = ensemble_targets

        # ── Step 5: Monte Carlo simulation ────────────────
        mc_out = None
        if self.enable_mc and snap.current_price > 0:
            try:
                mc_out = self.monte_carlo.simulate(
                    snap=snap,
                    ensemble_targets=ensemble_targets,
                    regime_probs=regime_probs,
                    confidence_scalar=confidence_scalar,
                    n_iterations=self.mc_iterations,
                    n_steps=self.mc_steps,
                    barrier_strike=barrier_strike,
                    seed=self.mc_seed,
                )
                result.mc_summary = {
                    k: v for k, v in mc_out.features.items()
                    if k != "envelope"  # too large for summary
                }
                # Use MC results for risk metrics
                result.var_95 = mc_out.features.get("mc_var_95", 0)
                result.var_99 = mc_out.features.get("mc_var_99", 0)
                result.cvar_95 = mc_out.features.get("mc_cvar_95", 0)
                result.cvar_99 = mc_out.features.get("mc_cvar_99", 0)

                result.module_outputs["monte_carlo"] = {
                    "confidence": 0.8,
                    "mc_mean_price": mc_out.features.get("mc_mean_price", 0),
                    "mc_median_price": mc_out.features.get("mc_median_price", 0),
                    "elapsed_ms": round(mc_out.elapsed_ms, 2),
                }
            except Exception as e:
                result.module_outputs["monte_carlo"] = {"error": str(e)}

        # ── Step 6: Package price-level outputs ───────────
        price = snap.current_price
        if price > 0:
            # Predicted price from expected return
            result.predicted_price = round(
                price * math.exp(ensemble_targets.expected_return), 2
            )

            # Direction
            if ensemble_targets.direction_prob > 0.55:
                result.direction = "Up"
            elif ensemble_targets.direction_prob < 0.45:
                result.direction = "Down"
            else:
                result.direction = "Flat"

            result.confidence = round(ensemble_targets.direction_prob, 4)
            if result.direction == "Down":
                result.confidence = round(1.0 - ensemble_targets.direction_prob, 4)

            result.volatility = round(ensemble_targets.volatility_forecast, 6)
            result.jump_prob = round(ensemble_targets.jump_prob, 4)
            result.crash_prob = round(ensemble_targets.crash_prob, 4)

            # Quantile prices
            sigma = ensemble_targets.return_std * confidence_scalar
            mu = ensemble_targets.expected_return
            if mc_out and "mc_p5_price" in mc_out.features:
                # Use MC-derived quantiles (more accurate with jumps)
                result.price_q05 = round(mc_out.features["mc_p5_price"], 2)
                result.price_q10 = round(mc_out.features.get("mc_p10_price",
                    mc_out.features["mc_p5_price"] * 0.5 + mc_out.features["mc_p25_price"] * 0.5), 2)
                result.price_q25 = round(mc_out.features["mc_p25_price"], 2)
                result.price_q50 = round(mc_out.features["mc_median_price"], 2)
                result.price_q75 = round(mc_out.features["mc_p75_price"], 2)
                result.price_q90 = round(mc_out.features.get("mc_p90_price",
                    mc_out.features["mc_p75_price"] * 0.5 + mc_out.features["mc_p95_price"] * 0.5), 2)
                result.price_q95 = round(mc_out.features["mc_p95_price"], 2)
            else:
                # Gaussian approximation
                result.price_q05 = round(price * math.exp(mu - 1.645 * sigma), 2)
                result.price_q10 = round(price * math.exp(mu - 1.28 * sigma), 2)
                result.price_q25 = round(price * math.exp(mu - 0.674 * sigma), 2)
                result.price_q50 = round(price * math.exp(mu), 2)
                result.price_q75 = round(price * math.exp(mu + 0.674 * sigma), 2)
                result.price_q90 = round(price * math.exp(mu + 1.28 * sigma), 2)
                result.price_q95 = round(price * math.exp(mu + 1.645 * sigma), 2)

            # Enforce monotonic quantile ordering (Q05 <= Q10 <= ... <= Q95)
            qs = [result.price_q05, result.price_q10, result.price_q25,
                  result.price_q50, result.price_q75, result.price_q90, result.price_q95]
            for i in range(1, len(qs)):
                if qs[i] < qs[i - 1]:
                    qs[i] = qs[i - 1]
            (result.price_q05, result.price_q10, result.price_q25,
             result.price_q50, result.price_q75, result.price_q90, result.price_q95) = qs

            # Barrier probs (from MC or ensemble)
            if mc_out and mc_out.targets.barrier_strike > 0:
                result.barrier_above_prob = round(mc_out.targets.barrier_above_prob, 4)
                result.barrier_below_prob = round(mc_out.targets.barrier_below_prob, 4)
                result.barrier_strike = mc_out.targets.barrier_strike
            elif ensemble_targets.barrier_strike > 0:
                result.barrier_above_prob = round(ensemble_targets.barrier_above_prob, 4)
                result.barrier_below_prob = round(ensemble_targets.barrier_below_prob, 4)
                result.barrier_strike = ensemble_targets.barrier_strike

        result.elapsed_ms = round((time.time() - t0) * 1000, 2)
        result.modules_run = len([
            o for o in result.module_outputs.values()
            if "error" not in o
        ])

        return result

    def forecast(self,
                 symbol: str = "BTCUSDT",
                 horizon_hours: float = 24.0,
                 barrier_strike: Optional[float] = None,
                 bars_limit: int = 2000,
                 ) -> ForecastResult:
        """
        Auto-fetch market data and run the ensemble.

        bars_limit: Number of 1h bars to fetch.  Defaults to 2000
        (~83 days) to match diagnostic back-test conditions.  Using
        fewer bars (e.g. 200) gives the engine a different data
        distribution than the one diagnostics validated against.
        """
        from .data import fetch_market_snapshot
        snap = fetch_market_snapshot(symbol, bars_limit=bars_limit)
        return self.forecast_from_snapshot(
            snap, horizon_hours, barrier_strike
        )

    def _simple_average(self,
                        outputs: list[ModuleOutput]) -> PredictionTargets:
        """Fallback: equal-weight average of all module outputs."""
        targets = PredictionTargets()
        n = len([o for o in outputs if o.confidence > 0])
        if n == 0:
            return targets

        for o in outputs:
            if o.confidence <= 0:
                continue
            targets.expected_return += o.targets.expected_return / n
            targets.return_std += o.targets.return_std / n
            targets.direction_prob += o.targets.direction_prob / n
            targets.volatility_forecast += o.targets.volatility_forecast / n
            targets.jump_prob += o.targets.jump_prob / n
            targets.crash_prob += o.targets.crash_prob / n
            targets.quantile_10 += o.targets.quantile_10 / n
            targets.quantile_50 += o.targets.quantile_50 / n
            targets.quantile_90 += o.targets.quantile_90 / n

        return targets

    def forecast_barrier(self,
                          symbol: str,
                          strike: float,
                          horizon_hours: float = 24.0,
                          ) -> dict:
        """
        Convenience: compute barrier probability for a Kalshi contract.
        Returns dict with {above_prob, below_prob, model_prob, edge_vs_market}.
        """
        result = self.forecast(symbol, horizon_hours, barrier_strike=strike)
        return {
            "symbol": symbol,
            "strike": strike,
            "horizon_hours": horizon_hours,
            "above_prob": result.barrier_above_prob,
            "below_prob": result.barrier_below_prob,
            "direction": result.direction,
            "confidence": result.confidence,
            "regime": result.regime,
            "current_price": result.current_price,
            "predicted_price": result.predicted_price,
            "volatility": result.volatility,
        }


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Crypto Forecaster -- 14-Paradigm Ensemble"
    )
    parser.add_argument("--symbol", "-s", default="BTCUSDT",
                        help="Symbol to forecast (default: BTCUSDT)")
    parser.add_argument("--horizon", "-H", type=float, default=24.0,
                        help="Forecast horizon in hours (default: 24)")
    parser.add_argument("--barrier", "-b", type=float, default=None,
                        help="Barrier strike for Kalshi probability")
    parser.add_argument("--mc-iterations", "-n", type=int, default=50_000,
                        help="Monte Carlo iterations (default: 50000)")
    parser.add_argument("--mc-steps", type=int, default=48,
                        help="Monte Carlo steps (default: 48)")
    parser.add_argument("--bars", type=int, default=2000,
                        help="Number of 1h bars to fetch (default: 2000)")
    parser.add_argument("--no-mc", action="store_true",
                        help="Disable Monte Carlo simulation")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path")
    args = parser.parse_args()

    fc = Forecaster(
        mc_iterations=args.mc_iterations,
        mc_steps=args.mc_steps,
        enable_mc=not args.no_mc,
    )

    print(f"Running 14-paradigm ensemble forecast for {args.symbol} "
          f"(horizon={args.horizon}h)...")
    print(f"Monte Carlo: {'ON' if not args.no_mc else 'OFF'} "
          f"({args.mc_iterations:,} iterations)")

    result = fc.forecast(
        symbol=args.symbol,
        horizon_hours=args.horizon,
        barrier_strike=args.barrier,
        bars_limit=args.bars,
    )

    if args.json:
        output = result.to_json()
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Results written to {args.output}")
        else:
            print(output)
    else:
        _print_report(result)
        if args.output:
            with open(args.output, "w") as f:
                f.write(result.to_json())
            print(f"\nJSON results saved to: {args.output}")


def _print_report(r: ForecastResult):
    """Pretty-print forecast result."""
    arrow = "\u2191" if r.direction == "Up" else "\u2193" if r.direction == "Down" else "\u2194"

    print(f"\n{'='*64}")
    print(f"  14-PARADIGM ENSEMBLE FORECAST -- {r.symbol}")
    print(f"{'='*64}")
    print(f"  Timestamp:        {r.timestamp}")
    print(f"  Horizon:          {r.horizon_hours}h")
    print(f"  Modules Run:      {r.modules_run}/14")
    print(f"  Elapsed:          {r.elapsed_ms:.1f}ms")
    print(f"{'─'*64}")
    print(f"  Current Price:    ${r.current_price:>12,.2f}")
    print(f"  Predicted Price:  ${r.predicted_price:>12,.2f}  {arrow}")
    print(f"  Direction:        {r.direction:>12} ({r.confidence*100:.1f}%)")
    print(f"  Volatility:       {r.volatility:>12.4f}")
    print(f"{'─'*64}")
    print(f"  REGIME:           {r.regime}")
    print(f"  Confidence Scale: {r.confidence_scalar:.2f}x")
    for regime_name, prob in sorted(r.regime_probs.items(),
                                     key=lambda x: -x[1]):
        if prob > 0.01:
            bar = "\u2588" * int(prob * 30)
            print(f"    {regime_name:20s} {prob*100:5.1f}% {bar}")
    print(f"{'─'*64}")
    print(f"  PRICE DISTRIBUTION:")
    print(f"    Q05:  ${r.price_q05:>12,.2f}")
    print(f"    Q10:  ${r.price_q10:>12,.2f}")
    print(f"    Q25:  ${r.price_q25:>12,.2f}")
    print(f"    Q50:  ${r.price_q50:>12,.2f}  (median)")
    print(f"    Q75:  ${r.price_q75:>12,.2f}")
    print(f"    Q90:  ${r.price_q90:>12,.2f}")
    print(f"    Q95:  ${r.price_q95:>12,.2f}")
    print(f"{'─'*64}")
    print(f"  RISK METRICS:")
    print(f"    VaR (95%):      {r.var_95*100:>8.3f}%")
    print(f"    VaR (99%):      {r.var_99*100:>8.3f}%")
    print(f"    CVaR (95%):     {r.cvar_95*100:>8.3f}%")
    print(f"    CVaR (99%):     {r.cvar_99*100:>8.3f}%")
    print(f"    Jump Prob:      {r.jump_prob*100:>8.2f}%")
    print(f"    Crash Prob:     {r.crash_prob*100:>8.2f}%")
    if r.barrier_strike > 0:
        print(f"{'─'*64}")
        print(f"  BARRIER @ ${r.barrier_strike:,.2f}:")
        print(f"    P(above):       {r.barrier_above_prob*100:>8.2f}%")
        print(f"    P(below):       {r.barrier_below_prob*100:>8.2f}%")
    print(f"{'─'*64}")
    print(f"  MODULE DETAILS:")
    for name, info in r.module_outputs.items():
        if "error" in info:
            print(f"    {name:20s} ERROR: {info['error']}")
        elif name == "regime_gate":
            # Gate shows action/tier, not standard module format
            action = info.get("action", "?")
            tier = info.get("tier", "?")
            orig = info.get("original_dir_prob", 0.5)
            gated = info.get("gated_dir_prob", 0.5)
            mult = info.get("multiplier", 1.0)
            ev = info.get("ev_edge", 0.0)
            min_ev = info.get("min_ev_required", 0.04)
            ev_ok = "✓" if ev >= min_ev else "✗"
            print(f"    {name:20s} {action:>8s} tier={tier} "
                  f"dir: {orig:.3f}→{gated:.3f} mult={mult:.2f} "
                  f"EV={ev:.3f}{ev_ok}(min={min_ev:.3f})")
        elif name in ("particle_candle", "manifold_pattern"):
            # Show pattern-specific info
            conf = info.get("confidence", 0)
            dp = info.get("direction_prob", 0.5)
            d_arrow = "\u2191" if dp > 0.55 else "\u2193" if dp < 0.45 else "\u2194"
            extra = ""
            if name == "particle_candle":
                extra = f" bars={info.get('n_event_bars', '?')}"
            elif name == "manifold_pattern":
                extra = f" {info.get('pattern', '?')}({info.get('breakout_bias', '?')})"
            print(f"    {name:20s} conf={conf:.2f} dir={dp:.3f}{d_arrow}{extra} "
                  f"t={info.get('elapsed_ms', 0):.0f}ms")
        else:
            conf = info.get("confidence", 0)
            dp = info.get("direction_prob", 0.5)
            d_arrow = "\u2191" if dp > 0.55 else "\u2193" if dp < 0.45 else "\u2194"
            print(f"    {name:20s} conf={conf:.2f} dir={dp:.3f}{d_arrow} "
                  f"vol={info.get('volatility', 0):.4f} "
                  f"t={info.get('elapsed_ms', 0):.0f}ms")
    print(f"{'─'*64}")
    print(f"  GATING WEIGHTS:")
    for name, w in sorted(r.gating_weights.items(), key=lambda x: -x[1]):
        bar = "\u2588" * int(w * 30)
        print(f"    {name:20s} {w:.3f} {bar}")
    print(f"{'='*64}")
    print(f"\n  {r.summary()}")


if __name__ == "__main__":
    main()
