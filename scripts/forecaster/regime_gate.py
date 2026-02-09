"""
Regime Gating & Arbitrage Detection
=======================================
Addresses the core problem: HR spikes mid-sample (60-80%) but crashes
elsewhere (20-40%). Model is anti-predictive in trend_up regime.

Solution:
1. Regime Gate: Only emit confident forecasts in regimes with proven edge
2. Confidence Dampening: Shrink confidence in weak regimes toward 0.50
3. Arbitrage Scanner: Detect YES/NO spread inefficiencies (inspired by
   X bots achieving 95%+ win rates on Kalshi/Polymarket lags)

Regime performance from diagnostics (2000-bar, 75-forecast backtest):
  - trend_up:    HR=66.7% (n=3)  — too few samples, unreliable
  - range:       HR=41.8% (n=55) — anti-predictive, GATE THIS
  - trend_down:  HR=~50%  (n=~10) — no edge
  - post_jump:   HR=60-80% (mid-sample) — real edge, BOOST THIS
  - trend_up:    HR=0%    (n=5)  — fully anti-predictive, BLOCK + INVERT
  - range:       HR=46.2% (n=52) — anti-predictive, BLOCK
  - trend_down:  HR=58.3% (n=12) — only regime with edge
  - post_jump:   HR=50%   (n=6)  — coin flip, no edge

Usage:
    from scripts.forecaster.regime_gate import (
        RegimeGate,
        ArbitrageDetector,
        should_trade,
    )
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .schemas import Regime, PredictionTargets


# ═══════════════════════════════════════════════════════════════
# REGIME PERFORMANCE PRIORS
# ═══════════════════════════════════════════════════════════════

# Calibrated from backtest diagnostics.
# Regimes are classified into three tiers:
#   STRONG:   HR > 55%, n > 10 — trade with full confidence
#   WEAK:     40% < HR < 55%, or n < 10 — trade with dampened confidence
#   BLOCKED:  HR < 40%, n > 10 — do not trade (anti-predictive)

@dataclass
class RegimeProfile:
    """Historical performance profile for a regime."""
    name: str
    tier: str = "weak"          # "strong", "weak", "blocked"
    observed_hr: float = 0.50   # historical hit rate
    n_samples: int = 0          # how many samples in backtest
    confidence_multiplier: float = 1.0  # scale for forecast confidence
    min_confidence_to_trade: float = 0.52  # minimum direction prob
    notes: str = ""


# Default profiles (updated as backtests accumulate)
DEFAULT_REGIME_PROFILES = {
    Regime.TREND_UP: RegimeProfile(
        name="trend_up",
        tier="weak",             # Too few samples (n=3) to be confident
        observed_hr=0.667,
        n_samples=3,
        confidence_multiplier=0.85,
        min_confidence_to_trade=0.55,
        notes="Potentially strong but n=3 is unreliable",
    ),
    Regime.TREND_DOWN: RegimeProfile(
        name="trend_down",
        tier="weak",
        observed_hr=0.50,
        n_samples=10,
        confidence_multiplier=0.75,
        min_confidence_to_trade=0.55,
        notes="No clear edge in backtest",
    ),
    Regime.RANGE: RegimeProfile(
        name="range",
        tier="blocked",          # HR=41.8% is anti-predictive
        observed_hr=0.418,
        n_samples=55,
        tier="blocked",          # HR=0% (n=5) — fully anti-predictive
        observed_hr=0.00,
        n_samples=5,
        confidence_multiplier=0.30,  # heavy dampening
        min_confidence_to_trade=0.65,
        notes="Anti-predictive (0/5). Signal is INVERTED in this regime.",
    ),
    Regime.TREND_DOWN: RegimeProfile(
        name="trend_down",
        tier="strong",           # HR=58.3% (n=12) — only edge regime
        observed_hr=0.583,
        n_samples=12,
        confidence_multiplier=1.05,  # slight boost
        min_confidence_to_trade=0.52,
        notes="Best regime. Only regime with statistical edge.",
    ),
    Regime.RANGE: RegimeProfile(
        name="range",
        tier="blocked",          # HR=46.2% (n=52) — anti-predictive
        observed_hr=0.462,
        n_samples=52,
        confidence_multiplier=0.50,
        min_confidence_to_trade=0.60,  # Very high bar to trade
        notes="Anti-predictive. Direction signals should be dampened.",
    ),
    Regime.VOL_EXPANSION: RegimeProfile(
        name="vol_expansion",
        tier="weak",
        observed_hr=0.50,
        n_samples=5,
        confidence_multiplier=0.70,
        min_confidence_to_trade=0.55,
        notes="Vol models useful but direction uncertain",
    ),
    Regime.POST_JUMP: RegimeProfile(
        name="post_jump",
        tier="strong",
        observed_hr=0.65,
        n_samples=8,
        confidence_multiplier=1.10,
        min_confidence_to_trade=0.52,
        notes="Best regime for directional trading",
        tier="weak",             # HR=50% (n=6) — coin flip
        observed_hr=0.50,
        n_samples=6,
        confidence_multiplier=0.75,
        min_confidence_to_trade=0.55,
        notes="Coin flip. Not enough edge to trade confidently.",
    ),
    Regime.CASCADE_RISK: RegimeProfile(
        name="cascade_risk",
        tier="blocked",
        observed_hr=0.40,
        n_samples=5,
        confidence_multiplier=0.40,
        min_confidence_to_trade=0.65,
        notes="Extreme uncertainty — derivatives module dominates",
    ),
    Regime.EVENT_WINDOW: RegimeProfile(
        name="event_window",
        tier="blocked",
        observed_hr=0.50,
        n_samples=0,
        confidence_multiplier=0.30,
        min_confidence_to_trade=0.70,
        notes="Event-driven regime — model has no edge",
    ),
    Regime.ILLIQUID: RegimeProfile(
        name="illiquid",
        tier="blocked",
        observed_hr=0.45,
        n_samples=2,
        confidence_multiplier=0.30,
        min_confidence_to_trade=0.65,
        notes="Thin liquidity — model unreliable, wide spreads",
    ),
}


# ═══════════════════════════════════════════════════════════════
# REGIME GATE
# ═══════════════════════════════════════════════════════════════

class RegimeGate:
    """
    Gates forecast outputs based on regime quality.

    Applies three operations:
    1. Confidence dampening for weak/blocked regimes
    2. Direction clamping toward 0.50 in anti-predictive regimes
    3. Trade/no-trade decision based on gated confidence
    Applies four operations:
    1. Confidence dampening for weak/blocked regimes
    2. Signal inversion for anti-predictive regimes (HR < 30%)
    3. EV threshold: only trade when p_correct > breakeven
    4. Trade/no-trade decision based on gated confidence

    The gate modifies PredictionTargets in-place and returns a
    GateDecision with the action and reasoning.
    """

    def __init__(self, profiles: dict[Regime, RegimeProfile] = None):
        self.profiles = profiles or DEFAULT_REGIME_PROFILES
    # Default Kalshi fee structure: ~7% round-trip on 50c contracts.
    # Breakeven p_correct = 1 / (2 - fee_pct) ≈ 0.517 for 3.4% fee,
    # but we add margin → 0.54 minimum.  This is the MINIMUM
    # directional probability (distance from 0.50) required to trade.
    DEFAULT_MIN_EV_EDGE = 0.04   # |dir_prob - 0.50| must exceed this

    def __init__(self, profiles: dict[Regime, RegimeProfile] = None,
                 min_ev_edge: float = None):
        self.profiles = profiles or DEFAULT_REGIME_PROFILES
        self.min_ev_edge = min_ev_edge if min_ev_edge is not None \
            else self.DEFAULT_MIN_EV_EDGE
        # Track live performance to update profiles
        self._live_hits: dict[str, list[bool]] = {}

    def apply(self,
              targets: PredictionTargets,
              regime_probs: dict[Regime, float],
              ) -> "GateDecision":
        """
        Apply regime gating to prediction targets.

        Modifies targets.direction_prob and returns GateDecision.
        """
        # Dominant regime
        dominant = max(regime_probs, key=regime_probs.get) \
            if regime_probs else Regime.RANGE
        profile = self.profiles.get(dominant, self.profiles[Regime.RANGE])

        original_dir_prob = targets.direction_prob

        # ── Step 1: Confidence multiplier ──────────────────
        # Scale the distance from 0.50 by the regime multiplier
        distance = targets.direction_prob - 0.50
        scaled_distance = distance * profile.confidence_multiplier
        targets.direction_prob = 0.50 + scaled_distance

        # Clamp
        targets.direction_prob = max(0.05, min(0.95, targets.direction_prob))

        # ── Step 2: Blocked regime dampening ───────────────
        if profile.tier == "blocked":
            # For anti-predictive regimes, aggressively shrink toward 0.50
            dampen = 0.60  # 60% shrinkage
        # ── Step 2: Anti-predictive regime handling ─────────
        if profile.tier == "blocked":
            # For regimes where the model is anti-predictive,
            # invert the signal if HR < 30% (reliably wrong = usable),
            # otherwise dampen toward 0.50.
            if profile.observed_hr < 0.30:
                # INVERT: model is reliably wrong, flip the direction
                targets.direction_prob = 1.0 - targets.direction_prob
                targets.expected_return = -targets.expected_return
                # Then dampen the inverted signal (don't fully trust it)
                dampen = 0.40  # 40% shrinkage on inverted signal
            else:
                # DAMPEN: model is noisy but not reliably wrong
                dampen = 0.60  # 60% shrinkage
            targets.direction_prob = (
                targets.direction_prob * (1 - dampen) + 0.50 * dampen
            )
            # Also shrink expected return
            targets.expected_return *= (1 - dampen * 0.7)

        # ── Step 3: Trade decision ─────────────────────────
        final_confidence = abs(targets.direction_prob - 0.50)
        min_conf_distance = abs(profile.min_confidence_to_trade - 0.50)

        should_trade = final_confidence >= min_conf_distance
        # ── Step 3: EV threshold + trade decision ──────────
        final_confidence = abs(targets.direction_prob - 0.50)
        min_conf_distance = abs(profile.min_confidence_to_trade - 0.50)

        # Must clear BOTH the regime-specific threshold AND the EV edge
        should_trade = (final_confidence >= min_conf_distance
                        and final_confidence >= self.min_ev_edge)

        # Blocked regimes need extra justification
        if profile.tier == "blocked" and final_confidence < 0.10:
            should_trade = False

        action = "trade" if should_trade else "skip"
        if profile.tier == "blocked":
            action = "skip" if not should_trade else "trade_cautious"
        # Only allow trading in regimes with enough samples and
        # demonstrated edge.  This is the selectivity constraint:
        # fewer trades, only when p_correct is meaningfully > breakeven.
        if profile.tier not in ("strong",) and not should_trade:
            action = "skip"
        elif profile.tier == "blocked":
            action = "skip" if not should_trade else "trade_cautious"
        else:
            action = "trade" if should_trade else "skip"

        return GateDecision(
            action=action,
            regime=dominant.value if isinstance(dominant, Regime) else str(dominant),
            tier=profile.tier,
            original_direction_prob=original_dir_prob,
            gated_direction_prob=targets.direction_prob,
            confidence_multiplier=profile.confidence_multiplier,
            ev_edge=round(final_confidence, 4),
            min_ev_required=round(self.min_ev_edge, 4),
            reason=profile.notes,
        )

    def record_outcome(self, regime: str, was_correct: bool):
        """Record live outcome for adaptive learning."""
        if regime not in self._live_hits:
            self._live_hits[regime] = []
        self._live_hits[regime].append(was_correct)

        # Update profile if we have enough live samples
        hits = self._live_hits[regime]
        if len(hits) >= 20:
            live_hr = sum(hits) / len(hits)
            for r, p in self.profiles.items():
                if p.name == regime:
                    # Blend live HR with prior (70% live, 30% prior)
                    blended = 0.7 * live_hr + 0.3 * p.observed_hr
                    p.observed_hr = blended
                    # Update tier
                    if blended > 0.55:
                        p.tier = "strong"
                        p.confidence_multiplier = min(1.2, 0.8 + blended)
                    elif blended > 0.45:
                        p.tier = "weak"
                        p.confidence_multiplier = 0.6 + blended * 0.4
                    else:
                        p.tier = "blocked"
                        p.confidence_multiplier = max(0.3, blended)
                    break


@dataclass
class GateDecision:
    """Output of the regime gate."""
    action: str = "skip"               # "trade", "skip", "trade_cautious"
    regime: str = "range"
    tier: str = "weak"
    original_direction_prob: float = 0.50
    gated_direction_prob: float = 0.50
    confidence_multiplier: float = 1.0
    ev_edge: float = 0.0              # |dir_prob - 0.50| after gating
    min_ev_required: float = 0.04     # minimum edge to trade
    reason: str = ""


# ═══════════════════════════════════════════════════════════════
# ARBITRAGE DETECTOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    ticker: str = ""
    type: str = "spread"           # "spread", "lag", "cross_market"
    buy_side: str = "yes"
    buy_price_cents: int = 0
    sell_side: str = "no"
    sell_price_cents: int = 0
    spread_cents: int = 0          # guaranteed profit in cents
    profit_pct: float = 0.0
    confidence: float = 0.95
    max_contracts: int = 10
    notes: str = ""


class ArbitrageDetector:
    """
    Detects risk-free or near-risk-free arbitrage opportunities
    in prediction markets.

    Types of arbitrage:
    1. YES/NO Spread: Buy YES + Buy NO for less than $1.00
    2. Cross-market Lag: Same event priced differently on Kalshi vs model
    3. Time Decay: Buy cheap options near expiry when model is confident

    Inspired by successful X bots (PolyXBot, etc.) achieving 95%+ win rates.
    """

    def __init__(self, min_spread_cents: int = 2):
        self.min_spread_cents = min_spread_cents

    def check_spread_arb(self,
                          yes_ask: int,
                          no_ask: int,
                          ticker: str = "") -> Optional[ArbitrageOpportunity]:
        """
        Check for YES/NO spread arbitrage.

        If yes_ask + no_ask < 100 cents, guaranteed profit exists.
        Example: YES @ 48c + NO @ 49c = 97c cost, always pays $1.00.
        """
        if yes_ask <= 0 or no_ask <= 0:
            return None

        total_cost = yes_ask + no_ask
        if total_cost >= 100:
            return None  # No arb

        spread = 100 - total_cost
        if spread < self.min_spread_cents:
            return None  # Below threshold

        profit_pct = spread / total_cost * 100
        max_contracts = min(50, 500 // total_cost)  # Max $5 per position

        return ArbitrageOpportunity(
            ticker=ticker,
            type="spread",
            buy_side="yes+no",
            buy_price_cents=total_cost,
            sell_side="guaranteed_payout",
            sell_price_cents=100,
            spread_cents=spread,
            profit_pct=round(profit_pct, 2),
            confidence=0.99,
            max_contracts=max_contracts,
            notes=f"Buy YES@{yes_ask}c + NO@{no_ask}c = {total_cost}c, payout $1.00",
        )

    def check_model_edge_arb(self,
                               model_prob: float,
                               market_prob: float,
                               yes_ask: int,
                               no_ask: int,
                               ticker: str = "",
                               min_edge: float = 0.15) -> Optional[ArbitrageOpportunity]:
        """
        Check for model-vs-market edge that qualifies as near-arbitrage.

        When the model is very confident (>85%) and the market is
        significantly mispriced, this approaches risk-free.
        """
        edge = abs(model_prob - market_prob)
        if edge < min_edge:
            return None

        if model_prob > market_prob + min_edge:
            # Model says YES is underpriced
            if yes_ask <= 0:
                return None
            ev_cents = model_prob * (100 - yes_ask) - (1 - model_prob) * yes_ask
            if ev_cents < 5:  # Need at least 5c EV
                return None
            return ArbitrageOpportunity(
                ticker=ticker,
                type="model_edge",
                buy_side="yes",
                buy_price_cents=yes_ask,
                sell_side="payout",
                sell_price_cents=100,
                spread_cents=int(ev_cents),
                profit_pct=round(ev_cents / yes_ask * 100, 2) if yes_ask > 0 else 0,
                confidence=min(0.95, model_prob),
                max_contracts=min(20, 200 // yes_ask) if yes_ask > 0 else 0,
                notes=f"Model={model_prob:.0%} vs Market={market_prob:.0%}, edge={edge:.0%}",
            )
        elif model_prob < market_prob - min_edge:
            # Model says NO is underpriced
            if no_ask <= 0:
                return None
            no_model_prob = 1 - model_prob
            ev_cents = no_model_prob * (100 - no_ask) - model_prob * no_ask
            if ev_cents < 5:
                return None
            return ArbitrageOpportunity(
                ticker=ticker,
                type="model_edge",
                buy_side="no",
                buy_price_cents=no_ask,
                sell_side="payout",
                sell_price_cents=100,
                spread_cents=int(ev_cents),
                profit_pct=round(ev_cents / no_ask * 100, 2) if no_ask > 0 else 0,
                confidence=min(0.95, no_model_prob),
                max_contracts=min(20, 200 // no_ask) if no_ask > 0 else 0,
                notes=f"Model={model_prob:.0%} vs Market={market_prob:.0%}, edge={edge:.0%}",
            )

        return None

    def scan_markets(self, markets: list[dict],
                      model_probs: dict[str, float] = None) -> list[ArbitrageOpportunity]:
        """
        Scan a list of market dicts for arbitrage opportunities.

        Args:
            markets: List of Kalshi market dicts
            model_probs: Optional dict of ticker -> model_prob from ensemble

        Returns:
            List of ArbitrageOpportunity sorted by profit potential.
        """
        opportunities = []

        for mkt in markets:
            ticker = mkt.get("ticker", "")
            yes_ask = mkt.get("yes_ask", 0) or 0
            no_ask = mkt.get("no_ask", 0) or 0

            # Check spread arb
            arb = self.check_spread_arb(yes_ask, no_ask, ticker)
            if arb:
                opportunities.append(arb)

            # Check model edge arb
            if model_probs and ticker in model_probs:
                yes_bid = mkt.get("yes_bid", 0) or 0
                market_prob = (yes_bid + yes_ask) / 200.0 if (yes_bid > 0 and yes_ask > 0) \
                    else yes_ask / 100.0 if yes_ask > 0 else 0.5
                edge_arb = self.check_model_edge_arb(
                    model_probs[ticker], market_prob,
                    yes_ask, no_ask, ticker,
                )
                if edge_arb:
                    opportunities.append(edge_arb)

        # Sort by profit potential
        opportunities.sort(key=lambda x: x.spread_cents, reverse=True)
        return opportunities


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def should_trade(targets: PredictionTargets,
                 regime_probs: dict[Regime, float],
                 gate: RegimeGate = None) -> tuple[bool, GateDecision]:
    """
    Quick check: should we trade based on regime gating?

    Returns (should_trade: bool, decision: GateDecision).
    """
    if gate is None:
        gate = RegimeGate()
    decision = gate.apply(targets, regime_probs)
    return decision.action in ("trade", "trade_cautious"), decision
