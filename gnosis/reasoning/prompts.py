"""
Prompt Builder — Structured prompts for LLM reasoning modes.
==============================================================
Each mode produces a system prompt and a data-rich user prompt
that the LLM uses to reason about the forecast pipeline output.

Modes:
  FULL_ANALYSIS    — Complete forecast interpretation + trade suggestion
  REGIME_DEEP_DIVE — Deep analysis of KPCOFGS regime + historical context
  TRADE_PLAN       — Specific entry/exit/sizing recommendation
  RISK_ASSESSMENT  — Downside scenarios, position sizing, hedging
  EXTRAPOLATION    — Forward-looking scenarios based on current state
  SELF_CRITIQUE    — Challenge the forecast assumptions and identify weaknesses
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT_BASE = """You are the reasoning core of a dual-bot crypto trading system.

SYSTEM ARCHITECTURE:
- ClawdBot-V1: Real-time 14-paradigm ensemble forecaster (500ms latency)
  * 9 base modules: technical, classical_stats, macro, derivatives, microstructure,
    onchain, sentiment, sequence_model, crowd_prior
  * 3 meta-modules: regime_detector, meta_learner, monte_carlo (50K paths)
  * 2 advanced: particle_candle (event-quantized bars), manifold_pattern (motif detection)
  * Regime gate: blocks anti-predictive regimes, adjusts confidence
  * Auto-fix: isotonic/Platt calibration of direction probabilities

- Yoshi-Bot: Research harness + execution layer
  * KPCOFGS: 7-level regime classifier (K=kinetics, P=pressure, C=crowd,
    O=order-flow, F=force, G=game-state, S=scenario)
  * Walk-forward validation with purge/embargo (no lookahead)
  * Proper scoring: pinball loss, coverage, sharpness, CRPS
  * Backtester: PnL, Sharpe, drawdown, per-regime breakdown
  * Ralph Loop: nested hyperparameter optimization
  * Execution: Kalshi API, Telegram alerts

YOUR ROLE:
You receive the complete pipeline output — forecast, regime classification,
validation metrics, backtest stats — and produce structured analysis.
You must be HONEST about signal quality. If the forecast is poor, say so.
If the regime suggests no edge, recommend sitting out.
Never hallucinate edges that don't exist in the data.

CRITICAL RULES:
1. Base ALL analysis on the provided data — never invent statistics
2. If hit_rate < 55% or p-value > 0.05, the signal is NOT statistically significant
3. If regime_gate action = "skip", respect it — the gate has calibrated edge data
4. Expected Value (EV) must exceed 4% (|direction_prob - 0.50| > 0.04) to suggest a trade
5. Always quantify uncertainty — use the confidence intervals and regime entropy
6. Reference specific numbers from the data in your analysis
7. If validation/backtest data is missing, note it and reduce confidence
"""


def _format_number(v, decimals=4):
    """Format a number for prompt inclusion."""
    if isinstance(v, float):
        if abs(v) < 0.0001 and v != 0:
            return f"{v:.6f}"
        return f"{v:.{decimals}f}"
    return str(v)


class PromptBuilder:
    """Builds structured prompts for different reasoning modes."""

    @staticmethod
    def full_analysis(
        forecast: Dict[str, Any],
        kpcofgs: Dict[str, Any],
        kpcofgs_regime: str,
        validation: Dict[str, Any] = None,
        backtest: Dict[str, Any] = None,
        opportunities: List[Dict] = None,
    ) -> tuple:
        """Build prompt for full analysis mode.

        Returns (system_prompt, user_prompt) tuple.
        """
        system = SYSTEM_PROMPT_BASE + """
TASK: Produce a comprehensive analysis of the current forecast.

OUTPUT FORMAT (JSON):
{
  "signal_quality": "STRONG|MODERATE|WEAK|POOR",
  "confidence_level": 0.0-1.0,
  "narrative": "2-3 sentence plain-English summary of what the data shows",
  "regime_interpretation": {
    "current_regime": "string",
    "kpcofgs_scenario": "string",
    "regime_confidence": 0.0-1.0,
    "regime_implications": "what this regime historically means for price action"
  },
  "forecast_assessment": {
    "direction_conviction": "STRONG_UP|LEAN_UP|NEUTRAL|LEAN_DOWN|STRONG_DOWN",
    "price_target_24h": number,
    "price_range_24h": [low, high],
    "key_drivers": ["driver1", "driver2"],
    "key_risks": ["risk1", "risk2"]
  },
  "trade_suggestion": {
    "action": "BUY|SELL|HOLD|SKIP",
    "reason": "1-2 sentence justification",
    "entry_zone": [low, high] or null,
    "stop_loss": number or null,
    "take_profit": number or null,
    "position_size_pct": 0.0-5.0,
    "time_horizon_hours": number,
    "edge_source": "which module/regime provides the edge"
  },
  "risk_assessment": {
    "var_95_interpretation": "string",
    "tail_risk_level": "LOW|MODERATE|HIGH|EXTREME",
    "jump_crash_assessment": "string",
    "max_recommended_exposure_pct": 0.0-100.0
  },
  "validation_commentary": "assessment of walk-forward and backtest results",
  "extrapolations": [
    {
      "scenario": "description",
      "probability": 0.0-1.0,
      "price_impact": "string"
    }
  ],
  "self_critique": "honest assessment of forecast weaknesses",
  "next_steps": ["step1", "step2"]
}
"""
        user = _build_data_section(
            forecast, kpcofgs, kpcofgs_regime,
            validation, backtest, opportunities,
        )
        return system, user

    @staticmethod
    def regime_deep_dive(
        forecast: Dict[str, Any],
        kpcofgs: Dict[str, Any],
        kpcofgs_regime: str,
    ) -> tuple:
        """Build prompt for deep regime analysis."""
        system = SYSTEM_PROMPT_BASE + """
TASK: Deep analysis of the current market regime using KPCOFGS classification.

Focus on:
1. What each KPCOFGS level tells us independently
2. How the levels interact (e.g., K=TRENDING + G=EXHAUSTION = potential reversal)
3. Historical behavior of this specific regime combination
4. Regime transition probabilities — what regime is most likely next?
5. Optimal strategy for this regime (trend-follow, mean-revert, sit-out)

OUTPUT FORMAT (JSON):
{
  "regime_stack": {
    "K_analysis": "kinetics interpretation",
    "P_analysis": "pressure interpretation",
    "C_analysis": "crowd flow interpretation",
    "O_analysis": "order-flow interpretation",
    "F_analysis": "force/momentum interpretation",
    "G_analysis": "game-state interpretation",
    "S_analysis": "scenario interpretation"
  },
  "regime_interactions": ["interaction1", "interaction2"],
  "transition_forecast": {
    "most_likely_next_regime": "string",
    "transition_probability": 0.0-1.0,
    "transition_trigger": "what would cause the transition"
  },
  "optimal_strategy": {
    "strategy_type": "TREND_FOLLOW|MEAN_REVERT|BREAKOUT|SIT_OUT",
    "reason": "string",
    "confidence": 0.0-1.0
  },
  "regime_entropy_interpretation": "what the entropy level means for certainty",
  "historical_context": "how this regime has historically resolved"
}
"""
        data = f"""SYMBOL: {forecast.get('symbol', 'BTCUSDT')}

KPCOFGS CLASSIFICATION:
{json.dumps(kpcofgs, indent=2)}

MAPPED REGIME: {kpcofgs_regime}

CLAWDBOT REGIME: {forecast.get('regime', 'unknown')}
REGIME PROBABILITIES: {json.dumps(forecast.get('regime_probs', {}), indent=2)}

VOLATILITY: {forecast.get('volatility', 0)}
DIRECTION PROB: {forecast.get('confidence', 0.5)}
GATE DECISION: {json.dumps(forecast.get('gate_decision', {}), indent=2)}
"""
        return system, data

    @staticmethod
    def trade_plan(
        forecast: Dict[str, Any],
        kpcofgs: Dict[str, Any],
        kpcofgs_regime: str,
        backtest: Dict[str, Any] = None,
        risk_budget_usd: float = 500.0,
        max_leverage: float = 2.0,
    ) -> tuple:
        """Build prompt for specific trade plan."""
        system = SYSTEM_PROMPT_BASE + f"""
TASK: Generate a specific trade plan given the forecast and risk constraints.

RISK CONSTRAINTS:
- Maximum position size: ${risk_budget_usd} USD
- Maximum leverage: {max_leverage}x
- Daily loss limit: ${risk_budget_usd * 0.2} USD (20% of budget)
- Must use stop-loss on every trade

OUTPUT FORMAT (JSON):
{{
  "decision": "TRADE|NO_TRADE",
  "reason": "1-2 sentences",
  "trade": {{
    "side": "LONG|SHORT",
    "entry_price": number,
    "entry_type": "MARKET|LIMIT",
    "stop_loss": number,
    "take_profit": number,
    "position_size_usd": number,
    "leverage": number,
    "risk_reward_ratio": number,
    "max_loss_usd": number,
    "expected_profit_usd": number,
    "time_horizon_hours": number
  }} or null,
  "scaling_plan": "how to scale in/out",
  "invalidation": "what would invalidate this trade",
  "alternative_actions": ["alt1", "alt2"]
}}
"""
        user = _build_data_section(
            forecast, kpcofgs, kpcofgs_regime,
            backtest=backtest,
        )
        user += f"\n\nRISK BUDGET: ${risk_budget_usd} | MAX LEVERAGE: {max_leverage}x"
        return system, user

    @staticmethod
    def risk_assessment(
        forecast: Dict[str, Any],
        kpcofgs: Dict[str, Any],
        backtest: Dict[str, Any] = None,
    ) -> tuple:
        """Build prompt for risk assessment."""
        system = SYSTEM_PROMPT_BASE + """
TASK: Comprehensive risk assessment of the current market state.

Focus on:
1. Downside tail scenarios (VaR/CVaR interpretation)
2. Jump and crash probabilities
3. Regime-specific risks
4. Position sizing given the risk profile
5. Hedging suggestions

OUTPUT FORMAT (JSON):
{
  "overall_risk_level": "LOW|MODERATE|HIGH|EXTREME",
  "risk_score": 0.0-10.0,
  "tail_risk_analysis": {
    "var_95_meaning": "string",
    "var_99_meaning": "string",
    "cvar_interpretation": "string",
    "worst_case_24h": "string"
  },
  "jump_crash_analysis": {
    "jump_probability": number,
    "crash_probability": number,
    "cascade_risk": "string",
    "black_swan_proximity": "LOW|MODERATE|HIGH"
  },
  "regime_specific_risks": ["risk1", "risk2"],
  "position_sizing": {
    "max_recommended_pct": number,
    "kelly_fraction_estimate": number,
    "reason": "string"
  },
  "hedging_suggestions": ["suggestion1", "suggestion2"],
  "risk_triggers": ["trigger that would escalate risk level"]
}
"""
        mc = forecast.get("mc_summary", {}) if isinstance(forecast.get("mc_summary"), dict) else {}
        user = f"""SYMBOL: {forecast.get('symbol', 'BTCUSDT')}

RISK DATA:
Current Price: ${forecast.get('current_price', 0):,.2f}
Predicted Price: ${forecast.get('predicted_price', 0):,.2f}
Volatility: {forecast.get('volatility', 0)}
Direction: {forecast.get('direction', 'flat')} ({forecast.get('confidence', 0.5)*100:.1f}%)

VaR (95%): {forecast.get('var_95', 0)*100:.3f}%
VaR (99%): {forecast.get('var_99', 0)*100:.3f}%
Price Q05: ${forecast.get('price_q05', 0):,.2f}
Price Q95: ${forecast.get('price_q95', 0):,.2f}

Jump Prob: {mc.get('mc_jump_prob', forecast.get('jump_prob', 0))*100:.2f}%
Crash Prob: {mc.get('mc_crash_prob', forecast.get('crash_prob', 0))*100:.2f}%

REGIME: {forecast.get('regime', 'unknown')}
GATE DECISION: {json.dumps(forecast.get('gate_decision', {}), indent=2)}

KPCOFGS: {json.dumps(kpcofgs, indent=2)}
"""
        if backtest:
            user += f"\nBACKTEST STATS:\n{json.dumps(backtest, indent=2, default=str)}"
        return system, user

    @staticmethod
    def extrapolation(
        forecast: Dict[str, Any],
        kpcofgs: Dict[str, Any],
        kpcofgs_regime: str,
        validation: Dict[str, Any] = None,
    ) -> tuple:
        """Build prompt for forward-looking extrapolation."""
        system = SYSTEM_PROMPT_BASE + """
TASK: Generate forward-looking scenarios based on the current state.

Consider:
1. What happens if current regime persists for 24h/48h/1 week?
2. What catalysts could shift the regime?
3. What does the KPCOFGS entropy suggest about regime stability?
4. Monte Carlo distribution shape — is it fat-tailed, skewed, bimodal?
5. Cross-asset implications (SPX, DXY correlation if available)

OUTPUT FORMAT (JSON):
{
  "base_case": {
    "scenario": "description",
    "probability": 0.0-1.0,
    "price_24h": number,
    "price_48h": number,
    "price_1w": number,
    "key_assumption": "string"
  },
  "bull_case": {
    "scenario": "description",
    "probability": 0.0-1.0,
    "price_24h": number,
    "trigger": "what would cause this"
  },
  "bear_case": {
    "scenario": "description",
    "probability": 0.0-1.0,
    "price_24h": number,
    "trigger": "what would cause this"
  },
  "black_swan": {
    "scenario": "description",
    "probability": 0.0-1.0,
    "price_impact": "string"
  },
  "regime_transition_forecast": {
    "current_stability": "STABLE|TRANSITIONING|UNSTABLE",
    "most_likely_transition": "string",
    "transition_timeline": "string"
  },
  "key_levels_to_watch": [
    {"price": number, "significance": "string"}
  ],
  "actionable_triggers": [
    {"condition": "string", "action": "string"}
  ]
}
"""
        user = _build_data_section(
            forecast, kpcofgs, kpcofgs_regime, validation=validation,
        )
        return system, user

    @staticmethod
    def self_critique(
        forecast: Dict[str, Any],
        kpcofgs: Dict[str, Any],
        validation: Dict[str, Any] = None,
        backtest: Dict[str, Any] = None,
    ) -> tuple:
        """Build prompt for self-critique of the forecast."""
        system = SYSTEM_PROMPT_BASE + """
TASK: Critically evaluate the forecast system's output. Be brutally honest.

Evaluate:
1. Are the module weights reasonable? Which modules had real data?
2. Is the regime classification trustworthy given the data?
3. Does the walk-forward hit rate suggest actual edge or random chance?
4. Are the confidence levels calibrated (overconfident? underconfident?)
5. What could the system be getting wrong?
6. What data is missing that would improve the forecast?

OUTPUT FORMAT (JSON):
{
  "overall_assessment": "TRUSTWORTHY|CAUTIOUS|SKEPTICAL|UNRELIABLE",
  "confidence_calibration": {
    "is_overconfident": boolean,
    "is_underconfident": boolean,
    "calibration_notes": "string"
  },
  "data_quality": {
    "modules_with_real_data": ["module1", "module2"],
    "modules_with_priors_only": ["module3"],
    "critical_missing_data": ["data1", "data2"]
  },
  "statistical_validity": {
    "hit_rate_significant": boolean,
    "sample_size_adequate": boolean,
    "look_ahead_risk": "LOW|MODERATE|HIGH",
    "overfitting_risk": "LOW|MODERATE|HIGH"
  },
  "regime_reliability": {
    "classification_confidence": 0.0-1.0,
    "entropy_assessment": "string",
    "alternative_regimes": ["regime that could also fit"]
  },
  "known_weaknesses": ["weakness1", "weakness2"],
  "improvement_suggestions": ["suggestion1", "suggestion2"],
  "bottom_line": "one-sentence honest verdict"
}
"""
        user = _build_data_section(
            forecast, kpcofgs, "unknown",
            validation=validation, backtest=backtest,
        )
        return system, user


def _build_data_section(
    forecast: Dict[str, Any],
    kpcofgs: Dict[str, Any],
    kpcofgs_regime: str,
    validation: Dict[str, Any] = None,
    backtest: Dict[str, Any] = None,
    opportunities: List[Dict] = None,
) -> str:
    """Build the data section that's common across prompts."""
    sections = []

    # Forecast data
    sections.append("=" * 60)
    sections.append("CLAWDBOT 14-PARADIGM ENSEMBLE FORECAST")
    sections.append("=" * 60)
    fc = forecast or {}
    sections.append(f"Symbol: {fc.get('symbol', 'BTCUSDT')}")
    sections.append(f"Current Price: ${fc.get('current_price', 0):,.2f}")
    sections.append(f"Predicted Price: ${fc.get('predicted_price', 0):,.2f}")
    sections.append(f"Direction: {fc.get('direction', 'flat')}")
    sections.append(f"Confidence: {fc.get('confidence', 0.5)*100:.1f}%")
    sections.append(f"Volatility: {fc.get('volatility', 0)}")
    sections.append(f"Regime (ClawdBot): {fc.get('regime', 'unknown')}")
    sections.append(f"Modules Run: {fc.get('modules_run', 0)}/14")
    sections.append(f"VaR 95%: {fc.get('var_95', 0)*100:.3f}%")
    sections.append(f"VaR 99%: {fc.get('var_99', 0)*100:.3f}%")
    sections.append(f"Price Q05: ${fc.get('price_q05', 0):,.2f}")
    sections.append(f"Price Q50: ${fc.get('price_q50', 0):,.2f}")
    sections.append(f"Price Q95: ${fc.get('price_q95', 0):,.2f}")

    # Gate decision
    gate = fc.get("gate_decision", {})
    if gate:
        sections.append(f"\nREGIME GATE:")
        sections.append(f"  Action: {gate.get('action', '?')}")
        sections.append(f"  Tier: {gate.get('tier', '?')}")
        sections.append(f"  Original Dir Prob: {gate.get('original_dir_prob', 0.5):.4f}")
        sections.append(f"  Gated Dir Prob: {gate.get('gated_dir_prob', 0.5):.4f}")
        sections.append(f"  EV Edge: {gate.get('ev_edge', 0):.4f}")
        sections.append(f"  Min EV Required: {gate.get('min_ev_required', 0.04):.4f}")

    # Gating weights
    weights = fc.get("gating_weights", {})
    if weights:
        sections.append(f"\nGATING WEIGHTS:")
        for name, w in sorted(weights.items(), key=lambda x: -x[1]):
            sections.append(f"  {name}: {w:.3f}")

    # KPCOFGS
    sections.append(f"\n{'='*60}")
    sections.append("YOSHI KPCOFGS 7-LEVEL REGIME")
    sections.append("=" * 60)
    for key, val in sorted(kpcofgs.items()):
        sections.append(f"  {key}: {val}")
    sections.append(f"  Mapped regime: {kpcofgs_regime}")

    # Validation
    if validation:
        sections.append(f"\n{'='*60}")
        sections.append("WALK-FORWARD VALIDATION (Purge/Embargo)")
        sections.append("=" * 60)
        sections.append(f"  Hit Rate: {validation.get('hit_rate', 0)*100:.1f}%")
        sections.append(f"  P-Value: {validation.get('hit_rate_p_value', 1):.4f}")
        sections.append(f"  Coverage (90%): {validation.get('coverage_90', 0)*100:.1f}%")
        sections.append(f"  CRPS: {validation.get('crps', 0):.6f}")
        sections.append(f"  Sharpness: {validation.get('sharpness', 0):.6f}")
        sections.append(f"  MAE: {validation.get('mae', 0):.6f}")
        sections.append(f"  Samples: {validation.get('n_samples', 0)}")
        sections.append(f"  Folds: {validation.get('n_folds', 0)}")
        fold_hrs = validation.get("fold_hit_rates", [])
        if fold_hrs:
            sections.append(f"  Per-fold HR: {[f'{h:.1%}' for h in fold_hrs]}")

    # Backtest
    if backtest:
        sections.append(f"\n{'='*60}")
        sections.append("BACKTEST RESULTS")
        sections.append("=" * 60)
        sections.append(f"  Trades: {backtest.get('n_trades', 0)}")
        sections.append(f"  Win Rate: {backtest.get('win_rate', 0)*100:.1f}%")
        sections.append(f"  Total PnL: ${backtest.get('total_pnl', 0):+.2f}")
        sections.append(f"  Total Return: {backtest.get('total_return_pct', 0):+.1f}%")
        sections.append(f"  Sharpe: {backtest.get('sharpe', 0):.3f}")
        sections.append(f"  Max Drawdown: {backtest.get('max_drawdown_pct', 0):.1f}%")
        sections.append(f"  Profit Factor: {backtest.get('profit_factor', 0):.2f}")
        per_regime = backtest.get("per_regime", {})
        if per_regime:
            sections.append(f"  Per-regime breakdown:")
            for regime, stats in per_regime.items():
                wr = stats['wins'] / stats['n'] * 100 if stats['n'] > 0 else 0
                sections.append(
                    f"    {regime}: {stats['n']} trades, "
                    f"WR={wr:.0f}%, PnL=${stats['pnl']:+.2f}"
                )

    # Opportunities
    if opportunities:
        sections.append(f"\n{'='*60}")
        sections.append("KALSHI OPPORTUNITIES")
        sections.append("=" * 60)
        for opp in opportunities:
            sections.append(f"  {json.dumps(opp)}")

    return "\n".join(sections)
