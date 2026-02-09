"""
Reasoning Engine — Orchestrates LLM analysis of the full pipeline.
====================================================================
This is the main entry point for the reasoning layer. It:
  1. Collects all pipeline data (forecast, KPCOFGS, validation, backtest)
  2. Selects the appropriate analysis mode
  3. Builds structured prompts
  4. Calls the LLM
  5. Parses and validates the response
  6. Packages into a ReasoningResult

The engine runs AFTER the forecast pipeline completes, adding an
intelligence layer that can interpret, extrapolate, and suggest actions
that the raw numbers alone cannot provide.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .client import LLMClient, LLMConfig, LLMResponse
from .prompts import PromptBuilder


class AnalysisMode(Enum):
    """Available reasoning modes."""
    FULL_ANALYSIS = "full"
    REGIME_DEEP_DIVE = "regime"
    TRADE_PLAN = "trade"
    RISK_ASSESSMENT = "risk"
    EXTRAPOLATION = "extrapolation"
    SELF_CRITIQUE = "critique"
    AUTO = "auto"  # Engine picks the best mode based on data


@dataclass
class ReasoningConfig:
    """Configuration for the reasoning engine."""
    mode: AnalysisMode = AnalysisMode.FULL_ANALYSIS
    llm_config: LLMConfig = field(default_factory=LLMConfig.from_yaml)
    risk_budget_usd: float = 500.0
    max_leverage: float = 2.0
    # Which sub-analyses to include in AUTO mode
    auto_include_critique: bool = True
    auto_include_extrapolation: bool = True
    # Verbose output
    verbose: bool = True


@dataclass
class ReasoningResult:
    """Complete output from the reasoning engine."""
    # Primary analysis (from the chosen mode)
    analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Mode that was run
    mode: str = "full"
    
    # Raw LLM response
    raw_response: str = ""
    
    # Additional analyses (populated in AUTO mode)
    critique: Optional[Dict[str, Any]] = None
    extrapolation: Optional[Dict[str, Any]] = None
    
    # LLM metadata
    model_used: str = ""
    tokens_used: Dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    is_stub: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "analysis": self.analysis,
            "mode": self.mode,
            "critique": self.critique,
            "extrapolation": self.extrapolation,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "elapsed_ms": self.elapsed_ms,
            "is_stub": self.is_stub,
            "error": self.error,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def print_summary(self):
        """Print a human-readable summary."""
        a = self.analysis or {}
        
        print(f"\n{'='*64}")
        print(f"  LLM REASONING ANALYSIS ({self.mode.upper()})")
        print(f"{'='*64}")
        
        if self.error:
            print(f"  ERROR: {self.error}")
            return
        
        if self.is_stub:
            print(f"  [STUB MODE — No LLM API configured]")
        
        # Signal quality
        sq = a.get("signal_quality", a.get("overall_assessment", "?"))
        print(f"  Signal Quality:    {sq}")
        
        # Narrative
        narrative = a.get("narrative", a.get("bottom_line", ""))
        if narrative:
            print(f"  Summary:           {narrative}")
        
        # Trade suggestion
        ts = a.get("trade_suggestion", a.get("trade", {}))
        if ts:
            action = ts.get("action", ts.get("decision", "?"))
            reason = ts.get("reason", "")
            print(f"\n  Trade Action:      {action}")
            if reason:
                print(f"  Reason:            {reason}")
            if ts.get("entry_zone"):
                print(f"  Entry Zone:        {ts['entry_zone']}")
            if ts.get("stop_loss"):
                print(f"  Stop Loss:         ${ts['stop_loss']:,.2f}")
            if ts.get("take_profit"):
                print(f"  Take Profit:       ${ts['take_profit']:,.2f}")
            if ts.get("position_size_pct"):
                print(f"  Position Size:     {ts['position_size_pct']}%")
        
        # Risk
        risk = a.get("risk_assessment", {})
        if risk:
            print(f"\n  Tail Risk:         {risk.get('tail_risk_level', '?')}")
            print(f"  Max Exposure:      {risk.get('max_recommended_exposure_pct', '?')}%")
        
        # Self-critique
        critique = self.critique or a.get("self_critique", "")
        if isinstance(critique, dict):
            bl = critique.get("bottom_line", "")
            if bl:
                print(f"\n  Self-Critique:     {bl}")
        elif isinstance(critique, str) and critique:
            print(f"\n  Self-Critique:     {critique}")
        
        # Extrapolations
        extrap = self.extrapolation or a.get("extrapolations", [])
        if isinstance(extrap, dict):
            base = extrap.get("base_case", {})
            if base:
                print(f"\n  Base Case:         {base.get('scenario', '?')} "
                      f"(p={base.get('probability', '?')})")
        elif isinstance(extrap, list) and extrap:
            for e in extrap[:3]:
                print(f"  Scenario:          {e.get('scenario', '?')} "
                      f"(p={e.get('probability', '?')})")
        
        # Next steps
        ns = a.get("next_steps", [])
        if ns:
            print(f"\n  Next Steps:")
            for step in ns[:5]:
                print(f"    - {step}")
        
        print(f"\n  Model: {self.model_used} | "
              f"Tokens: {self.tokens_used.get('total_tokens', 0)} | "
              f"Time: {self.elapsed_ms:.0f}ms")
        print(f"{'='*64}")


class ReasoningEngine:
    """Orchestrates LLM reasoning over pipeline data."""

    def __init__(self, config: ReasoningConfig = None):
        self.config = config or ReasoningConfig()
        self.client = LLMClient(self.config.llm_config)
        self.prompt_builder = PromptBuilder()

    def analyze(
        self,
        forecast: Dict[str, Any],
        kpcofgs: Dict[str, Any],
        kpcofgs_regime: str = "range",
        validation: Dict[str, Any] = None,
        backtest: Dict[str, Any] = None,
        opportunities: List[Dict] = None,
        mode: AnalysisMode = None,
    ) -> ReasoningResult:
        """Run LLM reasoning on pipeline data.

        Args:
            forecast: ClawdBot forecast dict
            kpcofgs: KPCOFGS classification summary
            kpcofgs_regime: Mapped ClawdBot regime from KPCOFGS
            validation: Walk-forward validation metrics
            backtest: Backtest statistics
            opportunities: Kalshi arbitrage opportunities
            mode: Override the configured mode

        Returns:
            ReasoningResult with structured analysis
        """
        t0 = time.time()
        mode = mode or self.config.mode
        result = ReasoningResult(mode=mode.value)

        # Auto mode: pick based on available data
        if mode == AnalysisMode.AUTO:
            mode = self._pick_mode(forecast, validation, backtest)
            result.mode = mode.value

        if self.config.verbose:
            print(f"\n[LLM] Running {mode.value} analysis...")

        # Build prompt for the chosen mode
        system_prompt, user_prompt = self._build_prompt(
            mode, forecast, kpcofgs, kpcofgs_regime,
            validation, backtest, opportunities,
        )

        # Call LLM
        response = self.client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format="json",
        )

        result.raw_response = response.content
        result.model_used = response.model
        result.tokens_used = response.usage
        result.is_stub = response.is_stub
        result.error = response.error

        if response.parsed:
            result.analysis = response.parsed
        elif response.content:
            # Try to extract useful content even if not valid JSON
            result.analysis = {"raw_text": response.content}
        else:
            result.analysis = {"error": response.error or "empty response"}

        # In AUTO mode, run additional analyses
        if self.config.mode == AnalysisMode.AUTO:
            if self.config.auto_include_critique and not response.is_stub:
                result.critique = self._run_sub_analysis(
                    AnalysisMode.SELF_CRITIQUE,
                    forecast, kpcofgs, kpcofgs_regime,
                    validation, backtest,
                )

            if self.config.auto_include_extrapolation and not response.is_stub:
                result.extrapolation = self._run_sub_analysis(
                    AnalysisMode.EXTRAPOLATION,
                    forecast, kpcofgs, kpcofgs_regime,
                    validation, backtest,
                )

        result.elapsed_ms = round((time.time() - t0) * 1000, 1)

        if self.config.verbose:
            sq = result.analysis.get("signal_quality",
                                      result.analysis.get("overall_assessment", "?"))
            print(f"[LLM] Analysis complete: signal={sq} "
                  f"({result.elapsed_ms:.0f}ms, "
                  f"{result.tokens_used.get('total_tokens', 0)} tokens)")

        return result

    def _pick_mode(
        self,
        forecast: Dict,
        validation: Dict = None,
        backtest: Dict = None,
    ) -> AnalysisMode:
        """Auto-select the best analysis mode based on available data."""
        # If we have validation + backtest, do full analysis
        if validation and backtest:
            return AnalysisMode.FULL_ANALYSIS

        # If gate says skip, focus on risk
        gate = forecast.get("gate_decision", {})
        if gate.get("action") == "skip":
            return AnalysisMode.RISK_ASSESSMENT

        # If high volatility or jump risk, focus on risk
        vol = forecast.get("volatility", 0)
        if vol > 0.05:
            return AnalysisMode.RISK_ASSESSMENT

        # Default to full analysis
        return AnalysisMode.FULL_ANALYSIS

    def _build_prompt(
        self,
        mode: AnalysisMode,
        forecast: Dict,
        kpcofgs: Dict,
        kpcofgs_regime: str,
        validation: Dict = None,
        backtest: Dict = None,
        opportunities: List[Dict] = None,
    ) -> tuple:
        """Build the appropriate prompt for the given mode."""
        if mode == AnalysisMode.FULL_ANALYSIS:
            return self.prompt_builder.full_analysis(
                forecast, kpcofgs, kpcofgs_regime,
                validation, backtest, opportunities,
            )
        elif mode == AnalysisMode.REGIME_DEEP_DIVE:
            return self.prompt_builder.regime_deep_dive(
                forecast, kpcofgs, kpcofgs_regime,
            )
        elif mode == AnalysisMode.TRADE_PLAN:
            return self.prompt_builder.trade_plan(
                forecast, kpcofgs, kpcofgs_regime,
                backtest,
                self.config.risk_budget_usd,
                self.config.max_leverage,
            )
        elif mode == AnalysisMode.RISK_ASSESSMENT:
            return self.prompt_builder.risk_assessment(
                forecast, kpcofgs, backtest,
            )
        elif mode == AnalysisMode.EXTRAPOLATION:
            return self.prompt_builder.extrapolation(
                forecast, kpcofgs, kpcofgs_regime, validation,
            )
        elif mode == AnalysisMode.SELF_CRITIQUE:
            return self.prompt_builder.self_critique(
                forecast, kpcofgs, validation, backtest,
            )
        else:
            return self.prompt_builder.full_analysis(
                forecast, kpcofgs, kpcofgs_regime,
                validation, backtest, opportunities,
            )

    def _run_sub_analysis(
        self,
        mode: AnalysisMode,
        forecast: Dict,
        kpcofgs: Dict,
        kpcofgs_regime: str,
        validation: Dict = None,
        backtest: Dict = None,
    ) -> Optional[Dict]:
        """Run a sub-analysis and return just the parsed result."""
        try:
            system_prompt, user_prompt = self._build_prompt(
                mode, forecast, kpcofgs, kpcofgs_regime,
                validation, backtest,
            )
            response = self.client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format="json",
            )
            return response.parsed
        except Exception:
            return None
