"""
LLM Reasoning Layer — The Brain of the Twin-Bot System
========================================================
Sits atop ClawdBot (forecasting) and Yoshi (validation/execution),
consuming all pipeline data and producing structured analysis,
extrapolations, and trade suggestions via an LLM API.

Architecture:
  ClawdBot forecast (14-paradigm ensemble, MC, regime gate)
  + Yoshi KPCOFGS regime (7-level classification)
  + Yoshi walk-forward validation (scoring, purge/embargo)
  + Yoshi backtest (PnL, Sharpe, drawdown)
  → LLM Reasoning Layer (this package)
    → Forecast interpretation & narrative
    → Regime analysis & extrapolation
    → Trade suggestions with risk parameters
    → Risk assessment & position sizing
    → Self-critique & confidence calibration
"""

from .engine import (
    ReasoningEngine,
    ReasoningConfig,
    ReasoningResult,
    AnalysisMode,
)
from .prompts import PromptBuilder
from .client import LLMClient, LLMConfig

__all__ = [
    "ReasoningEngine",
    "ReasoningConfig",
    "ReasoningResult",
    "AnalysisMode",
    "PromptBuilder",
    "LLMClient",
    "LLMConfig",
]
