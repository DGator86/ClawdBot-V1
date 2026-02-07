#!/usr/bin/env python3
"""
Unified CLI -- ClawdBot + Yoshi + LLM Reasoning Pipeline
==========================================================
Runs the full integrated pipeline:
  forecast -> KPCOFGS regime -> walk-forward validation -> backtest
  -> LLM reasoning -> actionable output

The LLM layer reads all pipeline data and produces:
  - Forecast interpretation & narrative
  - Regime analysis & extrapolation
  - Trade suggestions with risk parameters
  - Self-critique & confidence calibration

Usage:
    # Quick forecast + LLM analysis (default)
    python3 -m scripts.unified --symbol BTCUSDT --horizon 24

    # Full pipeline: forecast + walk-forward + backtest + LLM
    python3 -m scripts.unified --mode full --bars 2000

    # Forecast + specific LLM reasoning mode
    python3 -m scripts.unified --reasoning trade --risk-budget 1000

    # Deep regime analysis
    python3 -m scripts.unified --reasoning regime

    # Risk assessment with custom leverage
    python3 -m scripts.unified --reasoning risk --max-leverage 3

    # Forward-looking extrapolation
    python3 -m scripts.unified --reasoning extrapolation

    # Self-critique of forecast
    python3 -m scripts.unified --reasoning critique

    # Skip LLM reasoning (just forecast + KPCOFGS)
    python3 -m scripts.unified --reasoning off

    # JSON output
    python3 -m scripts.unified --mode full --json --output results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Unified ClawdBot + Yoshi + LLM Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Reasoning Modes:
  auto           Engine picks best mode based on data (default)
  full           Complete analysis: forecast + regime + trade + risk
  regime         Deep KPCOFGS 7-level regime analysis
  trade          Specific trade plan with entry/exit/sizing
  risk           Risk assessment, position sizing, hedging
  extrapolation  Forward-looking scenarios and predictions
  critique       Self-critique of forecast assumptions
  off            Skip LLM reasoning

Pipeline Modes:
  forecast       Quick: ClawdBot 14-paradigm + KPCOFGS + LLM (default)
  reason         Same as forecast but always includes LLM
  validate       + walk-forward validation with purge/embargo
  backtest       + backtest with PnL/Sharpe/drawdown
  full           All of the above

Examples:
  # Standard forecast with LLM interpretation
  python3 -m scripts.unified -s BTCUSDT -H 24

  # Full pipeline with trade plan and $1000 budget
  python3 -m scripts.unified --mode full --reasoning trade --risk-budget 1000

  # Risk analysis on 2000 bars
  python3 -m scripts.unified --reasoning risk --bars 2000

  # Save full analysis to file
  python3 -m scripts.unified --mode full --json -o analysis.json
""",
    )
    parser.add_argument("-s", "--symbol", default="BTCUSDT",
                        help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("-H", "--horizon", type=float, default=24.0,
                        help="Forecast horizon in hours (default: 24)")
    parser.add_argument("-b", "--bars", type=int, default=2000,
                        help="Number of bars to fetch (default: 2000)")
    parser.add_argument("-m", "--mode", default="forecast",
                        choices=["forecast", "reason", "validate", "backtest", "full"],
                        help="Pipeline mode (default: forecast)")
    parser.add_argument("-n", "--mc-iterations", type=int, default=50_000,
                        help="Monte Carlo iterations (default: 50000)")

    # LLM Reasoning options
    reasoning_group = parser.add_argument_group("LLM Reasoning")
    reasoning_group.add_argument("-r", "--reasoning", default="auto",
                                  choices=["auto", "full", "regime", "trade",
                                           "risk", "extrapolation", "critique", "off"],
                                  help="LLM reasoning mode (default: auto)")
    reasoning_group.add_argument("--risk-budget", type=float, default=500.0,
                                  help="Risk budget in USD for trade plans (default: 500)")
    reasoning_group.add_argument("--max-leverage", type=float, default=2.0,
                                  help="Max leverage for trade plans (default: 2.0)")

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--json", action="store_true",
                               help="Output JSON instead of human-readable")
    output_group.add_argument("-o", "--output", default=None,
                               help="Save results to file")
    output_group.add_argument("-q", "--quiet", action="store_true",
                               help="Suppress verbose output")

    args = parser.parse_args()

    from gnosis.bridge import run_unified

    result = run_unified(
        symbol=args.symbol,
        horizon_hours=args.horizon,
        bars_limit=args.bars,
        mode=args.mode,
        mc_iterations=args.mc_iterations,
        verbose=not args.quiet,
        reasoning_mode=args.reasoning,
        risk_budget_usd=args.risk_budget,
        max_leverage=args.max_leverage,
    )

    if args.json or args.output:
        output = result.to_json(indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"\nResults saved to {args.output}")
        else:
            print(output)


if __name__ == "__main__":
    main()
