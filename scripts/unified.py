#!/usr/bin/env python3
"""
Unified CLI — ClawdBot + Yoshi Pipeline
=========================================
Runs the full integrated pipeline:
  forecast → KPCOFGS regime → walk-forward validation → backtest → Kalshi scan

Usage:
    # Quick forecast with KPCOFGS enrichment
    python3 -m scripts.unified --symbol BTCUSDT --horizon 24

    # Full pipeline: forecast + walk-forward + backtest
    python3 -m scripts.unified --mode full --bars 2000

    # Validate only: walk-forward with purge/embargo
    python3 -m scripts.unified --mode validate --bars 2000

    # Backtest only
    python3 -m scripts.unified --mode backtest --bars 2000

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
        description="Unified ClawdBot + Yoshi Pipeline"
    )
    parser.add_argument("-s", "--symbol", default="BTCUSDT",
                        help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("-H", "--horizon", type=float, default=24.0,
                        help="Forecast horizon in hours (default: 24)")
    parser.add_argument("-b", "--bars", type=int, default=2000,
                        help="Number of bars to fetch (default: 2000)")
    parser.add_argument("-m", "--mode", default="forecast",
                        choices=["forecast", "validate", "backtest", "full"],
                        help="Pipeline mode (default: forecast)")
    parser.add_argument("-n", "--mc-iterations", type=int, default=50_000,
                        help="Monte Carlo iterations (default: 50000)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of human-readable")
    parser.add_argument("-o", "--output", default=None,
                        help="Save results to file")
    parser.add_argument("-q", "--quiet", action="store_true",
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
