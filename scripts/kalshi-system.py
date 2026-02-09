#!/usr/bin/env python3
"""
Kalshi Trading System — Single-Command Entry Point.
=====================================================
Runs the full ClawdBot × Yoshi × Kalshi × Ralph Wiggum pipeline.

    One command to rule them all:
        python3 scripts/kalshi-system.py

    What it does:
        1. ClawdBot forecasts BTC/ETH with 14-paradigm ensemble
        2. Yoshi enriches with KPCOFGS 7-level regime classification
        3. Kalshi scanner finds edge on binary option markets
        4. LLM analyzer filters real edge from noise (free tier)
        5. Ralph Wiggum records, resolves, learns, and optimizes
        6. Repeat every 60 seconds (configurable)

    Options:
        --once          Run a single cycle and exit
        --interval N    Seconds between cycles (default: 60)
        --cycles N      Max cycles (default: unlimited)
        --no-kalshi     Skip Kalshi scanning
        --no-ralph      Skip Ralph learning
        --no-forecast   Skip forecast (Kalshi-only mode)
        --explore N     Exploration rate (default: 0.10)
        --verbose       Extra output (default: on)
        --quiet         Minimal output
        --status        Show system status and exit
"""
import argparse
import json
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(
        description="ClawdBot × Yoshi × Kalshi × Ralph Wiggum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--once", action="store_true",
                    help="Run a single cycle and exit")
    p.add_argument("--interval", type=float, default=60,
                    help="Seconds between cycles (default: 60)")
    p.add_argument("--cycles", type=int, default=0,
                    help="Max cycles (0 = unlimited)")
    p.add_argument("--no-kalshi", action="store_true",
                    help="Skip Kalshi scanning")
    p.add_argument("--no-ralph", action="store_true",
                    help="Skip Ralph learning")
    p.add_argument("--no-forecast", action="store_true",
                    help="Skip forecast (Kalshi-only mode)")
    p.add_argument("--explore", type=float, default=0.10,
                    help="Ralph exploration rate (default: 0.10)")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--status", action="store_true",
                    help="Show system status and exit")
    p.add_argument("--json", action="store_true",
                    help="Output results as JSON")
    p.add_argument("--series", type=str, default="KXBTC,KXETH",
                    help="Kalshi series (comma-separated, default: KXBTC,KXETH)")
    p.add_argument("--horizon", type=float, default=1.0,
                    help="Forecast horizon in hours (default: 1.0)")
    return p.parse_args()


def show_status():
    """Show system status and exit."""
    from gnosis.ralph.learner import RalphLearner
    from gnosis.reasoning.client import LLMConfig

    print("=" * 55)
    print("  SYSTEM STATUS")
    print("=" * 55)

    # LLM routing
    try:
        cfg = LLMConfig.from_yaml()
        print(f"\n  LLM:        {cfg._environment} ({cfg.model})")
        print(f"  Base URL:   {cfg.base_url}")
    except Exception as e:
        print(f"\n  LLM:        ERROR ({e})")

    # Ralph state
    try:
        learner = RalphLearner()
        summary = learner.get_learning_summary()
        tracker = summary["tracker"]
        params = summary["params"]
        metrics = summary["metrics"]

        print(f"\n  Ralph:")
        print(f"    Cycle:        {params.get('cycle', 0)}")
        print(f"    Mode:         {'EXPLORE' if params.get('is_exploring') else 'EXPLOIT'}")
        print(f"    Predictions:  {tracker.get('total', 0)} total, {tracker.get('resolved', 0)} resolved")
        print(f"    Best score:   {params.get('best_score', 0):.4f}")
        print(f"    History:      {params.get('history_size', 0)} snapshots")

        if metrics.get("n_resolved", 0) > 0:
            print(f"\n  Performance:")
            print(f"    Brier:        {metrics.get('brier_score', '?')}")
            print(f"    Hit Rate:     {metrics.get('hit_rate', '?')}")
            print(f"    PnL (cents):  {metrics.get('total_pnl_cents', 0):+.0f}")
    except Exception as e:
        print(f"\n  Ralph:      No data yet ({e})")

    # Kalshi connectivity
    try:
        from gnosis.kalshi.scanner import KalshiAPIClient, _load_env_files
        _load_env_files()
        client = KalshiAPIClient()
        status = client.get_exchange_status()
        active = status.get("exchange_active", False) if status else False
        print(f"\n  Kalshi:     {'CONNECTED' if active else 'OFFLINE'}")
        if status:
            print(f"    Exchange: {'OPEN' if active else 'CLOSED'}")
            print(f"    Trading:  {'ACTIVE' if status.get('trading_active') else 'INACTIVE'}")
    except Exception as e:
        print(f"\n  Kalshi:     NOT CONFIGURED ({e})")

    print(f"\n{'='*55}")


def main():
    args = parse_args()

    if args.status:
        show_status()
        return

    verbose = not args.quiet

    from gnosis.orchestrator import UnifiedOrchestrator, OrchestratorConfig
    from gnosis.ralph.learner import LearningConfig

    # Build config from CLI args
    learning_cfg = LearningConfig(
        explore_rate=args.explore,
        verbose=verbose,
    )

    series = [s.strip() for s in args.series.split(",") if s.strip()]

    config = OrchestratorConfig(
        kalshi_series=series,
        horizon_hours=args.horizon,
        learning=learning_cfg,
        cycle_interval_s=args.interval,
        max_cycles=1 if args.once else args.cycles,
        enable_forecast=not args.no_forecast,
        enable_kalshi=not args.no_kalshi,
        enable_ralph=not args.no_ralph,
        verbose=verbose,
    )

    # Create orchestrator
    orch = UnifiedOrchestrator(config=config)

    if args.once:
        # Single cycle
        result = orch.run_cycle()
        if args.json:
            print(result.to_json())
        else:
            result.print_report()
    else:
        # Continuous loop
        def on_cycle(result):
            if args.json:
                print(result.to_json())

        orch.run_loop(
            max_cycles=config.max_cycles or None,
            interval_s=config.cycle_interval_s,
            on_cycle=on_cycle if args.json else None,
        )


if __name__ == "__main__":
    main()
