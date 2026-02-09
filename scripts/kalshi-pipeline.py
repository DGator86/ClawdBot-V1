#!/usr/bin/env python3
"""
Kalshi Trading Pipeline CLI — Single-command market scanner.
=============================================================
Scans Kalshi binary markets, scores by edge/EV, runs LLM analysis,
and prints a phone-friendly report.

One command to run from VPS:
  python3 scripts/kalshi-pipeline.py

Options:
  python3 scripts/kalshi-pipeline.py --series KXBTC        # BTC only
  python3 scripts/kalshi-pipeline.py --top 10               # More results
  python3 scripts/kalshi-pipeline.py --min-edge 5           # Stricter filter
  python3 scripts/kalshi-pipeline.py --json                 # JSON output
  python3 scripts/kalshi-pipeline.py --json -o report.json  # Save to file
  python3 scripts/kalshi-pipeline.py --loop --interval 120  # Continuous

Env vars: KALSHI_KEY_ID, KALSHI_PRIVATE_KEY, OPENAI_API_KEY (optional for LLM)
"""
import argparse
import json
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(
        description="Kalshi Trading Pipeline — Scan, Analyze, Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/kalshi-pipeline.py                    # Quick scan
  python3 scripts/kalshi-pipeline.py --series KXBTC     # BTC markets only
  python3 scripts/kalshi-pipeline.py --top 10           # Top 10 picks
  python3 scripts/kalshi-pipeline.py --json             # JSON output
  python3 scripts/kalshi-pipeline.py --loop --interval 60   # Continuous
        """,
    )
    parser.add_argument(
        "--series", nargs="+", default=["KXBTC", "KXETH"],
        help="Kalshi series to scan (default: KXBTC KXETH)",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of top picks (default: 5)",
    )
    parser.add_argument(
        "--min-edge", type=float, default=3.0,
        help="Minimum edge %% to consider (default: 3.0)",
    )
    parser.add_argument(
        "--min-ev", type=float, default=1.0,
        help="Minimum EV in cents (default: 1.0)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "-o", "--output", type=str,
        help="Save output to file",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Run continuously",
    )
    parser.add_argument(
        "--interval", type=int, default=120,
        help="Seconds between scans in loop mode (default: 120)",
    )
    parser.add_argument(
        "--no-balance", action="store_true",
        help="Skip balance/position check",
    )
    args = parser.parse_args()

    from gnosis.kalshi.pipeline import KalshiPipeline

    pipeline = KalshiPipeline(
        series=args.series,
        top_n=args.top,
        min_edge_pct=args.min_edge,
        min_ev_cents=args.min_ev,
        show_balance=not args.no_balance,
        show_positions=not args.no_balance,
    )

    cycle = 0
    while True:
        cycle += 1
        if args.loop and cycle > 1:
            print(f"\n--- Cycle #{cycle} ---")

        print("=" * 50)
        print(f"  KALSHI PIPELINE: {', '.join(args.series)}")
        print("=" * 50)

        result = pipeline.run()

        if args.json:
            output = result.to_json()
            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
                print(f"\nSaved to {args.output}")
            else:
                print(output)
        else:
            result.print_report()

        if args.output and not args.json:
            with open(args.output, "w") as f:
                result.print_report(file=f)
            print(f"\nSaved to {args.output}")

        if not args.loop:
            break

        print(f"\nNext scan in {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
