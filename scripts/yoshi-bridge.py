#!/usr/bin/env python3
"""Yoshi-Bridge: Polls Yoshi-Bot scanner logs and forwards signals to the
Trading Core /propose endpoint, making them available for ClawdBot to read
and present as Kalshi trade suggestions via Telegram.

Architecture:
  Yoshi scanner (logs) -> yoshi-bridge.py -> Trading Core /propose -> ClawdBot reads /status

Usage:
  python3 scripts/yoshi-bridge.py
  python3 scripts/yoshi-bridge.py --log-path /home/root/Yoshi-Bot/logs/scanner.log
  python3 scripts/yoshi-bridge.py --poll-interval 30
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import request, error


TRADING_CORE_URL = os.getenv("TRADING_CORE_URL", "http://127.0.0.1:8000")

# Patterns to extract signal data from scanner log lines
EDGE_PATTERN = re.compile(
    r"EDGE:\s*([+-]?\d+\.?\d*)%", re.IGNORECASE
)
SYMBOL_PATTERN = re.compile(
    r"\*?(BTC|ETH|SOL)USDT\*?", re.IGNORECASE
)
STRIKE_PATTERN = re.compile(
    r"Strike:.*?(?:Above|Below)?\s*\$?([\d,]+\.?\d*)", re.IGNORECASE
)
MODEL_PROB_PATTERN = re.compile(
    r"Model\s*Prob:\s*(\d+\.?\d*)%", re.IGNORECASE
)
MARKET_PROB_PATTERN = re.compile(
    r"Market\s*Prob:\s*(\d+\.?\d*)%", re.IGNORECASE
)
ACTION_PATTERN = re.compile(
    r"ACTION:\s*`?(BUY\s*(?:YES|NO)|NEUTRAL)`?", re.IGNORECASE
)
PRICE_PATTERN = re.compile(
    r"Price:\s*\$?([\d,]+\.?\d*)", re.IGNORECASE
)
FORECAST_PATTERN = re.compile(
    r"Forecast:\s*\$?([\d,]+\.?\d*)", re.IGNORECASE
)


def parse_number(s: str) -> float:
    """Parse a number string, removing commas."""
    return float(s.replace(",", ""))


def check_trading_core() -> bool:
    """Check if the Trading Core API is reachable."""
    try:
        req = request.Request(f"{TRADING_CORE_URL}/health")
        with request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "healthy"
    except Exception:
        return False


def get_trading_status() -> dict:
    """Get current trading status from Trading Core."""
    try:
        req = request.Request(f"{TRADING_CORE_URL}/status")
        with request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {}


def propose_trade(signal: dict) -> dict:
    """Send a trade proposal to the Trading Core API."""
    symbol = signal.get("symbol", "BTCUSDT")
    edge = signal.get("edge", 0)
    side = "buy" if edge > 0 else "sell"

    payload = {
        "exchange": "kalshi",
        "symbol": symbol,
        "side": side,
        "type": "market",
        "amount": signal.get("amount", 0.01),
    }

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{TRADING_CORE_URL}/propose",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode()}
    except Exception as e:
        return {"error": str(e)}


def parse_signal_block(block: str) -> dict | None:
    """Parse a YOSHI KALSHI ALERT block from scanner log output."""
    edge_match = EDGE_PATTERN.search(block)
    if not edge_match:
        return None

    symbol_match = SYMBOL_PATTERN.search(block)
    strike_match = STRIKE_PATTERN.search(block)
    model_prob_match = MODEL_PROB_PATTERN.search(block)
    market_prob_match = MARKET_PROB_PATTERN.search(block)
    action_match = ACTION_PATTERN.search(block)
    price_match = PRICE_PATTERN.search(block)
    forecast_match = FORECAST_PATTERN.search(block)

    signal = {
        "edge": float(edge_match.group(1)),
        "symbol": f"{symbol_match.group(1).upper()}USDT" if symbol_match else "BTCUSDT",
        "timestamp": datetime.utcnow().isoformat(),
    }

    if strike_match:
        signal["strike"] = parse_number(strike_match.group(1))
    if model_prob_match:
        signal["model_prob"] = float(model_prob_match.group(1)) / 100
    if market_prob_match:
        signal["market_prob"] = float(market_prob_match.group(1)) / 100
    if action_match:
        signal["action"] = action_match.group(1).strip().upper()
    if price_match:
        signal["current_price"] = parse_number(price_match.group(1))
    if forecast_match:
        signal["forecast_price"] = parse_number(forecast_match.group(1))

    return signal


def tail_log(log_path: str, last_position: int) -> tuple[str, int]:
    """Read new content from the scanner log since last_position."""
    try:
        path = Path(log_path)
        if not path.exists():
            return "", last_position

        size = path.stat().st_size
        if size < last_position:
            # Log was rotated or truncated
            last_position = 0

        if size == last_position:
            return "", last_position

        with open(path, "r") as f:
            f.seek(last_position)
            new_content = f.read()
            new_position = f.tell()

        return new_content, new_position
    except Exception as e:
        print(f"[{datetime.now()}] Error reading log: {e}")
        return "", last_position


def find_scanner_log() -> str:
    """Find the Yoshi-Bot scanner log on the VPS."""
    candidates = [
        "/home/root/Yoshi-Bot/logs/scanner.log",
        "/root/Yoshi-Bot/logs/scanner.log",
        os.path.expanduser("~/Yoshi-Bot/logs/scanner.log"),
        "./logs/scanner.log",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return candidates[0]  # Default, will be created when scanner runs


def main():
    parser = argparse.ArgumentParser(description="Yoshi-Bridge: Scanner -> Trading Core")
    parser.add_argument(
        "--log-path", type=str, default=None,
        help="Path to Yoshi scanner log (auto-detected if not set)"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=30,
        help="Seconds between log polls (default: 30)"
    )
    parser.add_argument(
        "--min-edge", type=float, default=5.0,
        help="Minimum edge %% to forward to Trading Core (default: 5.0)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and print signals without sending to Trading Core"
    )
    args = parser.parse_args()

    log_path = args.log_path or find_scanner_log()
    last_position = 0
    proposals_sent = 0
    cooldown_map: dict[str, float] = {}
    cooldown_seconds = 3600  # 1 hour per symbol/strike combo

    print(f"[{datetime.now()}] Yoshi-Bridge starting")
    print(f"  Scanner log: {log_path}")
    print(f"  Trading Core: {TRADING_CORE_URL}")
    print(f"  Poll interval: {args.poll_interval}s")
    print(f"  Min edge: {args.min_edge}%")
    print(f"  Dry run: {args.dry_run}")

    # Check Trading Core connectivity
    if check_trading_core():
        print(f"  Trading Core: CONNECTED")
        status = get_trading_status()
        print(f"  Trading paused: {status.get('is_paused', 'unknown')}")
        print(f"  Kill switch: {status.get('kill_switch_active', 'unknown')}")
        print(f"  Open positions: {status.get('positions_count', 'unknown')}")
    else:
        print(f"  Trading Core: NOT REACHABLE (will retry)")

    # If log file exists, skip to end (only process new signals)
    if Path(log_path).exists():
        last_position = Path(log_path).stat().st_size
        print(f"  Skipping existing log ({last_position} bytes)")

    print(f"[{datetime.now()}] Watching for signals...\n")

    while True:
        try:
            new_content, last_position = tail_log(log_path, last_position)

            if new_content:
                # Split on alert boundaries
                blocks = re.split(r"={10,}", new_content)

                for block in blocks:
                    if "KALSHI ALERT" not in block.upper() and "EDGE" not in block.upper():
                        continue

                    signal = parse_signal_block(block)
                    if not signal:
                        continue

                    edge = abs(signal.get("edge", 0))
                    if edge < args.min_edge:
                        print(f"[{datetime.now()}] Signal edge {signal['edge']}% below threshold {args.min_edge}%, skipping")
                        continue

                    # Check cooldown
                    cooldown_key = f"{signal['symbol']}:{signal.get('strike', 'any')}"
                    last_sent = cooldown_map.get(cooldown_key, 0)
                    if time.time() - last_sent < cooldown_seconds:
                        print(f"[{datetime.now()}] Signal for {cooldown_key} in cooldown, skipping")
                        continue

                    print(f"[{datetime.now()}] SIGNAL DETECTED:")
                    print(f"  Symbol: {signal['symbol']}")
                    print(f"  Edge: {signal['edge']}%")
                    print(f"  Action: {signal.get('action', 'N/A')}")
                    print(f"  Strike: {signal.get('strike', 'N/A')}")

                    if args.dry_run:
                        print(f"  [DRY RUN] Would propose to Trading Core")
                    else:
                        # Check system status before proposing
                        status = get_trading_status()
                        if status.get("is_paused"):
                            print(f"  Trading is PAUSED, skipping proposal")
                            continue
                        if status.get("kill_switch_active"):
                            print(f"  Kill switch ACTIVE, skipping proposal")
                            continue

                        result = propose_trade(signal)
                        print(f"  Proposal result: {json.dumps(result)}")
                        proposals_sent += 1
                        cooldown_map[cooldown_key] = time.time()

                    print()

        except KeyboardInterrupt:
            print(f"\n[{datetime.now()}] Yoshi-Bridge stopped. Proposals sent: {proposals_sent}")
            sys.exit(0)
        except Exception as e:
            print(f"[{datetime.now()}] Error: {e}")

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
