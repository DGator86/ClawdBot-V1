#!/usr/bin/env python3
"""
Kalshi Order Placement Helper
==============================
Places orders on Kalshi using the V2 API.
Designed to be called by ClawdBot after user approves a pick.

Usage:
  python3 scripts/kalshi-order.py --ticker KXBTC-26FEB06-T64000 --side yes --count 5
  python3 scripts/kalshi-order.py --ticker KXBTC-26FEB06-T64000 --side no --count 3 --type limit --price 45
  python3 scripts/kalshi-order.py --cancel ORDER_ID
  python3 scripts/kalshi-order.py --positions    # show current positions
  python3 scripts/kalshi-order.py --orders       # show open orders
  python3 scripts/kalshi-order.py --balance      # show account balance

Env vars: KALSHI_KEY_ID, KALSHI_PRIVATE_KEY
Standalone — no Yoshi-Bot dependency. Requires: cryptography (pip3 install cryptography)
"""

import argparse
import json
import logging
import os
import sys

from kalshi_client import KalshiClient


# Keys that must be overwritten (systemd EnvironmentFile mangles multi-line PEM)
_FORCE_OVERWRITE_KEYS = {"KALSHI_PRIVATE_KEY"}


def load_env():
    """Source .env files for Kalshi credentials.
    
    KALSHI_PRIVATE_KEY is force-overwritten because systemd's
    EnvironmentFile truncates multi-line values to one line.
    """
    for env_path in [
        "/root/Yoshi-Bot/.env",
        "/root/ClawdBot-V1/.env",
        "/home/root/Yoshi-Bot/.env",
        os.path.expanduser("~/.env"),
    ]:
        if os.path.isfile(env_path):
            try:
                with open(env_path) as f:
                    for raw in f:
                        raw = raw.strip()
                        if not raw or raw.startswith("#") or "=" not in raw:
                            continue
                        key, _, val = raw.partition("=")
                        key = key.strip()
                        val = val.strip()
                        # Handle multi-line PEM values
                        if val.startswith('"') and not val.endswith('"'):
                            lines = [val[1:]]
                            for extra in f:
                                extra = extra.rstrip("\n")
                                if extra.endswith('"'):
                                    lines.append(extra[:-1])
                                    break
                                lines.append(extra)
                            val = "\n".join(lines)
                        else:
                            val = val.strip('"').strip("'")
                        if key and val:
                            if key in _FORCE_OVERWRITE_KEYS:
                                os.environ[key] = val
                            else:
                                os.environ.setdefault(key, val)
            except Exception as e:
                # Don't swallow env-loading errors silently
                logging.warning(f"Failed to load .env from {env_path}: {e}")
                import traceback
                logging.debug(traceback.format_exc())


def place_order(client, ticker, side, count, order_type="market", price=None):
    path = "/portfolio/orders"
    payload = {
        "ticker": ticker,
        "side": side,
        "action": "buy",
        "type": order_type,
        "count": count,
    }
    if order_type == "limit" and price is not None:
        if side == "yes":
            payload["yes_price"] = price
        else:
            payload["no_price"] = price
    body = json.dumps(payload)
    return client._request("POST", path, body)


def main():
    parser = argparse.ArgumentParser(description="Kalshi Order Placement")
    parser.add_argument("--ticker", type=str, help="Kalshi contract ticker")
    parser.add_argument("--side", type=str, choices=["yes", "no"], help="Side to buy")
    parser.add_argument("--count", type=int, default=1, help="Number of contracts")
    parser.add_argument("--type", type=str, default="market", choices=["market", "limit"])
    parser.add_argument("--price", type=int, help="Limit price in cents (1-99)")
    parser.add_argument("--cancel", type=str, help="Cancel order by ID")
    parser.add_argument("--positions", action="store_true", help="Show positions")
    parser.add_argument("--orders", action="store_true", help="Show open orders")
    parser.add_argument("--balance", action="store_true", help="Show balance")
    args = parser.parse_args()

    load_env()

    try:
        client = KalshiClient()
        print(f"Connected (key: {client.key_id[:12]}...)")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.positions:
        result = client.get_positions(limit=100)
        print(json.dumps(result, indent=2))
    elif args.orders:
        result = client.list_orders(status="resting")
        print(json.dumps(result, indent=2))
    elif args.balance:
        result = client.get_balance()
        print(json.dumps(result, indent=2))
    elif args.cancel:
        result = client.cancel_order(args.cancel)
        print(json.dumps(result, indent=2))
    elif args.ticker and args.side:
        print(f"Placing order: {args.count}x {args.side.upper()} on {args.ticker} ({args.type})")
        result = place_order(client, args.ticker, args.side, args.count,
                             args.type, args.price)
        print(json.dumps(result, indent=2))
        if isinstance(result, dict) and "order" in result:
            order = result["order"]
            print("\n Order placed!")
            print(f"   Order ID: {order.get('order_id')}")
            print(f"   Status:   {order.get('status')}")
        elif isinstance(result, dict) and ("error" in result or "code" in result):
            print(f"\n Order failed: {result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
