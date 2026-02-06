#!/usr/bin/env python3
"""
Kalshi Order Placement Helper
==============================
Places orders on Kalshi using the V2 API.
Designed to be called by ClawdBot after user approves a pick.

Usage:
  python3 scripts/kalshi-order.py --ticker KXBTC-26FEB06-T64000 --side yes --count 5
  python3 scripts/kalshi-order.py --ticker KXBTC-26FEB06-T64000 --side no --count 3 --limit 45
  python3 scripts/kalshi-order.py --cancel ORDER_ID
  python3 scripts/kalshi-order.py --positions    # show current positions
  python3 scripts/kalshi-order.py --orders       # show open orders
  python3 scripts/kalshi-order.py --balance      # show account balance

Env vars: KALSHI_KEY_ID, KALSHI_PRIVATE_KEY (from Yoshi-Bot .env)
"""

import argparse
import base64
import json
import os
import sys
import time
import importlib.util
from pathlib import Path


def load_env():
    """Source Yoshi-Bot .env for Kalshi credentials."""
    for env_path in ["/root/Yoshi-Bot/.env", "/home/root/Yoshi-Bot/.env"]:
        if os.path.isfile(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and val:
                        os.environ.setdefault(key, val)
            return True
    return False


def load_client():
    """Load KalshiClient from Yoshi-Bot."""
    for yoshi_dir in ["/root/Yoshi-Bot", "/home/root/Yoshi-Bot"]:
        path = os.path.join(yoshi_dir, "src", "gnosis", "utils", "kalshi_client.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("kalshi_client", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.KalshiClient()
    raise ImportError("KalshiClient not found")


def make_request(client, method: str, path: str, body: str = "") -> dict:
    """Make an authenticated request to Kalshi API."""
    import requests
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    timestamp = str(int(time.time() * 1000))
    message = timestamp + method + path + body
    signature = client.private_key.sign(
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )

    headers = {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": client.key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }

    url = client.BASE_URL + path
    if method == "GET":
        resp = requests.get(url, headers=headers, timeout=10)
    elif method == "POST":
        resp = requests.post(url, headers=headers, data=body, timeout=10)
    elif method == "DELETE":
        resp = requests.delete(url, headers=headers, timeout=10)
    else:
        raise ValueError(f"Unknown method: {method}")

    return resp.json()


def place_order(client, ticker: str, side: str, count: int,
                order_type: str = "market", price: int = None) -> dict:
    """Place an order on Kalshi."""
    path = "/portfolio/orders"
    payload = {
        "ticker": ticker,
        "side": side,      # "yes" or "no"
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
    result = make_request(client, "POST", path, body)
    return result


def get_positions(client) -> dict:
    """Get current portfolio positions."""
    return make_request(client, "GET", "/portfolio/positions?limit=100")


def get_orders(client) -> dict:
    """Get open orders."""
    return make_request(client, "GET", "/portfolio/orders?status=resting")


def get_balance(client) -> dict:
    """Get account balance."""
    return make_request(client, "GET", "/portfolio/balance")


def cancel_order(client, order_id: str) -> dict:
    """Cancel an open order."""
    return make_request(client, "DELETE", f"/portfolio/orders/{order_id}")


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
        client = load_client()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.positions:
        result = get_positions(client)
        print(json.dumps(result, indent=2))
    elif args.orders:
        result = get_orders(client)
        print(json.dumps(result, indent=2))
    elif args.balance:
        result = get_balance(client)
        print(json.dumps(result, indent=2))
    elif args.cancel:
        result = cancel_order(client, args.cancel)
        print(json.dumps(result, indent=2))
    elif args.ticker and args.side:
        print(f"Placing order: {args.count}x {args.side.upper()} on {args.ticker} ({args.type})")
        result = place_order(client, args.ticker, args.side, args.count,
                             args.type, args.price)
        print(json.dumps(result, indent=2))

        if "order" in result:
            order = result["order"]
            print(f"\n✅ Order placed!")
            print(f"   Order ID: {order.get('order_id')}")
            print(f"   Status:   {order.get('status')}")
            print(f"   Ticker:   {order.get('ticker')}")
            print(f"   Side:     {order.get('side')}")
            print(f"   Count:    {order.get('initial_count')}")
        elif "error" in result or "code" in result:
            print(f"\n❌ Order failed: {result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
