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
import base64
import json
import os
import sys
import time
from urllib import request, parse as urlparse


# Keys that must be overwritten (systemd EnvironmentFile mangles multi-line PEM)
_FORCE_OVERWRITE_KEYS = {"KALSHI_PRIVATE_KEY"}

# Try to use shared utilities
try:
    from scripts.lib.pem_utils import fix_pem as _shared_fix_pem, load_env_files as _shared_load_env_files
    _HAS_SHARED_UTILS = True
except ImportError:
    _HAS_SHARED_UTILS = False


def load_env():
    """Source .env files for Kalshi credentials.
    
    KALSHI_PRIVATE_KEY is force-overwritten because systemd's
    EnvironmentFile truncates multi-line values to one line.
    Uses shared pem_utils when available.
    """
    if _HAS_SHARED_UTILS:
        _shared_load_env_files()
        return

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
            except Exception:
                pass


class KalshiClient:
    """Standalone Kalshi V2 API client with RSA-PSS SHA-256 auth."""
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self):
        self.key_id = os.environ.get("KALSHI_KEY_ID", "").strip()
        pk_raw = os.environ.get("KALSHI_PRIVATE_KEY", "").strip()
        if not self.key_id:
            raise ValueError("KALSHI_KEY_ID not set")
        if not pk_raw:
            for pk_path in [
                os.path.expanduser("~/.kalshi/private_key.pem"),
                "/root/.kalshi/private_key.pem",
            ]:
                if os.path.isfile(pk_path):
                    with open(pk_path) as f:
                        pk_raw = f.read().strip()
                    break
        if not pk_raw:
            raise ValueError("KALSHI_PRIVATE_KEY not set and no PEM file found")

        # Fix PEM formatting — env vars often have literal \n instead of newlines
        pk_raw = self._fix_pem(pk_raw)

        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        self.private_key = load_pem_private_key(pk_raw.encode(), password=None)

    @staticmethod
    def _fix_pem(raw: str) -> str:
        """Normalize a PEM key. Delegates to shared pem_utils when available."""
        if _HAS_SHARED_UTILS:
            return _shared_fix_pem(raw)
        import re
        if "\\n" in raw:
            raw = raw.replace("\\n", "\n")
        if "-----BEGIN" in raw and raw.count("\n") <= 2:
            m = re.search(r"-----BEGIN [A-Z ]+-----\s*(.*?)\s*-----END [A-Z ]+-----", raw, re.DOTALL)
            if m:
                header_match = re.search(r"(-----BEGIN [A-Z ]+-----)", raw)
                footer_match = re.search(r"(-----END [A-Z ]+-----)", raw)
                if header_match and footer_match:
                    body = m.group(1).replace(" ", "").replace("\n", "").replace("\r", "")
                    lines = [body[i:i+64] for i in range(0, len(body), 64)]
                    raw = header_match.group(1) + "\n" + "\n".join(lines) + "\n" + footer_match.group(1)
            return raw.strip()
        if "-----BEGIN" not in raw:
            body = re.sub(r"\s+", "", raw)
            if len(body) > 100 and re.match(r"^[A-Za-z0-9+/=]+$", body):
                lines = [body[i:i+64] for i in range(0, len(body), 64)]
                raw = "-----BEGIN RSA PRIVATE KEY-----\n" + "\n".join(lines) + "\n-----END RSA PRIVATE KEY-----"
        return raw.strip()

    def _sign(self, method: str, path: str, body: str = "") -> dict:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp = str(int(time.time() * 1000))
        message = timestamp + method + path + body
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    def _request(self, method: str, path: str, body: str = ""):
        headers = self._sign(method, path, body)
        url = self.BASE_URL + path
        data = body.encode() if body else None
        req = request.Request(url, data=data, headers=headers, method=method)
        with request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())


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
        result = client._request("GET", "/portfolio/positions?limit=100")
        print(json.dumps(result, indent=2))
    elif args.orders:
        result = client._request("GET", "/portfolio/orders?status=resting")
        print(json.dumps(result, indent=2))
    elif args.balance:
        result = client._request("GET", "/portfolio/balance")
        print(json.dumps(result, indent=2))
    elif args.cancel:
        result = client._request("DELETE", f"/portfolio/orders/{args.cancel}")
        print(json.dumps(result, indent=2))
    elif args.ticker and args.side:
        print(f"Placing order: {args.count}x {args.side.upper()} on {args.ticker} ({args.type})")
        result = place_order(client, args.ticker, args.side, args.count,
                             args.type, args.price)
        print(json.dumps(result, indent=2))
        if isinstance(result, dict) and "order" in result:
            order = result["order"]
            print(f"\n Order placed!")
            print(f"   Order ID: {order.get('order_id')}")
            print(f"   Status:   {order.get('status')}")
        elif isinstance(result, dict) and ("error" in result or "code" in result):
            print(f"\n Order failed: {result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
