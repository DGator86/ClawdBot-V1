#!/usr/bin/env python3
"""
Kalshi Edge Scanner — Continuous Best-Pick Finder
==================================================
Scans ALL open Kalshi crypto markets every cycle, scores each contract
by expected value, and surfaces the top 1-2 best trades to ClawdBot
via the Trading Core /propose endpoint and a local JSON state file.

Runs as a systemd service on the VPS alongside yoshi-bridge.

Architecture:
  Kalshi API -> kalshi-edge-scanner.py -> top_picks.json + Trading Core /propose
  ClawdBot reads top_picks.json and presents to user via Telegram

Scoring model:
  For each open contract we compute:
    1. Implied probability from market mid (yes_bid + yes_ask) / 200
    2. Model probability from Yoshi's PriceTimeManifold (if available)
       OR a simpler statistical estimate from recent price action
    3. Edge = model_prob - market_prob
    4. Expected Value = edge * payout - (1 - edge) * cost
    5. Kelly fraction for optimal sizing
    6. Composite score = EV * confidence_weight * liquidity_weight

  We rank by composite score and pick the top 1-2.

Usage:
  python3 scripts/kalshi-edge-scanner.py                      # single scan
  python3 scripts/kalshi-edge-scanner.py --loop --interval 60  # every 60s
  python3 scripts/kalshi-edge-scanner.py --loop --interval 120 --top 2

Env vars (from Yoshi-Bot .env):
  KALSHI_KEY_ID, KALSHI_PRIVATE_KEY
  TRADING_CORE_URL (default http://127.0.0.1:8000)
"""

import argparse
import base64
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request, error as urlerror, parse as urlparse

from kalshi_client import KalshiClient

# ── Config ───────────────────────────────────────────────
TRADING_CORE_URL = os.getenv("TRADING_CORE_URL", "http://127.0.0.1:8000")
STATE_DIR = Path(__file__).parent.parent  # ClawdBot-V1 root
STATE_FILE = STATE_DIR / "data" / "top_picks.json"
LOG_FILE = STATE_DIR / "logs" / "edge-scanner.log"

# Series we scan
SERIES = ["KXBTC", "KXETH"]
SYMBOL_MAP = {"KXBTC": "BTCUSDT", "KXETH": "ETHUSDT"}

# Scoring weights
MIN_EDGE_PCT = 3.0          # minimum edge % to consider
MIN_EV_CENTS = 1.0          # minimum EV in cents per contract
MAX_CONTRACTS_DEFAULT = 10  # default position size suggestion
KELLY_FRACTION = 0.25       # quarter-Kelly for safety

# Ensemble mode: if True, skip contracts where ensemble returns None
# (don't fall back to price-distance heuristic)
REQUIRE_ENSEMBLE = os.getenv("REQUIRE_ENSEMBLE", "0") == "1"


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── .env loader (uses shared pem_utils when available) ───
# Keys that must be overwritten (systemd EnvironmentFile mangles multi-line PEM)
_FORCE_OVERWRITE_KEYS = {"KALSHI_PRIVATE_KEY"}

# Try to use shared utilities
try:
    from scripts.lib.pem_utils import fix_pem as _shared_fix_pem, load_env_file as _shared_load_env
    _HAS_SHARED_UTILS = True
except ImportError:
    _HAS_SHARED_UTILS = False


def _source_env(path: str):
    """Read a .env file and set vars in os.environ.
    
    KALSHI_PRIVATE_KEY is force-overwritten because systemd's
    EnvironmentFile truncates multi-line values to one line.
    """
    try:
        with open(path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, _, val = raw.partition("=")
                key = key.strip()
                val = val.strip()
                # Handle multi-line PEM values wrapped in quotes
                if val.startswith('"') and not val.endswith('"'):
                    # Multi-line value — keep reading
                    lines = [val[1:]]  # strip opening quote
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
                        os.environ[key] = val  # overwrite systemd's truncated value
                    else:
                        os.environ.setdefault(key, val)
    except Exception:
        pass


# ── Standalone Kalshi API Client ─────────────────────────
class KalshiClient:
    """
    Standalone Kalshi V2 API client with RSA-PSS SHA-256 auth.
    No external dependencies beyond `cryptography` (system package).
    
    Reads KALSHI_KEY_ID and KALSHI_PRIVATE_KEY from environment.
    """
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self):
        self.key_id = os.environ.get("KALSHI_KEY_ID", "").strip()
        pk_raw = os.environ.get("KALSHI_PRIVATE_KEY", "").strip()
        if not self.key_id:
            raise ValueError("KALSHI_KEY_ID not set")
        if not pk_raw:
            # Try loading from file
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

        # Load the RSA private key
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            self.private_key = load_pem_private_key(pk_raw.encode(), password=None)
        except ImportError:
            raise ImportError(
                "cryptography package not installed. Run: pip3 install cryptography"
            )

    @staticmethod
    def _fix_pem(raw: str) -> str:
        """
        Normalize a PEM key that may have been mangled by env var storage.
        Delegates to shared pem_utils when available; falls back to inline logic.
        """
        if _HAS_SHARED_UTILS:
            return _shared_fix_pem(raw)

        import re

        # Replace literal \n with real newlines
        if "\\n" in raw:
            raw = raw.replace("\\n", "\n")

        # If it has PEM headers but is mangled onto one/two lines
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

        # NO PEM headers — raw base64 (possibly with spaces instead of newlines)
        if "-----BEGIN" not in raw:
            # Strip all whitespace to get clean base64
            body = re.sub(r"\s+", "", raw)
            # Validate it looks like base64
            if len(body) > 100 and re.match(r"^[A-Za-z0-9+/=]+$", body):
                # Wrap at 64 chars and add RSA PRIVATE KEY headers
                lines = [body[i:i+64] for i in range(0, len(body), 64)]
                raw = "-----BEGIN RSA PRIVATE KEY-----\n" + "\n".join(lines) + "\n-----END RSA PRIVATE KEY-----"

        return raw.strip()

    def _sign(self, method: str, path: str, body: str = "") -> dict:
        """Create authenticated headers for a Kalshi API request."""
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
        """Make an authenticated HTTP request to Kalshi."""
        headers = self._sign(method, path, body)
        url = self.BASE_URL + path
        data = body.encode() if body else None
        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            log(f"Kalshi API error ({method} {path}): {e}", "ERROR")
            return None

    def get_exchange_status(self) -> dict | None:
        return self._request("GET", "/exchange/status")

    def list_markets(self, limit: int = 200, **kwargs) -> list[dict]:
        params = {"limit": str(limit)}
        for k, v in kwargs.items():
            params[k] = str(v)
        qs = urlparse.urlencode(params)
        result = self._request("GET", f"/markets?{qs}")
        if result and "markets" in result:
            return result["markets"]
        return result if isinstance(result, list) else []

    def get_market(self, ticker: str) -> dict | None:
        return self._request("GET", f"/markets/{ticker}")


def load_kalshi_client():
    """Load env files and return a configured KalshiClient instance."""
    # Source all known .env files for credentials
    for env_path in [
        "/root/Yoshi-Bot/.env",
        "/root/ClawdBot-V1/.env",
        "/home/root/Yoshi-Bot/.env",
        os.path.expanduser("~/.env"),
    ]:
        if os.path.isfile(env_path):
            _source_env(env_path)
            log(f"Loaded env from {env_path}")

    # Verify we have credentials
    if not os.environ.get("KALSHI_KEY_ID"):
        raise ImportError("KALSHI_KEY_ID not found in any .env file")

    # Create and return the client instance
    try:
        client = KalshiClient()
        log(f"Kalshi client initialized (key: {client.key_id[:12]}...)")
        return client
    except Exception as e:
        raise ImportError(f"Cannot create Kalshi client: {e}") from e


# ── Price fetching (lightweight, no ccxt dependency) ─────
def get_current_price(symbol: str) -> float | None:
    """Get current price from CoinGecko or Trading Core."""
    # Try Trading Core first
    try:
        req = request.Request(f"{TRADING_CORE_URL}/status")
        with request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            # Some cores expose last prices
            for pos in data.get("positions", []):
                if pos.get("symbol") == symbol:
                    return float(pos["current_price"])
    except Exception:
        pass

    # CoinGecko public API (no key needed for simple price)
    coin_map = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum"}
    coin_id = coin_map.get(symbol)
    if coin_id:
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            req = request.Request(url, headers={"Accept": "application/json"})
            with request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return float(data[coin_id]["usd"])
        except Exception as e:
            log(f"CoinGecko price fetch failed: {e}", "WARN")

    return None


# ── Edge & EV Calculation ────────────────────────────────
def compute_edge(market: dict, current_price: float | None) -> dict | None:
    """
    Compute edge, EV, and Kelly for a single Kalshi market contract.
    
    Returns a scored dict or None if not tradeable.
    """
    ticker = market.get("ticker", "")
    status = market.get("status", "")
    if status != "active":
        return None

    yes_bid = market.get("yes_bid", 0) or 0
    yes_ask = market.get("yes_ask", 0) or 0
    no_bid = market.get("no_bid", 0) or 0
    no_ask = market.get("no_ask", 0) or 0

    # Need at least one side with liquidity
    if yes_bid == 0 and yes_ask == 0:
        return None

    # Market implied probability (midpoint)
    if yes_bid > 0 and yes_ask > 0:
        market_prob = (yes_bid + yes_ask) / 200.0
    elif yes_ask > 0:
        market_prob = yes_ask / 100.0
    else:
        market_prob = yes_bid / 100.0

    # Extract strike
    strike = market.get("floor_strike")
    if strike is None:
        strike = market.get("strike_price")
    if strike is None:
        try:
            if "-T" in ticker:
                strike = float(ticker.split("-T")[-1])
            elif "-B" in ticker:
                strike = float(ticker.split("-B")[-1])
        except (ValueError, IndexError):
            return None
    if strike is None:
        return None
    strike = float(strike)

    # ── Model probability estimate ───────────────────────
    # Try ensemble forecaster first (12-paradigm), fall back to
    # simple price-distance logistic if ensemble unavailable.
    model_prob = None
    model_source = "none"
    forecast_meta = {}

    # Determine horizon from contract expiry
    close_time = market.get("close_time") or market.get("expiration_time") or ""
    horizon_hours = 24.0  # default
    if close_time:
        try:
            from datetime import datetime, timezone
            exp = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            delta = exp - datetime.now(timezone.utc)
            horizon_hours = max(1.0, delta.total_seconds() / 3600)
        except Exception:
            pass

    fallback_reason = None
    if current_price and current_price > 0:
        # ── Try 12-paradigm ensemble first ────────────────
        try:
            from scripts.forecaster.bridge import get_ensemble_model_prob
            series = market.get("series_ticker", "")
            symbol = SYMBOL_MAP.get(series, "BTCUSDT")
            ens_prob, ens_source, ens_meta = get_ensemble_model_prob(
                symbol=symbol,
                strike=strike,
                current_price=current_price,
                horizon_hours=horizon_hours,
            )
            if ens_prob is not None:
                model_prob = ens_prob
                model_source = ens_source
                forecast_meta = ens_meta
            else:
                fallback_reason = "ensemble_returned_none"
        except ImportError:
            fallback_reason = "ensemble_import_error"
        except Exception as e:
            fallback_reason = f"ensemble_error: {e}"
            log(f"Ensemble forecast error for {ticker}: {e}", "WARN")

        # ── Fallback: simple price-distance logistic ──────
        if model_prob is None:
            if REQUIRE_ENSEMBLE:
                log(f"Skipping {ticker}: ensemble unavailable ({fallback_reason})", "WARN")
                return None
            dist = (current_price - strike) / current_price
            spread = abs(yes_ask - yes_bid) / 100.0 if (yes_ask > 0 and yes_bid > 0) else 0.05
            hourly_vol = max(spread, 0.005)
            z = dist / hourly_vol if hourly_vol > 0 else 0
            model_prob = 1.0 / (1.0 + math.exp(-1.7 * z))
            model_source = "price-distance"
            if fallback_reason:
                log(f"Fallback to price-distance for {ticker}: {fallback_reason}", "INFO")
            if abs(dist) < 0.001:
                model_prob = 0.50 + (model_prob - 0.50) * 0.5

    if model_prob is None:
        return None

    # ── Edge ─────────────────────────────────────────────
    edge = model_prob - market_prob
    edge_pct = edge * 100.0

    # Determine the best side to trade
    if edge > 0:
        # Model says YES is underpriced -> BUY YES
        side = "yes"
        action = "buy"
        cost_cents = yes_ask if yes_ask > 0 else int(market_prob * 100)
        payout_cents = 100  # binary: pays $1 on YES
    else:
        # Model says YES is overpriced -> BUY NO (equivalent to selling YES)
        side = "no"
        action = "buy"
        cost_cents = no_ask if no_ask > 0 else int((1 - market_prob) * 100)
        payout_cents = 100
        edge = -edge  # flip to positive for scoring
        edge_pct = edge * 100.0
        model_prob = 1.0 - model_prob
        market_prob = 1.0 - market_prob

    if cost_cents <= 0 or cost_cents >= 100:
        return None

    # ── Expected Value per contract ──────────────────────
    # EV = prob_win * profit - prob_lose * cost
    prob_win = model_prob
    profit_cents = payout_cents - cost_cents
    ev_cents = prob_win * profit_cents - (1 - prob_win) * cost_cents

    # ── Kelly criterion ──────────────────────────────────
    # f* = (bp - q) / b  where b = profit/cost odds, p = win prob, q = 1-p
    b = profit_cents / cost_cents if cost_cents > 0 else 0
    kelly_full = (b * prob_win - (1 - prob_win)) / b if b > 0 else 0
    kelly_safe = max(0, kelly_full * KELLY_FRACTION)

    # ── Liquidity score ──────────────────────────────────
    spread_cents = abs(yes_ask - yes_bid) if (yes_ask > 0 and yes_bid > 0) else 99
    liquidity_score = max(0, 1.0 - spread_cents / 20.0)  # tighter spread = better

    # Volume weight (if available)
    volume = market.get("volume", 0) or 0
    volume_weight = min(1.0, volume / 100.0) if volume > 0 else 0.5

    # ── Composite score ──────────────────────────────────
    # Combines EV, edge magnitude, liquidity, and volume
    composite = (
        ev_cents * 0.4 +
        edge_pct * 0.3 +
        liquidity_score * 10 * 0.15 +
        volume_weight * 10 * 0.15
    )

    # ── Time to expiry ───────────────────────────────────
    close_time = market.get("close_time") or market.get("expiration_time") or ""
    minutes_to_expiry = None
    if close_time:
        try:
            exp = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            delta = exp - datetime.now(timezone.utc)
            minutes_to_expiry = max(0, delta.total_seconds() / 60)
        except Exception:
            pass

    return {
        "ticker": ticker,
        "series": market.get("series_ticker", ""),
        "side": side,
        "action": action,
        "strike": strike,
        "current_price": current_price,
        "market_prob": round(market_prob, 4),
        "model_prob": round(model_prob, 4),
        "model_source": model_source,
        "edge_pct": round(edge_pct, 2),
        "cost_cents": cost_cents,
        "ev_cents": round(ev_cents, 2),
        "kelly_fraction": round(kelly_safe, 4),
        "spread_cents": spread_cents,
        "liquidity_score": round(liquidity_score, 2),
        "volume": volume,
        "composite_score": round(composite, 3),
        "minutes_to_expiry": round(minutes_to_expiry, 1) if minutes_to_expiry else None,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "suggested_contracts": max(1, min(
            MAX_CONTRACTS_DEFAULT,
            int(kelly_safe * 100) if kelly_safe > 0 else 1
        )),
        "max_cost_dollars": round(
            max(1, min(MAX_CONTRACTS_DEFAULT, int(kelly_safe * 100))) * cost_cents / 100, 2
        ),
        "forecast_meta": forecast_meta,  # ensemble details (empty dict if fallback)
        "fallback_reason": fallback_reason,  # None if ensemble succeeded
    }


# ── Trading Core integration ────────────────────────────
def propose_to_core(pick: dict) -> dict | None:
    """Send best pick as a trade proposal to Trading Core."""
    try:
        # Check if core is healthy and not paused
        req = request.Request(f"{TRADING_CORE_URL}/health")
        with request.urlopen(req, timeout=5) as resp:
            health = json.loads(resp.read().decode())
            if health.get("status") != "healthy":
                return None

        req = request.Request(f"{TRADING_CORE_URL}/status")
        with request.urlopen(req, timeout=5) as resp:
            status = json.loads(resp.read().decode())
            if status.get("is_paused") or status.get("kill_switch_active"):
                log("Trading Core paused or kill switch active, skipping proposal")
                return None

        symbol = SYMBOL_MAP.get(pick["series"], "BTCUSDT")
        payload = {
            "exchange": "kalshi",
            "symbol": symbol,
            "side": "buy",
            "type": "market",
            "amount": pick["suggested_contracts"],
        }
        body = json.dumps(payload).encode()
        req = request.Request(
            f"{TRADING_CORE_URL}/propose",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log(f"Trading Core proposal failed: {e}", "WARN")
        return None


# ── Scanner Main Loop ────────────────────────────────────
def scan_once(client: KalshiClient, top_n: int = 2) -> list[dict]:
    """
    Run one full scan cycle:
    1. Check exchange status
    2. Fetch all open markets for each series
    3. Get current crypto prices
    4. Score every contract
    5. Return top N picks sorted by composite score
    """

    # 1. Exchange status
    ex_status = client.get_exchange_status()
    if not ex_status:
        log("Cannot reach Kalshi API", "ERROR")
        return []
    if not ex_status.get("exchange_active"):
        log("Kalshi exchange is closed")
        return []
    trading_active = ex_status.get("trading_active", False)
    log(f"Exchange: active={ex_status.get('exchange_active')}, trading={trading_active}")

    all_scored = []

    for series in SERIES:
        symbol = SYMBOL_MAP.get(series, "BTCUSDT")

        # 2. Get current price
        price = get_current_price(symbol)
        if price:
            log(f"{symbol} current price: ${price:,.2f}")
        else:
            log(f"{symbol} price unavailable, skipping {series}", "WARN")
            continue

        # 3. Fetch open markets
        try:
            markets = client.list_markets(limit=200, series_ticker=series, status="open")
        except Exception as e:
            log(f"Error fetching {series} markets: {e}", "ERROR")
            continue

        active_markets = [m for m in markets if m.get("status") == "active"]
        log(f"{series}: {len(active_markets)} active markets (of {len(markets)} open)")

        # 4. Score each
        for mkt in active_markets:
            scored = compute_edge(mkt, price)
            if scored is None:
                continue
            # Filter by minimum thresholds
            if scored["edge_pct"] < MIN_EDGE_PCT:
                continue
            if scored["ev_cents"] < MIN_EV_CENTS:
                continue
            all_scored.append(scored)

    # 5. Sort by composite score, return top N
    all_scored.sort(key=lambda x: x["composite_score"], reverse=True)
    top = all_scored[:top_n]

    log(f"Scanned {len(all_scored)} contracts above threshold, top {top_n} selected")
    return top


def format_pick(pick: dict, rank: int) -> str:
    """Format a single pick for display/logging."""
    lines = [
        f"{'='*50}",
        f"  #{rank} BEST PICK — {pick['ticker']}",
        f"{'='*50}",
        f"  Series:          {pick['series']}",
        f"  Strike:          ${pick['strike']:,.2f}",
        f"  Current Price:   ${pick['current_price']:,.2f}" if pick['current_price'] else "",
        f"  Side:            {pick['action'].upper()} {pick['side'].upper()}",
        f"  Market Prob:     {pick['market_prob']:.1%}",
        f"  Model Prob:      {pick['model_prob']:.1%}",
        f"  Edge:            {pick['edge_pct']:+.2f}%",
        f"  Cost:            {pick['cost_cents']}c/contract",
        f"  EV:              {pick['ev_cents']:+.2f}c/contract",
        f"  Kelly (safe):    {pick['kelly_fraction']:.2%}",
        f"  Spread:          {pick['spread_cents']}c",
        f"  Volume:          {pick['volume']}",
        f"  Score:           {pick['composite_score']:.3f}",
        f"  Suggested Size:  {pick['suggested_contracts']} contracts (${pick['max_cost_dollars']:.2f})",
    ]
    if pick.get("minutes_to_expiry") is not None:
        lines.append(f"  Expires in:      {pick['minutes_to_expiry']:.0f} min")
    return "\n".join(l for l in lines if l)


def format_telegram_alert(picks: list[dict]) -> str:
    """Format picks as a Telegram-friendly message."""
    lines = [
        "=" * 40,
        "🎯 KALSHI EDGE SCANNER — TOP PICKS",
        f"   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 40,
        "",
    ]
    for i, p in enumerate(picks, 1):
        emoji = "🥇" if i == 1 else "🥈"
        lines.extend([
            f"{emoji} #{i}: {p['ticker']}",
            f"   {p['action'].upper()} {p['side'].upper()} @ {p['cost_cents']}c",
            f"   Strike ${p['strike']:,.0f} | Edge {p['edge_pct']:+.1f}% | EV {p['ev_cents']:+.1f}c",
            f"   Market {p['market_prob']:.0%} vs Model {p['model_prob']:.0%}",
            f"   Size: {p['suggested_contracts']} contracts (${p['max_cost_dollars']:.2f})",
            "",
        ])
    lines.extend([
        "=" * 40,
        '💬 Reply "approve" or "details" in Telegram',
    ])
    return "\n".join(lines)


def save_state(picks: list[dict], scan_meta: dict):
    """Save current top picks to JSON for ClawdBot to read."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "scan_meta": scan_meta,
        "top_picks": picks,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    log(f"State saved to {STATE_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Kalshi Edge Scanner")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=120,
                        help="Seconds between scans (default: 120)")
    parser.add_argument("--top", type=int, default=2,
                        help="Number of top picks to surface (default: 2)")
    parser.add_argument("--min-edge", type=float, default=3.0,
                        help="Minimum edge %% (default: 3.0)")
    parser.add_argument("--propose", action="store_true",
                        help="Send top pick to Trading Core /propose")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--require-ensemble", action="store_true",
                        help="Skip contracts where ensemble is unavailable (no price-distance fallback)")
    args = parser.parse_args()

    global MIN_EDGE_PCT, REQUIRE_ENSEMBLE
    MIN_EDGE_PCT = args.min_edge
    if args.require_ensemble:
        REQUIRE_ENSEMBLE = True

    # Load Kalshi client
    try:
        client = load_kalshi_client()
    except ImportError as e:
        log(str(e), "FATAL")
        sys.exit(1)

    log(f"Kalshi Edge Scanner starting (top={args.top}, min_edge={args.min_edge}%, interval={args.interval}s)")

    cycle = 0
    while True:
        cycle += 1
        scan_start = time.time()
        log(f"--- Scan #{cycle} ---")

        try:
            picks = scan_once(client, top_n=args.top)
        except Exception as e:
            log(f"Scan failed: {e}", "ERROR")
            picks = []

        elapsed = time.time() - scan_start
        scan_meta = {
            "cycle": cycle,
            "elapsed_seconds": round(elapsed, 2),
            "min_edge_pct": args.min_edge,
            "top_n": args.top,
        }

        if picks:
            for i, p in enumerate(picks, 1):
                print(format_pick(p, i))

            if args.json:
                print(json.dumps({"picks": picks, "meta": scan_meta}, indent=2))

            # Save state for ClawdBot
            save_state(picks, scan_meta)

            # Write to scanner log (for yoshi-bridge to pick up)
            alert_text = format_telegram_alert(picks)
            print(f"\n{alert_text}\n")

            # Write alert to Yoshi scanner log so yoshi-bridge forwards it
            for yoshi_log in ["/root/Yoshi-Bot/logs/scanner.log",
                              "/home/root/Yoshi-Bot/logs/scanner.log"]:
                try:
                    Path(yoshi_log).parent.mkdir(parents=True, exist_ok=True)
                    with open(yoshi_log, "a") as f:
                        f.write(f"\n{alert_text}\n")
                    log(f"Alert written to {yoshi_log}")
                    break
                except Exception:
                    continue

            # Propose to Trading Core if requested
            if args.propose and picks:
                result = propose_to_core(picks[0])
                if result:
                    log(f"Proposed to Trading Core: {json.dumps(result)}")
        else:
            log("No contracts meet edge threshold this cycle")
            save_state([], scan_meta)

        log(f"Scan #{cycle} complete in {elapsed:.1f}s")

        if not args.loop:
            break

        log(f"Next scan in {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
