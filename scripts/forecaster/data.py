"""
Market Data Fetcher -- Live Data for Forecaster Modules
========================================================
Fetches OHLCV, derivatives, and macro data from public APIs
to build a MarketSnapshot for the ensemble forecaster.

Data sources:
  - Binance spot/futures API (OHLCV, funding, OI, trades)
  - CoinGecko (fallback prices, fear & greed)
  - Alternative.me (Fear & Greed Index)

All fetches use urllib (no extra dependencies) with timeouts.

Usage:
    from scripts.forecaster.data import fetch_market_snapshot
    snap = fetch_market_snapshot("BTCUSDT")
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Optional
from urllib import request, error as urlerror

from .schemas import Bar, MarketSnapshot

# ── API endpoints ──────────────────────────────────────────
BINANCE_SPOT = "https://api.binance.com"
BINANCE_FUTURES = "https://fapi.binance.com"
COINGECKO = "https://api.coingecko.com/api/v3"
ALTERNATIVE_ME = "https://api.alternative.me/fng/"

# Symbol mappings
COINGECKO_IDS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "BNBUSDT": "binancecoin",
}

# Timeout for HTTP requests (seconds)
HTTP_TIMEOUT = 10


# ═══════════════════════════════════════════════════════════════
# HTTP HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_json(url: str, timeout: int = HTTP_TIMEOUT) -> Optional[dict | list]:
    """Fetch JSON from URL with timeout and error handling."""
    try:
        req = request.Request(url, headers={
            "User-Agent": "ClawdBot-Forecaster/1.0",
            "Accept": "application/json",
        })
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urlerror.URLError, urlerror.HTTPError, json.JSONDecodeError,
            OSError, TimeoutError) as e:
        # Silently fail -- data modules handle missing fields gracefully
        return None


def _get_trading_core(endpoint: str) -> Optional[dict]:
    """Try to fetch from the local Trading Core API."""
    url = os.getenv("TRADING_CORE_URL", "http://127.0.0.1:8000")
    return _get_json(f"{url}{endpoint}", timeout=3)


# ═══════════════════════════════════════════════════════════════
# BINANCE DATA FETCHERS
# ═══════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol: str, interval: str = "1h",
                          limit: int = 200) -> list[Bar]:
    """Fetch OHLCV klines from Binance spot."""
    url = (f"{BINANCE_SPOT}/api/v3/klines"
           f"?symbol={symbol}&interval={interval}&limit={limit}")
    data = _get_json(url)
    if not data:
        return []

    bars = []
    for k in data:
        try:
            bars.append(Bar(
                timestamp=k[0] / 1000.0,  # ms -> s
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
            ))
        except (IndexError, ValueError, TypeError):
            continue
    return bars


def fetch_binance_funding_rate(symbol: str, limit: int = 30) -> tuple[Optional[float], list[float]]:
    """Fetch current and historical funding rates from Binance futures."""
    url = (f"{BINANCE_FUTURES}/fapi/v1/fundingRate"
           f"?symbol={symbol}&limit={limit}")
    data = _get_json(url)
    if not data:
        return None, []

    rates = []
    for item in data:
        try:
            rates.append(float(item["fundingRate"]))
        except (KeyError, ValueError):
            continue

    current = rates[-1] if rates else None
    return current, rates


def fetch_binance_open_interest(symbol: str) -> Optional[float]:
    """Fetch current open interest from Binance futures."""
    url = f"{BINANCE_FUTURES}/fapi/v1/openInterest?symbol={symbol}"
    data = _get_json(url)
    if data and "openInterest" in data:
        try:
            return float(data["openInterest"])
        except (ValueError, TypeError):
            pass
    return None


def fetch_binance_oi_history(symbol: str, period: str = "1h",
                               limit: int = 30) -> list[float]:
    """Fetch OI history from Binance futures."""
    url = (f"{BINANCE_FUTURES}/futures/data/openInterestHist"
           f"?symbol={symbol}&period={period}&limit={limit}")
    data = _get_json(url)
    if not data:
        return []
    ois = []
    for item in data:
        try:
            ois.append(float(item.get("sumOpenInterest", 0)))
        except (ValueError, TypeError):
            continue
    return ois


def fetch_binance_long_short_ratio(symbol: str) -> Optional[float]:
    """Fetch long/short ratio from Binance futures."""
    url = (f"{BINANCE_FUTURES}/futures/data/globalLongShortAccountRatio"
           f"?symbol={symbol}&period=1h&limit=1")
    data = _get_json(url)
    if data and len(data) > 0:
        try:
            return float(data[-1].get("longShortRatio", 1.0))
        except (ValueError, TypeError):
            pass
    return None


def fetch_binance_orderbook(symbol: str, limit: int = 20) -> dict:
    """Fetch order book depth and spread."""
    url = f"{BINANCE_SPOT}/api/v3/depth?symbol={symbol}&limit={limit}"
    data = _get_json(url)
    if not data:
        return {}

    try:
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0

        # Total depth within first N levels
        bid_depth = sum(float(b[1]) for b in bids[:limit])
        ask_depth = sum(float(a[1]) for a in asks[:limit])

        spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "spread": spread,
        }
    except (IndexError, ValueError, TypeError):
        return {}


def fetch_binance_recent_trades(symbol: str, limit: int = 100) -> list[dict]:
    """Fetch recent trades for microstructure analysis."""
    url = f"{BINANCE_SPOT}/api/v3/trades?symbol={symbol}&limit={limit}"
    data = _get_json(url)
    if not data:
        return []

    trades = []
    for t in data:
        try:
            trades.append({
                "price": float(t["price"]),
                "qty": float(t["qty"]),
                "side": "sell" if t.get("isBuyerMaker", False) else "buy",
                "time": t.get("time", 0) / 1000.0,
            })
        except (KeyError, ValueError, TypeError):
            continue
    return trades


# ═══════════════════════════════════════════════════════════════
# ALTERNATIVE DATA FETCHERS
# ═══════════════════════════════════════════════════════════════

def fetch_fear_greed() -> Optional[float]:
    """Fetch Fear & Greed Index from alternative.me."""
    data = _get_json(ALTERNATIVE_ME)
    if data and "data" in data and len(data["data"]) > 0:
        try:
            return float(data["data"][0]["value"])
        except (KeyError, ValueError, TypeError):
            pass
    return None


def fetch_coingecko_price(symbol: str) -> Optional[float]:
    """Fallback price fetch from CoinGecko."""
    cg_id = COINGECKO_IDS.get(symbol)
    if not cg_id:
        return None
    url = f"{COINGECKO}/simple/price?ids={cg_id}&vs_currencies=usd"
    data = _get_json(url)
    if data and cg_id in data:
        try:
            return float(data[cg_id]["usd"])
        except (KeyError, ValueError, TypeError):
            pass
    return None


def fetch_macro_proxies() -> dict:
    """
    Fetch cross-asset proxies for macro factor module.
    Uses CoinGecko market data where available.
    Returns dict with spx_return_1d, dxy_return_1d, etc.
    """
    # ETH/BTC ratio from CoinGecko
    result = {}
    data = _get_json(
        f"{COINGECKO}/simple/price"
        f"?ids=ethereum&vs_currencies=btc"
    )
    if data and "ethereum" in data:
        try:
            result["eth_btc_ratio"] = float(data["ethereum"]["btc"])
        except (KeyError, ValueError):
            pass

    # For SPX/DXY/Gold, we'd need a paid API or proxy.
    # Leave as None and the macro module handles gracefully.
    return result


# ═══════════════════════════════════════════════════════════════
# KALSHI DATA (from existing scanner state)
# ═══════════════════════════════════════════════════════════════

def fetch_kalshi_priors(symbol: str) -> dict:
    """
    Read Kalshi barrier probabilities from top_picks.json
    (written by kalshi-edge-scanner.py).
    """
    priors = {}
    state_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "top_picks.json"),
        "/root/ClawdBot-V1/data/top_picks.json",
    ]
    for path in state_paths:
        try:
            with open(path) as f:
                data = json.load(f)
            picks = data.get("picks", [])
            series_key = "KXBTC" if "BTC" in symbol else "KXETH" if "ETH" in symbol else ""
            for pick in picks:
                if pick.get("series") == series_key:
                    strike = pick.get("strike", 0)
                    if strike > 0:
                        priors[str(strike)] = pick.get("market_prob", 0.5)
            if priors:
                break
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            continue
    return priors


# ═══════════════════════════════════════════════════════════════
# MAIN SNAPSHOT BUILDER
# ═══════════════════════════════════════════════════════════════

def fetch_market_snapshot(symbol: str = "BTCUSDT",
                           bars_limit: int = 200) -> MarketSnapshot:
    """
    Build a complete MarketSnapshot from all available data sources.
    Gracefully handles missing data -- each module checks for None fields.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        bars_limit: Number of hourly bars to fetch

    Returns:
        MarketSnapshot with all available data populated.
    """
    snap = MarketSnapshot(symbol=symbol, timestamp=time.time())

    # ── OHLCV bars ────────────────────────────────────────
    print(f"  Fetching {symbol} 1h bars...", end=" ", flush=True)
    snap.bars_1h = fetch_binance_klines(symbol, "1h", bars_limit)
    print(f"{len(snap.bars_1h)} bars")

    if len(snap.bars_1h) >= 24:
        # Build 4h bars by aggregation
        bars_4h = []
        for i in range(0, len(snap.bars_1h) - 3, 4):
            chunk = snap.bars_1h[i:i+4]
            bars_4h.append(Bar(
                timestamp=chunk[0].timestamp,
                open=chunk[0].open,
                high=max(b.high for b in chunk),
                low=min(b.low for b in chunk),
                close=chunk[-1].close,
                volume=sum(b.volume for b in chunk),
            ))
        snap.bars_4h = bars_4h

    if len(snap.bars_1h) >= 48:
        # Build daily bars
        bars_1d = []
        for i in range(0, len(snap.bars_1h) - 23, 24):
            chunk = snap.bars_1h[i:i+24]
            bars_1d.append(Bar(
                timestamp=chunk[0].timestamp,
                open=chunk[0].open,
                high=max(b.high for b in chunk),
                low=min(b.low for b in chunk),
                close=chunk[-1].close,
                volume=sum(b.volume for b in chunk),
            ))
        snap.bars_1d = bars_1d

    # ── Derivatives data ──────────────────────────────────
    print(f"  Fetching derivatives data...", end=" ", flush=True)
    fr, fr_hist = fetch_binance_funding_rate(symbol)
    snap.funding_rate = fr
    snap.funding_rate_history = fr_hist

    snap.open_interest = fetch_binance_open_interest(symbol)
    snap.oi_history = fetch_binance_oi_history(symbol)
    snap.long_short_ratio = fetch_binance_long_short_ratio(symbol)
    deriv_items = sum(1 for x in [fr, snap.open_interest, snap.long_short_ratio] if x is not None)
    print(f"{deriv_items}/3 items")

    # ── Order book / microstructure ───────────────────────
    print(f"  Fetching order book...", end=" ", flush=True)
    book = fetch_binance_orderbook(symbol)
    if book:
        snap.best_bid = book.get("best_bid")
        snap.best_ask = book.get("best_ask")
        snap.bid_depth = book.get("bid_depth")
        snap.ask_depth = book.get("ask_depth")
        snap.spread = book.get("spread")
        print("OK")
    else:
        print("unavailable")

    print(f"  Fetching recent trades...", end=" ", flush=True)
    snap.recent_trades = fetch_binance_recent_trades(symbol, limit=200)
    print(f"{len(snap.recent_trades)} trades")

    # ── Macro proxies ─────────────────────────────────────
    print(f"  Fetching macro proxies...", end=" ", flush=True)
    macro = fetch_macro_proxies()
    snap.eth_btc_ratio = macro.get("eth_btc_ratio")
    snap.spx_return_1d = macro.get("spx_return_1d")
    snap.dxy_return_1d = macro.get("dxy_return_1d")
    snap.gold_return_1d = macro.get("gold_return_1d")
    print(f"{len(macro)} items")

    # ── Sentiment ─────────────────────────────────────────
    print(f"  Fetching sentiment...", end=" ", flush=True)
    snap.fear_greed_index = fetch_fear_greed()
    print(f"FGI={snap.fear_greed_index}" if snap.fear_greed_index else "unavailable")

    # ── Kalshi priors ─────────────────────────────────────
    snap.kalshi_barrier_probs = fetch_kalshi_priors(symbol)
    if snap.kalshi_barrier_probs:
        print(f"  Kalshi priors: {len(snap.kalshi_barrier_probs)} strikes")

    print(f"  Snapshot ready: {symbol} @ ${snap.current_price:,.2f}")
    return snap


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch market data snapshot")
    parser.add_argument("--symbol", "-s", default="BTCUSDT")
    parser.add_argument("--output", "-o", default=None,
                        help="Save snapshot to JSON file")
    args = parser.parse_args()

    snap = fetch_market_snapshot(args.symbol)

    print(f"\n{'='*50}")
    print(f"  MARKET SNAPSHOT: {snap.symbol}")
    print(f"{'='*50}")
    print(f"  Price:     ${snap.current_price:,.2f}")
    print(f"  1h bars:   {len(snap.bars_1h)}")
    print(f"  4h bars:   {len(snap.bars_4h)}")
    print(f"  1d bars:   {len(snap.bars_1d)}")
    print(f"  Funding:   {snap.funding_rate}")
    print(f"  OI:        {snap.open_interest}")
    print(f"  L/S Ratio: {snap.long_short_ratio}")
    print(f"  Spread:    {snap.spread}")
    print(f"  FGI:       {snap.fear_greed_index}")
    print(f"{'='*50}")

    if args.output:
        # Serialize snapshot
        out = {
            "symbol": snap.symbol,
            "timestamp": snap.timestamp,
            "current_price": snap.current_price,
            "n_bars_1h": len(snap.bars_1h),
            "funding_rate": snap.funding_rate,
            "open_interest": snap.open_interest,
            "long_short_ratio": snap.long_short_ratio,
            "spread": snap.spread,
            "fear_greed_index": snap.fear_greed_index,
            "eth_btc_ratio": snap.eth_btc_ratio,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Snapshot metadata saved to {args.output}")
