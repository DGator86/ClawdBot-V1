"""
Market Data Fetcher -- Live Data for Forecaster Modules
========================================================
Fetches OHLCV, derivatives, and macro data from public APIs
to build a MarketSnapshot for the ensemble forecaster.

Data source priority (US-compatible):
  1. Coinbase Exchange API  (OHLCV, orderbook, trades)
  2. Kraken API             (OHLCV, orderbook, trades)
  3. CoinGecko API          (OHLCV fallback, prices)
  4. Alternative.me         (Fear & Greed Index)

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
COINBASE = "https://api.exchange.coinbase.com"
KRAKEN = "https://api.kraken.com/0/public"
COINGECKO = "https://api.coingecko.com/api/v3"
ALTERNATIVE_ME = "https://api.alternative.me/fng/"

# Symbol mappings per exchange
COINBASE_SYMBOLS = {
    "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD", "BNBUSDT": None,
}
KRAKEN_SYMBOLS = {
    "BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD",
    "SOLUSDT": "SOLUSD", "BNBUSDT": None,
}
COINGECKO_IDS = {
    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
}

HTTP_TIMEOUT = 12


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
    except urlerror.HTTPError:
        return None
    except (urlerror.URLError, json.JSONDecodeError,
            OSError, TimeoutError, Exception):
        return None


def _log(msg: str):
    """Internal debug logging."""
    print(f"  {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════
# COINBASE DATA FETCHERS (PRIMARY — works from US)
# ═══════════════════════════════════════════════════════════════

def fetch_coinbase_klines(symbol: str, granularity: int = 3600,
                           limit: int = 200) -> list[Bar]:
    """
    Fetch OHLCV candles from Coinbase Exchange with pagination.
    granularity: seconds per candle (3600=1h, 14400=4h, 86400=1d)
    Coinbase returns max 300 candles per request — we paginate backward
    from now to collect up to `limit` bars.
    """
    cb_symbol = COINBASE_SYMBOLS.get(symbol)
    if not cb_symbol:
        return []

    PAGE_SIZE = 300  # Coinbase max per request
    all_bars: list[Bar] = []
    seen_timestamps: set[float] = set()

    # Start from now, paginate backward
    end_ts = int(time.time())
    remaining = limit
    max_pages = (limit // PAGE_SIZE) + 2  # safety cap

    for _page in range(max_pages):
        if remaining <= 0:
            break

        # Each page covers PAGE_SIZE * granularity seconds
        page_seconds = PAGE_SIZE * granularity
        start_ts = end_ts - page_seconds

        start_iso = datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()
        end_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()

        url = (f"{COINBASE}/products/{cb_symbol}/candles"
               f"?granularity={granularity}&start={start_iso}&end={end_iso}")
        data = _get_json(url)
        if not data or not isinstance(data, list) or len(data) == 0:
            break

        page_bars = []
        for candle in data:
            try:
                # Coinbase format: [time, low, high, open, close, volume]
                ts = float(candle[0])
                if ts in seen_timestamps:
                    continue
                seen_timestamps.add(ts)
                page_bars.append(Bar(
                    timestamp=ts,
                    open=float(candle[3]),
                    high=float(candle[2]),
                    low=float(candle[1]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                ))
            except (IndexError, ValueError, TypeError):
                continue

        if not page_bars:
            break

        all_bars.extend(page_bars)
        remaining -= len(page_bars)

        # Move window backward for next page
        end_ts = start_ts

        # Rate limit courtesy
        time.sleep(0.15)

    # Sort chronologically and trim
    all_bars.sort(key=lambda b: b.timestamp)
    if len(all_bars) > limit:
        all_bars = all_bars[-limit:]
    return all_bars


def fetch_coinbase_orderbook(symbol: str, level: int = 2) -> dict:
    """Fetch order book from Coinbase Exchange."""
    cb_symbol = COINBASE_SYMBOLS.get(symbol)
    if not cb_symbol:
        return {}

    url = f"{COINBASE}/products/{cb_symbol}/book?level={level}"
    data = _get_json(url)
    if not data:
        return {}

    try:
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        bid_depth = sum(float(b[1]) for b in bids[:20])
        ask_depth = sum(float(a[1]) for a in asks[:20])
        spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0
        return {
            "best_bid": best_bid, "best_ask": best_ask,
            "bid_depth": bid_depth, "ask_depth": ask_depth,
            "spread": spread, "source": "coinbase",
        }
    except (IndexError, ValueError, TypeError):
        return {}


def fetch_coinbase_trades(symbol: str, limit: int = 100) -> list[dict]:
    """Fetch recent trades from Coinbase."""
    cb_symbol = COINBASE_SYMBOLS.get(symbol)
    if not cb_symbol:
        return []

    url = f"{COINBASE}/products/{cb_symbol}/trades?limit={limit}"
    data = _get_json(url)
    if not data:
        return []

    trades = []
    for t in data:
        try:
            trades.append({
                "price": float(t["price"]),
                "qty": float(t["size"]),
                "side": t.get("side", "buy"),
                "time": 0,
            })
        except (KeyError, ValueError, TypeError):
            continue
    return trades



# ═══════════════════════════════════════════════════════════════
# KRAKEN DATA FETCHERS (TERTIARY — works from US)
# ═══════════════════════════════════════════════════════════════

def fetch_kraken_klines(symbol: str, interval: int = 60,
                         limit: int = 200) -> list[Bar]:
    """
    Fetch OHLCV from Kraken with pagination. interval in minutes (60=1h).
    Kraken returns up to 720 candles per request and provides a "last"
    cursor for pagination.
    """
    kr_pair = KRAKEN_SYMBOLS.get(symbol)
    if not kr_pair:
        return []

    all_bars: list[Bar] = []
    seen_timestamps: set[float] = set()

    # Start from (limit * interval * 60) seconds ago
    since = int(time.time()) - limit * interval * 60
    max_pages = (limit // 720) + 2  # safety cap

    for _page in range(max_pages):
        if len(all_bars) >= limit:
            break

        url = f"{KRAKEN}/OHLC?pair={kr_pair}&interval={interval}&since={since}"
        data = _get_json(url)
        if not data or data.get("error"):
            break

        result = data.get("result", {})
        last_cursor = result.get("last", 0)

        page_count = 0
        for key, candles in result.items():
            if key == "last":
                continue
            for c in candles:
                try:
                    ts = float(c[0])
                    if ts in seen_timestamps:
                        continue
                    seen_timestamps.add(ts)
                    all_bars.append(Bar(
                        timestamp=ts,
                        open=float(c[1]),
                        high=float(c[2]),
                        low=float(c[3]),
                        close=float(c[4]),
                        volume=float(c[6]),
                    ))
                    page_count += 1
                except (IndexError, ValueError, TypeError):
                    continue

        if page_count == 0:
            break

        # Use Kraken's "last" cursor for next page
        if last_cursor and last_cursor > since:
            since = last_cursor
        else:
            break

        time.sleep(0.2)  # Kraken rate limit courtesy

    all_bars.sort(key=lambda b: b.timestamp)
    if len(all_bars) > limit:
        all_bars = all_bars[-limit:]
    return all_bars


def fetch_kraken_orderbook(symbol: str, count: int = 20) -> dict:
    """Fetch order book from Kraken."""
    kr_pair = KRAKEN_SYMBOLS.get(symbol)
    if not kr_pair:
        return {}

    url = f"{KRAKEN}/Depth?pair={kr_pair}&count={count}"
    data = _get_json(url)
    if not data or data.get("error"):
        return {}

    result = data.get("result", {})
    for key, book in result.items():
        if key == "last":
            continue
        try:
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            bid_depth = sum(float(b[1]) for b in bids[:count])
            ask_depth = sum(float(a[1]) for a in asks[:count])
            spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0
            return {
                "best_bid": best_bid, "best_ask": best_ask,
                "bid_depth": bid_depth, "ask_depth": ask_depth,
                "spread": spread, "source": "kraken",
            }
        except (IndexError, ValueError, TypeError):
            continue
    return {}


def fetch_kraken_trades(symbol: str) -> list[dict]:
    """Fetch recent trades from Kraken."""
    kr_pair = KRAKEN_SYMBOLS.get(symbol)
    if not kr_pair:
        return []

    url = f"{KRAKEN}/Trades?pair={kr_pair}&count=200"
    data = _get_json(url)
    if not data or data.get("error"):
        return []

    result = data.get("result", {})
    trades = []
    for key, trade_list in result.items():
        if key == "last":
            continue
        for t in trade_list[-200:]:
            try:
                # Kraken: [price, volume, time, buy/sell, market/limit, misc, trade_id]
                trades.append({
                    "price": float(t[0]),
                    "qty": float(t[1]),
                    "side": "buy" if t[3] == "b" else "sell",
                    "time": float(t[2]),
                })
            except (IndexError, ValueError, TypeError):
                continue
    return trades


# ═══════════════════════════════════════════════════════════════
# COINGECKO OHLCV FALLBACK
# ═══════════════════════════════════════════════════════════════

def fetch_coingecko_ohlc(symbol: str, days: int = 7) -> list[Bar]:
    """
    Fetch OHLC from CoinGecko (limited granularity).
    days=1: 30min candles, days=7-30: 4h candles, days=90+: daily.
    No volume data. Last resort.
    """
    cg_id = COINGECKO_IDS.get(symbol)
    if not cg_id:
        return []

    url = f"{COINGECKO}/coins/{cg_id}/ohlc?vs_currency=usd&days={days}"
    data = _get_json(url)
    if not data or not isinstance(data, list):
        return []

    bars = []
    for candle in data:
        try:
            # CoinGecko OHLC: [timestamp_ms, open, high, low, close]
            bars.append(Bar(
                timestamp=candle[0] / 1000.0,
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=0.0,  # CoinGecko OHLC doesn't include volume
            ))
        except (IndexError, ValueError, TypeError):
            continue

    bars.sort(key=lambda b: b.timestamp)
    return bars


# ═══════════════════════════════════════════════════════════════
# DERIVATIVES DATA (best-effort from available sources)
# ═══════════════════════════════════════════════════════════════

def fetch_funding_rate(symbol: str) -> tuple[Optional[float], list[float]]:
    """Fetch funding rate data. Currently no US-accessible REST source."""
    # No reliable US-accessible REST endpoint for funding rates.
    # Future: integrate CoinGlass API or websocket feeds.
    return None, []


def fetch_open_interest(symbol: str) -> Optional[float]:
    """Fetch open interest data. Currently no US-accessible REST source."""
    # No reliable US-accessible REST endpoint for OI.
    # Future: integrate CoinGlass API or websocket feeds.
    return None


# ═══════════════════════════════════════════════════════════════
# ALTERNATIVE DATA
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
    """Fetch price from CoinGecko."""
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
    """Fetch cross-asset proxies for macro factor module."""
    result = {}
    data = _get_json(
        f"{COINGECKO}/simple/price?ids=ethereum&vs_currencies=btc"
    )
    if data and "ethereum" in data:
        try:
            result["eth_btc_ratio"] = float(data["ethereum"]["btc"])
        except (KeyError, ValueError):
            pass
    return result


def fetch_kalshi_priors(symbol: str) -> dict:
    """Read Kalshi barrier probabilities from top_picks.json."""
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
# MULTI-SOURCE OHLCV FETCHER WITH FALLBACK CHAIN
# ═══════════════════════════════════════════════════════════════

def _merge_bar_lists(*bar_lists: list[Bar]) -> list[Bar]:
    """
    Merge multiple Bar lists by timestamp, preferring the first source
    that provides data for a given timestamp.
    """
    seen: dict[int, Bar] = {}  # timestamp (rounded to minute) -> Bar
    for bars in bar_lists:
        for b in bars:
            # Round to nearest minute to handle small timestamp differences
            key = int(b.timestamp) // 60
            if key not in seen:
                seen[key] = b
    merged = sorted(seen.values(), key=lambda b: b.timestamp)
    return merged


def fetch_ohlcv_bars(symbol: str, limit: int = 200) -> tuple[list[Bar], str]:
    """
    Fetch hourly OHLCV bars with pagination and multi-source merging.

    Strategy:
    1. Paginate Coinbase (primary, max 300/request)
    2. Paginate Kraken (secondary, max 720/request)
    3. Merge both sources by timestamp for maximum coverage
    4. Fall back to CoinGecko OHLC if both fail

    Returns (bars, source_name).
    """
    sources_used = []

    # 1) Coinbase (paginated)
    cb_bars = fetch_coinbase_klines(symbol, granularity=3600, limit=limit)
    if cb_bars:
        sources_used.append("coinbase")
        _log(f"  Coinbase: {len(cb_bars)} bars")

    # 2) Kraken (paginated) — fetch if Coinbase didn't fill the request
    kr_bars = []
    if len(cb_bars) < limit:
        kr_bars = fetch_kraken_klines(symbol, interval=60, limit=limit)
        if kr_bars:
            sources_used.append("kraken")
            _log(f"  Kraken: {len(kr_bars)} bars")

    # 3) Merge both sources
    if cb_bars or kr_bars:
        merged = _merge_bar_lists(cb_bars, kr_bars)
        if len(merged) > limit:
            merged = merged[-limit:]
        source = "+".join(sources_used)
        if len(merged) >= 20:
            return merged, source

    # 4) CoinGecko fallback (4h candles for 30 days ≈ 180 bars)
    bars = fetch_coingecko_ohlc(symbol, days=30)
    if len(bars) >= 10:
        return bars, "coingecko"

    return [], "none"


def fetch_orderbook(symbol: str) -> dict:
    """Fetch order book using fallback chain."""
    book = fetch_coinbase_orderbook(symbol)
    if book:
        return book

    book = fetch_kraken_orderbook(symbol)
    if book:
        return book

    return {}


def fetch_recent_trades(symbol: str, limit: int = 200) -> tuple[list[dict], str]:
    """Fetch recent trades using fallback chain."""
    trades = fetch_coinbase_trades(symbol, limit)
    if trades:
        return trades, "coinbase"

    trades = fetch_kraken_trades(symbol)
    if trades:
        return trades, "kraken"

    return [], "none"


# ═══════════════════════════════════════════════════════════════
# MAIN SNAPSHOT BUILDER
# ═══════════════════════════════════════════════════════════════

def fetch_market_snapshot(symbol: str = "BTCUSDT",
                           bars_limit: int = 200) -> MarketSnapshot:
    """
    Build a complete MarketSnapshot from all available data sources.
    Uses multi-source fallback: Coinbase -> Kraken -> CoinGecko.
    Gracefully handles missing data -- each module checks for None fields.
    """
    snap = MarketSnapshot(symbol=symbol, timestamp=time.time())

    # ── OHLCV bars (with fallback chain) ──────────────────
    _log(f"Fetching {symbol} 1h bars...")
    snap.bars_1h, ohlcv_source = fetch_ohlcv_bars(symbol, bars_limit)
    _log(f"  -> {len(snap.bars_1h)} bars from {ohlcv_source}")

    if not snap.bars_1h:
        _log("WARNING: No OHLCV data from any source!")

    # Build multi-timeframe bars by aggregation
    if len(snap.bars_1h) >= 24:
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

    # ── Derivatives data (best-effort) ──
    _log("Fetching derivatives data...")
    fr, fr_hist = fetch_funding_rate(symbol)
    snap.funding_rate = fr
    snap.funding_rate_history = fr_hist
    snap.open_interest = fetch_open_interest(symbol)
    deriv_items = sum(1 for x in [fr, snap.open_interest] if x is not None)
    _log(f"  -> {deriv_items}/2 items" + (" (no US-accessible source)" if deriv_items == 0 else ""))

    # ── Order book (with fallback chain) ──────────────────
    _log("Fetching order book...")
    book = fetch_orderbook(symbol)
    if book:
        snap.best_bid = book.get("best_bid")
        snap.best_ask = book.get("best_ask")
        snap.bid_depth = book.get("bid_depth")
        snap.ask_depth = book.get("ask_depth")
        snap.spread = book.get("spread")
        _log(f"  -> OK from {book.get('source', '?')}")
    else:
        _log("  -> unavailable")

    # ── Recent trades (with fallback chain) ────────────────
    _log("Fetching recent trades...")
    snap.recent_trades, trades_source = fetch_recent_trades(symbol, limit=200)
    _log(f"  -> {len(snap.recent_trades)} trades from {trades_source}")

    # ── Macro proxies ─────────────────────────────────────
    _log("Fetching macro proxies...")
    macro = fetch_macro_proxies()
    snap.eth_btc_ratio = macro.get("eth_btc_ratio")
    snap.spx_return_1d = macro.get("spx_return_1d")
    snap.dxy_return_1d = macro.get("dxy_return_1d")
    snap.gold_return_1d = macro.get("gold_return_1d")
    _log(f"  -> {len(macro)} items")

    # ── Sentiment ─────────────────────────────────────────
    _log("Fetching sentiment...")
    snap.fear_greed_index = fetch_fear_greed()
    _log(f"  -> FGI={snap.fear_greed_index}" if snap.fear_greed_index else "  -> unavailable")

    # ── Kalshi priors ─────────────────────────────────────
    snap.kalshi_barrier_probs = fetch_kalshi_priors(symbol)
    if snap.kalshi_barrier_probs:
        _log(f"Kalshi priors: {len(snap.kalshi_barrier_probs)} strikes")

    _log(f"Snapshot ready: {symbol} @ ${snap.current_price:,.2f} "
         f"({len(snap.bars_1h)} bars from {ohlcv_source})")
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
    print(f"  Spread:    {snap.spread}")
    print(f"  FGI:       {snap.fear_greed_index}")
    print(f"{'='*50}")

    if args.output:
        out = {
            "symbol": snap.symbol,
            "timestamp": snap.timestamp,
            "current_price": snap.current_price,
            "n_bars_1h": len(snap.bars_1h),
            "funding_rate": snap.funding_rate,
            "open_interest": snap.open_interest,
            "spread": snap.spread,
            "fear_greed_index": snap.fear_greed_index,
            "eth_btc_ratio": snap.eth_btc_ratio,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Snapshot metadata saved to {args.output}")
