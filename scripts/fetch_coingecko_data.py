#!/usr/bin/env python3
"""
Enhanced Multi-Asset CoinGecko Data Fetcher
=============================================
Fetches multi-asset data (BTC, ETH, SOL) with 90 days history
and on-chain metrics for the ClawdBot/Yoshi-Bot forecaster pipeline.

Output: data/extended_multi.parquet (or CSV fallback)

Data quality requirements:
  - Min 200 samples per asset (for statistical power)
  - NaN handling: drop incomplete rows
  - Gap detection: log warnings for >2h gaps

On-chain metrics (CoinGecko free tier):
  - Market cap, 24h volume, circulating supply
  - Price change percentages (1h, 24h, 7d, 30d)
  - Market cap / volume ratio (velocity proxy)
  - Supply ratio (circulating / total)

Usage:
    python3 scripts/fetch_coingecko_data.py
    python3 scripts/fetch_coingecko_data.py --days 90 --symbols bitcoin,ethereum,solana
    python3 scripts/fetch_coingecko_data.py --output data/extended_multi.parquet
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from urllib import request, error as urlerror

import numpy as np

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
HTTP_TIMEOUT = 15

DEFAULT_SYMBOLS = ["bitcoin", "ethereum", "solana"]
DEFAULT_DAYS = 90

# Minimum bars for statistical power per asset
MIN_SAMPLES_PER_ASSET = 200

# Rate limit: CoinGecko free tier = 10-30 calls/min
RATE_LIMIT_DELAY = 2.5  # seconds between calls

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"

# Symbol mappings
SYMBOL_MAP = {
    "bitcoin": {"ticker": "BTCUSDT", "coinbase": "BTC-USD"},
    "ethereum": {"ticker": "ETHUSDT", "coinbase": "ETH-USD"},
    "solana": {"ticker": "SOLUSDT", "coinbase": "SOL-USD"},
}


# ═══════════════════════════════════════════════════════════════
# HTTP UTILITIES
# ═══════════════════════════════════════════════════════════════

def _get_json(url: str, timeout: int = HTTP_TIMEOUT) -> Optional[dict]:
    """Fetch JSON from URL with error handling."""
    try:
        req = request.Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "ClawdBot-V1/1.0 (forecaster)",
        })
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urlerror.URLError, urlerror.HTTPError, json.JSONDecodeError,
            TimeoutError, OSError) as e:
        print(f"  [WARN] HTTP error for {url[:80]}: {e}")
        return None


def _log(msg: str):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}")


# ═══════════════════════════════════════════════════════════════
# COINGECKO OHLCV FETCHER
# ═══════════════════════════════════════════════════════════════

def fetch_coingecko_ohlc(symbol: str, days: int = 90) -> list[dict]:
    """
    Fetch OHLCV data from CoinGecko.
    
    For days <= 90: returns hourly data (4h candles from /ohlc)
    For days > 90: returns daily data
    
    Returns list of dicts with: timestamp, open, high, low, close
    """
    # CoinGecko /ohlc endpoint returns 4h candles for 1-90 days
    url = f"{COINGECKO_BASE}/coins/{symbol}/ohlc?vs_currency=usd&days={days}"
    data = _get_json(url)
    
    if not data or not isinstance(data, list):
        _log(f"No OHLC data for {symbol}")
        return []
    
    bars = []
    for candle in data:
        if len(candle) >= 5:
            bars.append({
                "timestamp": candle[0] / 1000.0,  # ms -> seconds
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "symbol": symbol,
            })
    
    return bars


def fetch_coingecko_market_chart(symbol: str, days: int = 90) -> list[dict]:
    """
    Fetch detailed price/volume data from CoinGecko market_chart.
    
    Returns hourly granularity for days <= 90.
    """
    url = (
        f"{COINGECKO_BASE}/coins/{symbol}/market_chart"
        f"?vs_currency=usd&days={days}&interval=hourly"
    )
    data = _get_json(url)
    
    if not data or "prices" not in data:
        _log(f"No market_chart data for {symbol}")
        return []
    
    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])
    market_caps = data.get("market_caps", [])
    
    # Build volume and market cap lookup by timestamp
    vol_map = {}
    for ts_ms, vol in volumes:
        vol_map[int(ts_ms / 1000)] = vol
    
    mcap_map = {}
    for ts_ms, mc in market_caps:
        mcap_map[int(ts_ms / 1000)] = mc
    
    bars = []
    for i, (ts_ms, price) in enumerate(prices):
        ts = ts_ms / 1000.0
        ts_key = int(ts)
        
        bar = {
            "timestamp": ts,
            "close": price,
            "symbol": symbol,
            "volume": vol_map.get(ts_key, 0),
            "market_cap": mcap_map.get(ts_key, 0),
        }
        
        # Estimate open/high/low from adjacent prices
        if i > 0:
            prev_price = prices[i - 1][1]
            bar["open"] = prev_price
            bar["high"] = max(price, prev_price)
            bar["low"] = min(price, prev_price)
        else:
            bar["open"] = price
            bar["high"] = price
            bar["low"] = price
        
        bars.append(bar)
    
    return bars


# ═══════════════════════════════════════════════════════════════
# ON-CHAIN METRICS FETCHER
# ═══════════════════════════════════════════════════════════════

def fetch_onchain_metrics(symbol: str) -> dict:
    """
    Fetch on-chain proxy metrics from CoinGecko coin data.
    
    Available (free tier):
    - Market cap, volume, circulating supply
    - Price change percentages
    - Market cap / volume ratio (velocity proxy)
    - Supply ratio
    """
    url = (
        f"{COINGECKO_BASE}/coins/{symbol}"
        f"?localization=false&tickers=false&community_data=false"
        f"&developer_data=false&sparkline=false"
    )
    data = _get_json(url)
    
    if not data or not isinstance(data, dict):
        return {}
    
    metrics = {"symbol": symbol}
    
    try:
        md = data.get("market_data", {})
        
        # Core on-chain proxies
        metrics["market_cap_usd"] = md.get("market_cap", {}).get("usd", 0)
        metrics["total_volume_usd"] = md.get("total_volume", {}).get("usd", 0)
        metrics["current_price_usd"] = md.get("current_price", {}).get("usd", 0)
        
        # Velocity proxy: market_cap / volume
        if metrics["market_cap_usd"] > 0 and metrics["total_volume_usd"] > 0:
            metrics["mcap_volume_ratio"] = (
                metrics["market_cap_usd"] / metrics["total_volume_usd"]
            )
        
        # Price changes
        metrics["price_change_1h"] = md.get(
            "price_change_percentage_1h_in_currency", {}
        ).get("usd", 0) or 0
        metrics["price_change_24h"] = md.get(
            "price_change_percentage_24h", 0
        ) or 0
        metrics["price_change_7d"] = md.get(
            "price_change_percentage_7d", 0
        ) or 0
        metrics["price_change_30d"] = md.get(
            "price_change_percentage_30d", 0
        ) or 0
        
        # Supply metrics
        circ = md.get("circulating_supply", 0) or 0
        total = md.get("total_supply", 0) or 0
        max_supply = md.get("max_supply", 0) or 0
        
        metrics["circulating_supply"] = circ
        metrics["total_supply"] = total
        metrics["max_supply"] = max_supply
        
        if circ > 0 and total > 0:
            metrics["supply_ratio"] = circ / total
        
        # ATH distance (proxy for MVRV)
        ath = md.get("ath", {}).get("usd", 0) or 0
        current = metrics.get("current_price_usd", 0)
        if ath > 0 and current > 0:
            metrics["ath_distance_pct"] = (current - ath) / ath * 100
        
        # Fully diluted valuation
        fdv = md.get("fully_diluted_valuation", {}).get("usd", 0) or 0
        if fdv > 0:
            metrics["fdv_usd"] = fdv
            if metrics["market_cap_usd"] > 0:
                metrics["mcap_fdv_ratio"] = metrics["market_cap_usd"] / fdv
    
    except (KeyError, TypeError, ZeroDivisionError):
        pass
    
    return metrics


# ═══════════════════════════════════════════════════════════════
# MAIN EXTENDED DATA FETCHER
# ═══════════════════════════════════════════════════════════════

def fetch_extended_data(
    symbols: list[str] = None,
    days: int = DEFAULT_DAYS,
    include_onchain: bool = True,
) -> dict:
    """
    Fetch extended multi-asset data with on-chain metrics.
    
    Args:
        symbols: CoinGecko IDs (default: bitcoin, ethereum, solana)
        days: History length (default: 90)
        include_onchain: Include on-chain proxy metrics
    
    Returns:
        Dict with:
          - "bars": {symbol: list of bar dicts}
          - "onchain": {symbol: dict of on-chain metrics}
          - "quality": {symbol: quality report dict}
    """
    if symbols is None:
        symbols = list(DEFAULT_SYMBOLS)
    
    result = {
        "bars": {},
        "onchain": {},
        "quality": {},
        "metadata": {
            "fetch_time": datetime.now(timezone.utc).isoformat(),
            "symbols": symbols,
            "days": days,
            "include_onchain": include_onchain,
        },
    }
    
    for symbol in symbols:
        _log(f"Fetching {symbol} ({days} days)...")
        
        # Fetch OHLCV via market_chart (hourly granularity)
        bars = fetch_coingecko_market_chart(symbol, days)
        time.sleep(RATE_LIMIT_DELAY)
        
        # Fallback to OHLC if market_chart fails
        if len(bars) < 50:
            _log(f"  market_chart insufficient, trying OHLC...")
            bars = fetch_coingecko_ohlc(symbol, days)
            time.sleep(RATE_LIMIT_DELAY)
        
        # Drop NaNs and zero prices
        clean_bars = [
            b for b in bars
            if b.get("close", 0) > 0
            and not math.isnan(b.get("close", 0))
            and not math.isinf(b.get("close", 0))
        ]
        
        result["bars"][symbol] = clean_bars
        _log(f"  {symbol}: {len(clean_bars)} bars (raw: {len(bars)})")
        
        # Quality validation
        quality = _validate_bars(clean_bars, symbol)
        result["quality"][symbol] = quality
        
        # On-chain metrics
        if include_onchain:
            _log(f"  Fetching on-chain metrics for {symbol}...")
            onchain = fetch_onchain_metrics(symbol)
            result["onchain"][symbol] = onchain
            time.sleep(RATE_LIMIT_DELAY)
    
    # Power check
    total_samples = sum(len(b) for b in result["bars"].values())
    min_required = MIN_SAMPLES_PER_ASSET * len(symbols)
    result["metadata"]["total_samples"] = total_samples
    result["metadata"]["min_required"] = min_required
    result["metadata"]["power_sufficient"] = total_samples >= min_required
    
    if total_samples < min_required:
        _log(f"  [WARN] Power insufficient: {total_samples} < {min_required}")
    else:
        _log(f"  Power check PASSED: {total_samples} >= {min_required}")
    
    return result


def _validate_bars(bars: list[dict], symbol: str) -> dict:
    """Validate data quality for a set of bars."""
    report = {
        "symbol": symbol,
        "n_bars": len(bars),
        "min_required": MIN_SAMPLES_PER_ASSET,
        "issues": [],
    }
    
    if len(bars) < MIN_SAMPLES_PER_ASSET:
        report["issues"].append(
            f"Only {len(bars)} bars (need {MIN_SAMPLES_PER_ASSET})"
        )
    
    # Gap detection
    gaps = 0
    for i in range(1, len(bars)):
        dt = bars[i]["timestamp"] - bars[i - 1]["timestamp"]
        if dt > 7200:  # > 2 hours
            gaps += 1
    report["n_gaps"] = gaps
    if gaps > 0:
        report["issues"].append(f"{gaps} gaps detected (>2h)")
    
    # Price range
    if bars:
        closes = [b["close"] for b in bars]
        report["price_range"] = {
            "min": round(min(closes), 2),
            "max": round(max(closes), 2),
            "current": round(closes[-1], 2),
        }
    
    report["passed"] = len(report["issues"]) == 0
    return report


# ═══════════════════════════════════════════════════════════════
# PARQUET / CSV OUTPUT
# ═══════════════════════════════════════════════════════════════

def save_to_parquet(data: dict, output_path: str = None) -> str:
    """
    Save extended data to Parquet (or CSV fallback).
    
    Returns the output file path.
    """
    if output_path is None:
        output_path = str(DATA_DIR / "extended_multi.parquet")
    
    # Flatten bars into rows
    rows = []
    for symbol, bars in data["bars"].items():
        onchain = data.get("onchain", {}).get(symbol, {})
        for bar in bars:
            row = dict(bar)
            # Attach on-chain metrics to each row
            for k, v in onchain.items():
                if k != "symbol" and isinstance(v, (int, float)):
                    row[f"onchain_{k}"] = v
            rows.append(row)
    
    if not rows:
        _log("[ERROR] No data to save")
        return ""
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Try parquet first, fall back to CSV
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        
        # Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        
        if output_path.endswith(".parquet"):
            df.to_parquet(output_path, index=False, engine="pyarrow")
        else:
            df.to_csv(output_path, index=False)
        
        _log(f"Saved {len(df)} rows to {output_path}")
        return output_path
        
    except ImportError:
        # Fallback: save as CSV without pandas
        csv_path = output_path.replace(".parquet", ".csv")
        if rows:
            import csv
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            _log(f"Saved {len(rows)} rows to {csv_path} (CSV fallback)")
            return csv_path
        return ""


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Asset CoinGecko Data Fetcher"
    )
    parser.add_argument(
        "--symbols", "-s",
        default=",".join(DEFAULT_SYMBOLS),
        help=f"Comma-separated CoinGecko IDs (default: {','.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument(
        "--days", "-d", type=int, default=DEFAULT_DAYS,
        help=f"Days of history (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DATA_DIR / "extended_multi.parquet"),
        help="Output file path",
    )
    parser.add_argument(
        "--no-onchain", action="store_true",
        help="Skip on-chain metrics",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output metadata as JSON",
    )
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    print(f"\n{'='*60}")
    print(f"  ENHANCED MULTI-ASSET DATA FETCHER")
    print(f"{'='*60}")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Days:    {args.days}")
    print(f"  Output:  {args.output}")
    print(f"  On-chain: {'NO' if args.no_onchain else 'YES'}")
    print(f"{'='*60}\n")
    
    # Fetch data
    data = fetch_extended_data(
        symbols=symbols,
        days=args.days,
        include_onchain=not args.no_onchain,
    )
    
    # Save output
    output_path = save_to_parquet(data, args.output)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    
    for symbol in symbols:
        bars = data["bars"].get(symbol, [])
        quality = data["quality"].get(symbol, {})
        onchain = data["onchain"].get(symbol, {})
        
        print(f"\n  {symbol.upper()}:")
        print(f"    Bars:     {len(bars)}")
        print(f"    Quality:  {'PASSED' if quality.get('passed') else 'ISSUES'}")
        if quality.get("issues"):
            for issue in quality["issues"]:
                print(f"    WARNING:  {issue}")
        if quality.get("price_range"):
            pr = quality["price_range"]
            print(f"    Price:    ${pr['current']:,.2f} "
                  f"(range: ${pr['min']:,.2f} - ${pr['max']:,.2f})")
        if onchain.get("market_cap_usd"):
            print(f"    MCap:     ${onchain['market_cap_usd']:,.0f}")
        if onchain.get("total_volume_usd"):
            print(f"    Volume:   ${onchain['total_volume_usd']:,.0f}")
        if onchain.get("mcap_volume_ratio"):
            print(f"    MCap/Vol: {onchain['mcap_volume_ratio']:.2f}")
    
    meta = data["metadata"]
    print(f"\n  Total samples: {meta['total_samples']}")
    print(f"  Min required:  {meta['min_required']}")
    print(f"  Power:         {'SUFFICIENT' if meta['power_sufficient'] else 'INSUFFICIENT'}")
    
    if output_path:
        print(f"\n  Output: {output_path}")
    
    if args.json:
        print(f"\n  Metadata JSON:")
        print(json.dumps(meta, indent=2))
    
    print(f"{'='*60}")
    
    return data


if __name__ == "__main__":
    main()
