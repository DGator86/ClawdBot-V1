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
import json
import math
import os
import sys
import time
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from urllib import request, error as urlerror

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


# ── Kalshi Client (load from Yoshi-Bot or standalone) ────
def load_kalshi_client():
    """Try to load KalshiClient from Yoshi-Bot, fallback to bundled."""
    # Try Yoshi-Bot locations
    for yoshi_dir in ["/root/Yoshi-Bot", "/home/root/Yoshi-Bot"]:
        client_path = os.path.join(yoshi_dir, "src", "gnosis", "utils", "kalshi_client.py")
        env_path = os.path.join(yoshi_dir, ".env")
        if os.path.isfile(client_path):
            # Source the .env
            if os.path.isfile(env_path):
                _source_env(env_path)
            # Direct import to avoid __init__.py chain
            spec = importlib.util.spec_from_file_location("kalshi_client", client_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            log(f"Loaded KalshiClient from {client_path}")
            return mod.KalshiClient
    raise ImportError("KalshiClient not found — is Yoshi-Bot installed?")


def _source_env(path: str):
    """Read a .env file and set vars in os.environ."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val:
                    os.environ.setdefault(key, val)
    except Exception:
        pass


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
    # If we have current price, we estimate probability that price
    # will be above the strike at contract expiry.
    # Simple approach: distance from strike as a z-score using
    # recent implied vol from the market's own spread.
    model_prob = None
    model_source = "none"

    if current_price and current_price > 0:
        # Distance to strike as fraction of price
        dist = (current_price - strike) / current_price

        # Use market-implied vol from the spread width
        spread = abs(yes_ask - yes_bid) / 100.0 if (yes_ask > 0 and yes_bid > 0) else 0.05
        # Hourly vol estimate: spread gives us a rough idea
        hourly_vol = max(spread, 0.005)  # floor at 0.5%

        # For "above strike" contracts:
        # If current price > strike, base prob is > 50%
        # z = distance / vol (positive means price is above strike)
        z = dist / hourly_vol if hourly_vol > 0 else 0

        # Convert to probability using logistic approximation
        # (faster than importing scipy)
        model_prob = 1.0 / (1.0 + math.exp(-1.7 * z))
        model_source = "price-distance"

        # Adjust for momentum: if price is well above strike,
        # increase confidence; if barely above, decrease
        if abs(dist) < 0.001:  # very close to strike — uncertainty
            model_prob = 0.50 + (model_prob - 0.50) * 0.5  # pull toward 50%

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
def scan_once(kalshi_client, top_n: int = 2) -> list[dict]:
    """
    Run one full scan cycle:
    1. Check exchange status
    2. Fetch all open markets for each series
    3. Get current crypto prices
    4. Score every contract
    5. Return top N picks sorted by composite score
    """
    client = kalshi_client()

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
    args = parser.parse_args()

    global MIN_EDGE_PCT
    MIN_EDGE_PCT = args.min_edge

    # Load Kalshi client
    try:
        KalshiClient = load_kalshi_client()
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
            picks = scan_once(KalshiClient, top_n=args.top)
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
