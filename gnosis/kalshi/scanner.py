"""
Kalshi Market Scanner — Fetches and scores open binary markets.
================================================================
Scans Kalshi markets, computes edge/EV/Kelly for each contract,
and returns ranked opportunities. Works standalone with no
external dependencies beyond `cryptography`.

Uses the standalone KalshiClient from scripts/kalshi-edge-scanner.py
auth pattern (RSA-PSS SHA-256) or falls back to gnosis.utils.kalshi_client.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib import request, error as urlerror, parse as urlparse


# ── PEM fix (inline, no dependencies) ──────────────────────
def _fix_pem(raw: str) -> str:
    """Normalize a PEM key that may have been mangled by env var storage."""
    if "\\n" in raw:
        raw = raw.replace("\\n", "\n")
    if "-----BEGIN" in raw and raw.count("\n") <= 2:
        m = re.search(
            r"-----BEGIN [A-Z ]+-----\s*(.*?)\s*-----END [A-Z ]+-----",
            raw, re.DOTALL,
        )
        if m:
            hdr = re.search(r"(-----BEGIN [A-Z ]+-----)", raw)
            ftr = re.search(r"(-----END [A-Z ]+-----)", raw)
            if hdr and ftr:
                body = m.group(1).replace(" ", "").replace("\n", "").replace("\r", "")
                lines = [body[i:i+64] for i in range(0, len(body), 64)]
                raw = hdr.group(1) + "\n" + "\n".join(lines) + "\n" + ftr.group(1)
        return raw.strip()
    if "-----BEGIN" not in raw:
        body = re.sub(r"\s+", "", raw)
        if len(body) > 100 and re.match(r"^[A-Za-z0-9+/=]+$", body):
            lines = [body[i:i+64] for i in range(0, len(body), 64)]
            raw = (
                "-----BEGIN RSA PRIVATE KEY-----\n"
                + "\n".join(lines)
                + "\n-----END RSA PRIVATE KEY-----"
            )
    return raw.strip()


# ── .env loader ────────────────────────────────────────────
def _load_env_files():
    """Source .env files for Kalshi credentials."""
    for env_path in [
        os.path.join(os.getcwd(), ".env"),
        "/root/ClawdBot-V1/.env",
        "/root/Yoshi-Bot/.env",
        "/root/kalshi_bot/.env",
        os.path.expanduser("~/.env"),
    ]:
        if os.path.isfile(env_path):
            try:
                with open(env_path) as f:
                    for raw_line in f:
                        raw_line = raw_line.strip()
                        if not raw_line or raw_line.startswith("#") or "=" not in raw_line:
                            continue
                        key, _, val = raw_line.partition("=")
                        key = key.strip()
                        val = val.strip()
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
                            if key == "KALSHI_PRIVATE_KEY":
                                os.environ[key] = val  # Always overwrite PEM
                            else:
                                os.environ.setdefault(key, val)
            except Exception:
                pass


# ── Standalone Kalshi API Client ───────────────────────────
class KalshiAPIClient:
    """
    Kalshi V2 API client with RSA-PSS SHA-256 authentication.
    No external deps beyond `cryptography`.
    """
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, key_id: str = None, private_key_pem: str = None):
        self.key_id = key_id or os.environ.get("KALSHI_KEY_ID", "").strip()
        pk_raw = private_key_pem or os.environ.get("KALSHI_PRIVATE_KEY", "").strip()

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

        pk_raw = _fix_pem(pk_raw)

        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        self.private_key = load_pem_private_key(pk_raw.encode(), password=None)

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

    def _request(self, method: str, path: str, body: str = "") -> Optional[dict]:
        headers = self._sign(method, path, body)
        url = self.BASE_URL + path
        data = body.encode() if body else None
        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"error": str(e)}

    def get_exchange_status(self) -> Optional[dict]:
        return self._request("GET", "/exchange/status")

    def list_markets(self, limit: int = 200, **kwargs) -> List[dict]:
        params = {"limit": str(limit)}
        for k, v in kwargs.items():
            params[k] = str(v)
        qs = urlparse.urlencode(params)
        result = self._request("GET", f"/markets?{qs}")
        if result and "markets" in result:
            return result["markets"]
        return result if isinstance(result, list) else []

    def get_market(self, ticker: str) -> Optional[dict]:
        result = self._request("GET", f"/markets/{ticker}")
        if result and "market" in result:
            return result["market"]
        return result

    def get_balance(self) -> Optional[dict]:
        return self._request("GET", "/portfolio/balance")

    def get_positions(self, limit: int = 100) -> List[dict]:
        result = self._request("GET", f"/portfolio/positions?limit={limit}")
        if result and "market_positions" in result:
            return result["market_positions"]
        return []

    def place_order(
        self, ticker: str, side: str, count: int,
        order_type: str = "market", price: int = None,
    ) -> Optional[dict]:
        payload: Dict[str, Any] = {
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
        return self._request("POST", "/portfolio/orders", body)


# ── Scan Result ────────────────────────────────────────────
@dataclass
class ScanResult:
    """A scored market opportunity."""
    ticker: str
    series: str = ""
    title: str = ""
    side: str = ""           # "yes" or "no"
    action: str = "buy"
    strike: float = 0.0
    market_prob: float = 0.0
    model_prob: float = 0.0
    model_source: str = ""
    edge_pct: float = 0.0
    cost_cents: int = 0
    ev_cents: float = 0.0
    kelly_fraction: float = 0.0
    spread_cents: int = 0
    volume: int = 0
    composite_score: float = 0.0
    minutes_to_expiry: Optional[float] = None
    yes_bid: int = 0
    yes_ask: int = 0
    no_bid: int = 0
    no_ask: int = 0
    suggested_contracts: int = 1
    max_cost_dollars: float = 0.0
    close_time: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def summary_line(self) -> str:
        return (
            f"{self.ticker}: {self.action.upper()} {self.side.upper()} "
            f"@ {self.cost_cents}c | "
            f"Edge {self.edge_pct:+.1f}% | EV {self.ev_cents:+.1f}c | "
            f"Market {self.market_prob:.0%} vs Model {self.model_prob:.0%}"
        )


# ── Scanner ────────────────────────────────────────────────
# Scoring constants
MIN_EDGE_PCT = 3.0
MIN_EV_CENTS = 1.0
KELLY_FRACTION = 0.25
MAX_CONTRACTS = 10


class KalshiScanner:
    """
    Scans Kalshi markets and scores each contract by edge/EV.

    Usage:
        scanner = KalshiScanner()
        results = scanner.scan(series=["KXBTC", "KXETH"])
    """

    def __init__(
        self,
        client: KalshiAPIClient = None,
        min_edge_pct: float = MIN_EDGE_PCT,
        min_ev_cents: float = MIN_EV_CENTS,
        max_contracts: int = MAX_CONTRACTS,
    ):
        _load_env_files()
        self.client = client or KalshiAPIClient()
        self.min_edge_pct = min_edge_pct
        self.min_ev_cents = min_ev_cents
        self.max_contracts = max_contracts

    def scan(
        self,
        series: List[str] = None,
        top_n: int = 5,
        current_prices: Dict[str, float] = None,
    ) -> List[ScanResult]:
        """
        Scan all open markets in the given series.

        Args:
            series: Kalshi series tickers (e.g., ["KXBTC"]). None = scan all.
            top_n: Number of top picks to return.
            current_prices: Optional dict of {symbol: price} for model prob.

        Returns:
            List of ScanResult sorted by composite score (descending).
        """
        # Check exchange status
        status = self.client.get_exchange_status()
        if not status or not status.get("exchange_active"):
            return []

        all_scored: List[ScanResult] = []

        if series:
            for s in series:
                markets = self.client.list_markets(
                    limit=200, series_ticker=s, status="open",
                )
                active = [m for m in markets if m.get("status") == "active"]
                symbol = _series_to_symbol(s)
                price = (current_prices or {}).get(symbol) or _get_price(symbol)
                for mkt in active:
                    result = self._score_market(mkt, price)
                    if result:
                        all_scored.append(result)
        else:
            # Scan all open markets
            markets = self.client.list_markets(limit=200, status="open")
            active = [m for m in markets if m.get("status") == "active"]
            for mkt in active:
                result = self._score_market(mkt, None)
                if result:
                    all_scored.append(result)

        all_scored.sort(key=lambda x: x.composite_score, reverse=True)
        return all_scored[:top_n]

    def _score_market(
        self, market: dict, current_price: Optional[float],
    ) -> Optional[ScanResult]:
        """Score a single market contract for edge and EV."""
        ticker = market.get("ticker", "")
        if market.get("status") != "active":
            return None

        yes_bid = market.get("yes_bid", 0) or 0
        yes_ask = market.get("yes_ask", 0) or 0
        no_bid = market.get("no_bid", 0) or 0
        no_ask = market.get("no_ask", 0) or 0

        if yes_bid == 0 and yes_ask == 0:
            return None

        # Market implied probability
        if yes_bid > 0 and yes_ask > 0:
            market_prob = (yes_bid + yes_ask) / 200.0
        elif yes_ask > 0:
            market_prob = yes_ask / 100.0
        else:
            market_prob = yes_bid / 100.0

        # Extract strike
        strike = market.get("floor_strike") or market.get("strike_price")
        if strike is None:
            try:
                if "-T" in ticker:
                    strike = float(ticker.split("-T")[-1])
                elif "-B" in ticker:
                    strike = float(ticker.split("-B")[-1])
            except (ValueError, IndexError):
                pass

        # Time to expiry
        close_time = market.get("close_time") or market.get("expiration_time") or ""
        minutes_to_expiry = None
        horizon_hours = 24.0
        if close_time:
            try:
                exp = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                delta = exp - datetime.now(timezone.utc)
                minutes_to_expiry = max(0, delta.total_seconds() / 60)
                horizon_hours = max(1.0, delta.total_seconds() / 3600)
            except Exception:
                pass

        # Model probability (price-distance logistic if we have price + strike)
        model_prob = None
        model_source = "none"

        if current_price and current_price > 0 and strike is not None:
            strike = float(strike)
            dist = (current_price - strike) / current_price
            spread = abs(yes_ask - yes_bid) / 100.0 if (yes_ask > 0 and yes_bid > 0) else 0.05
            hourly_vol = max(spread, 0.005)
            z = dist / hourly_vol if hourly_vol > 0 else 0
            model_prob = 1.0 / (1.0 + math.exp(-1.7 * z))
            model_source = "price-distance"
            if abs(dist) < 0.001:
                model_prob = 0.50 + (model_prob - 0.50) * 0.5
        elif strike is None:
            # Non-strike markets: use market prob as baseline
            # (LLM analyzer will refine later)
            model_prob = market_prob
            model_source = "market-baseline"

        if model_prob is None:
            return None

        # Edge calculation
        edge = model_prob - market_prob
        edge_pct = edge * 100.0

        if edge > 0:
            side = "yes"
            cost_cents = yes_ask if yes_ask > 0 else int(market_prob * 100)
        else:
            side = "no"
            cost_cents = no_ask if no_ask > 0 else int((1 - market_prob) * 100)
            edge = -edge
            edge_pct = edge * 100.0
            model_prob = 1.0 - model_prob
            market_prob = 1.0 - market_prob

        if cost_cents <= 0 or cost_cents >= 100:
            return None

        # Filter by minimums
        if edge_pct < self.min_edge_pct:
            return None

        # EV per contract
        profit_cents = 100 - cost_cents
        ev_cents = model_prob * profit_cents - (1 - model_prob) * cost_cents

        if ev_cents < self.min_ev_cents:
            return None

        # Kelly fraction
        b = profit_cents / cost_cents if cost_cents > 0 else 0
        kelly_full = (b * model_prob - (1 - model_prob)) / b if b > 0 else 0
        kelly_safe = max(0, kelly_full * KELLY_FRACTION)

        # Liquidity/volume scoring
        spread_cents = abs(yes_ask - yes_bid) if (yes_ask > 0 and yes_bid > 0) else 99
        liquidity_score = max(0, 1.0 - spread_cents / 20.0)
        volume = market.get("volume", 0) or 0
        volume_weight = min(1.0, volume / 100.0) if volume > 0 else 0.5

        composite = (
            ev_cents * 0.4
            + edge_pct * 0.3
            + liquidity_score * 10 * 0.15
            + volume_weight * 10 * 0.15
        )

        suggested = max(1, min(self.max_contracts, int(kelly_safe * 100) if kelly_safe > 0 else 1))

        return ScanResult(
            ticker=ticker,
            series=market.get("series_ticker", ""),
            title=market.get("title", ""),
            side=side,
            action="buy",
            strike=float(strike) if strike is not None else 0.0,
            market_prob=round(market_prob, 4),
            model_prob=round(model_prob, 4),
            model_source=model_source,
            edge_pct=round(edge_pct, 2),
            cost_cents=cost_cents,
            ev_cents=round(ev_cents, 2),
            kelly_fraction=round(kelly_safe, 4),
            spread_cents=spread_cents,
            volume=volume,
            composite_score=round(composite, 3),
            minutes_to_expiry=round(minutes_to_expiry, 1) if minutes_to_expiry is not None else None,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            suggested_contracts=suggested,
            max_cost_dollars=round(suggested * cost_cents / 100, 2),
            close_time=close_time,
        )


# ── Helpers ────────────────────────────────────────────────
_SERIES_MAP = {"KXBTC": "BTCUSDT", "KXETH": "ETHUSDT"}


def _series_to_symbol(series: str) -> str:
    return _SERIES_MAP.get(series, "BTCUSDT")


def _get_price(symbol: str) -> Optional[float]:
    """Get current price from CoinGecko (no key needed)."""
    coin_map = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum"}
    coin_id = coin_map.get(symbol)
    if not coin_id:
        return None
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        req = request.Request(url, headers={"Accept": "application/json"})
        with request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return float(data[coin_id]["usd"])
    except Exception:
        return None
