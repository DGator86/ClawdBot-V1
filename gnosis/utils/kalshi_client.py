"""Kalshi V2 API Client.

Provides authentication and data fetching for Kalshi prediction markets.
"""
import base64
import os
import time
from typing import Dict, Any, Optional, List

import requests  # type: ignore
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv

load_dotenv()


class KalshiClient:
    """Authenticating client for Kalshi V2 API."""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self):
        self.key_id = os.getenv("KALSHI_KEY_ID")
        self.private_key_str = os.getenv("KALSHI_PRIVATE_KEY")

        if not self.key_id or not self.private_key_str:
            raise ValueError("Kalshi credentials missing in .env")

        # Clean up the private key format if needed (remove quotes if added)
        if (self.private_key_str.startswith('"') and
                self.private_key_str.endswith('"')):
            self.private_key_str = self.private_key_str[1:-1]

        self.private_key = serialization.load_pem_private_key(
            self.private_key_str.encode(),
            password=None
        )

    def _get_headers(self, method: str, path: str,
                     body: str = "") -> Dict[str, str]:
        """Generate authentication headers for Kalshi request."""
        timestamp = str(int(time.time() * 1000))
        # Message format: timestamp + method + path + body
        message = timestamp + method + path + body

        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )

        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp
        }

    def get_market(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch market data for a specific ticker."""
        path = f"/markets/{ticker}"
        headers = self._get_headers("GET", path)

        try:
            response = requests.get(self.BASE_URL + path, headers=headers,
                                    timeout=10)
            if response.status_code == 200:
                return response.json().get("market", {})
            return None
        except requests.exceptions.RequestException as e:
            print(f"Kalshi API Error: {e}")
            return None

    def get_event_markets(self, event_ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch all markets for a specific event."""
        path = f"/events/{event_ticker}"
        headers = self._get_headers("GET", path)

        try:
            response = requests.get(self.BASE_URL + path, headers=headers,
                                    timeout=10)
            if response.status_code == 200:
                return response.json()
            print(f"Kalshi Event Error: {response.status_code} - "
                  f"{response.text}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Kalshi API Exception: {e}")
            return None

    def list_markets(self, limit: int = 100, **kwargs) -> List[Dict]:
        """List active markets with optional filters.

        Supported kwargs: ticker, series_ticker, status, event_ticker.
        """
        path = "/markets"
        params = [f"limit={limit}"]
        for key, value in kwargs.items():
            if value:
                params.append(f"{key}={value}")

        query_string = "?" + "&".join(params)
        headers = self._get_headers("GET", path)

        try:
            response = requests.get(
                self.BASE_URL + path + query_string, headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("markets", [])
            print(f"Kalshi List Error: {response.status_code} - "
                  f"{response.text}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Kalshi API Exception: {e}")
            return []

    def get_exchange_status(self) -> Optional[Dict[str, Any]]:
        """Check if the exchange and trading are active."""
        path = "/exchange/status"
        headers = self._get_headers("GET", path)

        try:
            response = requests.get(self.BASE_URL + path, headers=headers,
                                    timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException as e:
            print(f"Kalshi Status Error: {e}")
            return None

    def get_series(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch details for a specific series."""
        path = f"/series/{ticker}"
        headers = self._get_headers("GET", path)

        try:
            response = requests.get(self.BASE_URL + path, headers=headers,
                                    timeout=10)
            if response.status_code == 200:
                return response.json().get("series")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Kalshi Series Error: {e}")
            return None

    def list_series(self, limit: int = 100) -> List[Dict]:
        """List all available series."""
        path = "/series"
        params = f"?limit={limit}"
        headers = self._get_headers("GET", path)

        try:
            response = requests.get(
                self.BASE_URL + path + params, headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("series", [])
            return []
        except requests.exceptions.RequestException as e:
            print(f"Kalshi Series List Error: {e}")
            return []
