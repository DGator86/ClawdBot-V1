#!/usr/bin/env python3
"""
Shared Kalshi API Client Module
================================
Standalone Kalshi V2 API client with RSA-PSS SHA-256 authentication.
Used by both kalshi-order.py and kalshi-edge-scanner.py.

Requires: cryptography (pip3 install cryptography)
Environment: KALSHI_KEY_ID, KALSHI_PRIVATE_KEY
"""

import base64
import json
import os
import socket
import time
from urllib import error as urlerror
from urllib import request


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
        except ImportError as err:
            raise ImportError(
                "cryptography package not installed. Run: pip3 install cryptography"
            ) from err

    @staticmethod
    def _fix_pem(raw: str) -> str:
        """
        Normalize a PEM key that may have been mangled by env var storage.
        Handles:
          - literal \\n instead of real newlines
          - single-line PEM with headers but no line breaks
          - raw base64 with NO headers at all (spaces instead of newlines)
          - raw base64 with no whitespace at all
        
        Tries both PKCS#1 (RSA PRIVATE KEY) and PKCS#8 (PRIVATE KEY) formats.
        """
        import re
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

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
                # Try PKCS#1 first
                lines = [body[i:i+64] for i in range(0, len(body), 64)]
                pkcs1_pem = "-----BEGIN RSA PRIVATE KEY-----\n" + "\n".join(lines) + "\n-----END RSA PRIVATE KEY-----"
                try:
                    load_pem_private_key(pkcs1_pem.encode(), password=None)
                    return pkcs1_pem
                except Exception:
                    # Try PKCS#8 format
                    pkcs8_pem = "-----BEGIN PRIVATE KEY-----\n" + "\n".join(lines) + "\n-----END PRIVATE KEY-----"
                    try:
                        load_pem_private_key(pkcs8_pem.encode(), password=None)
                        return pkcs8_pem
                    except Exception:
                        # Return PKCS#1 anyway and let the caller handle the error
                        return pkcs1_pem

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
        except urlerror.HTTPError as e:
            # Log HTTP errors with context
            error_msg = f"Kalshi API HTTP error ({method} {path}): {e.code} {e.reason}"
            try:
                error_body = e.read().decode()
                error_msg += f" - {error_body}"
            except Exception:
                pass
            print(f"ERROR: {error_msg}", flush=True)
            return None
        except urlerror.URLError as e:
            print(f"ERROR: Kalshi API URL error ({method} {path}): {e.reason}", flush=True)
            return None
        except socket.timeout:
            print(f"ERROR: Kalshi API timeout ({method} {path})", flush=True)
            return None
        except Exception as e:
            print(f"ERROR: Kalshi API error ({method} {path}): {e}", flush=True)
            return None

    def get_exchange_status(self) -> dict | None:
        """Get Kalshi exchange status."""
        return self._request("GET", "/exchange/status")

    def list_markets(self, limit: int = 200, **kwargs) -> list[dict]:
        """
        List markets with query parameters.
        
        Properly separates signing path from request path to comply with Kalshi signature requirements.
        """
        from urllib import parse as urlparse
        
        params = {"limit": str(limit)}
        for k, v in kwargs.items():
            params[k] = str(v)
        qs = urlparse.urlencode(params)
        
        # Sign only the clean path, but request with full query string
        clean_path = "/markets"
        full_path = f"{clean_path}?{qs}"
        
        # Override _request to handle split paths
        headers = self._sign("GET", clean_path, "")
        url = self.BASE_URL + full_path
        req = request.Request(url, headers=headers, method="GET")
        try:
            with request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
                if result and "markets" in result:
                    return result["markets"]
                return result if isinstance(result, list) else []
        except Exception as e:
            print(f"ERROR: Kalshi API error (GET {full_path}): {e}", flush=True)
            return []

    def get_market(self, ticker: str) -> dict | None:
        """Get details for a specific market by ticker."""
        return self._request("GET", f"/markets/{ticker}")

    def get_positions(self, limit: int = 100) -> dict | None:
        """Get current portfolio positions."""
        return self._request("GET", f"/portfolio/positions?limit={limit}")

    def list_orders(self, status: str = "resting") -> dict | None:
        """List orders with optional status filter."""
        return self._request("GET", f"/portfolio/orders?status={status}")

    def get_balance(self) -> dict | None:
        """Get account balance."""
        return self._request("GET", "/portfolio/balance")

    def cancel_order(self, order_id: str) -> dict | None:
        """Cancel an order by ID."""
        return self._request("DELETE", f"/portfolio/orders/{order_id}")
