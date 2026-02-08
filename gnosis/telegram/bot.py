"""
Telegram Bot — Long-polling bot for the trading system.
=========================================================
Uses stdlib urllib (no extra deps) for reliability on VPS.
Runs the orchestrator in a background thread and sends
BUY alerts + cycle reports to the configured chat.

Architecture:
  Main thread:  Telegram long-polling (getUpdates)
  Worker thread: Orchestrator.run_loop() with callback

Commands:
  /scan    — Force a scan cycle now
  /status  — System status + Ralph learning state
  /ralph   — Detailed Ralph learning report
  /params  — Current hyperparameters
  /help    — Command list
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from urllib import request, error as urlerror, parse as urlparse

from gnosis.telegram.formatter import MessageFormatter


# ── Telegram API Client (stdlib only) ─────────────────────────
class TelegramAPI:
    """Minimal Telegram Bot API client using urllib."""

    BASE = "https://api.telegram.org"

    def __init__(self, token: str):
        self.token = token
        self.base_url = f"{self.BASE}/bot{token}"

    def _call(self, method: str, data: dict = None, timeout: int = 30) -> Optional[dict]:
        """Call a Telegram Bot API method."""
        url = f"{self.base_url}/{method}"
        body = json.dumps(data).encode() if data else None
        headers = {"Content-Type": "application/json"} if body else {}
        req = request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode())
                if result.get("ok"):
                    return result.get("result")
                return None
        except urlerror.HTTPError as e:
            try:
                err_body = e.read().decode()
                err_data = json.loads(err_body)
                # Silently handle "can't parse entities" by retrying plain
                if err_data.get("error_code") == 400:
                    return {"parse_error": True, "description": err_data.get("description", "")}
            except Exception:
                pass
            return None
        except Exception:
            return None

    def get_me(self) -> Optional[dict]:
        return self._call("getMe")

    def get_updates(self, offset: int = 0, timeout: int = 30) -> List[dict]:
        result = self._call("getUpdates", {
            "offset": offset,
            "timeout": timeout,
            "allowed_updates": ["message"],
        }, timeout=timeout + 5)
        return result if isinstance(result, list) else []

    def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str = "MarkdownV2",
        disable_preview: bool = True,
    ) -> Optional[dict]:
        """Send a message. Falls back to plain text if Markdown fails."""
        data = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": disable_preview,
        }
        if parse_mode:
            data["parse_mode"] = parse_mode

        result = self._call("sendMessage", data)

        # If markdown parse failed, retry as plain text
        if isinstance(result, dict) and result.get("parse_error"):
            # Strip markdown and retry
            plain = text
            for ch in r"*_`[]()~>#+-=|{}.!\\":
                plain = plain.replace(f"\\{ch}", ch).replace(ch, "")
            data["text"] = plain
            data.pop("parse_mode", None)
            result = self._call("sendMessage", data)

        return result


# ── Telegram Bot ──────────────────────────────────────────────
class TelegramBot:
    """
    Long-polling Telegram bot integrated with the trading system.

    Usage:
        bot = TelegramBot(token="...", chat_id="...")
        bot.run()  # blocking, handles commands + runs orchestrator

    Or programmatic:
        bot = TelegramBot(token="...", chat_id="...")
        bot.send_text("Hello!")
        bot.send_buy_alert(value_play_dict)
    """

    def __init__(
        self,
        token: str = None,
        chat_id: str = None,
        orchestrator=None,
        auto_scan_interval: float = 60.0,
        notify_on_buy: bool = True,
        notify_on_cycle: bool = False,  # only BUY alerts by default
    ):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.api = TelegramAPI(self.token) if self.token else None
        self.orchestrator = orchestrator
        self.auto_scan_interval = auto_scan_interval
        self.notify_on_buy = notify_on_buy
        self.notify_on_cycle = notify_on_cycle

        self._update_offset = 0
        self._running = False
        self._scan_thread: Optional[threading.Thread] = None
        self._last_cycle_result = None
        self._fmt = MessageFormatter()

    # ── Sending ───────────────────────────────────────────
    def send_text(self, text: str, parse_mode: str = "MarkdownV2") -> bool:
        """Send a text message to the configured chat."""
        if not self.api or not self.chat_id:
            print(f"[TG] Not configured (token={'set' if self.token else 'missing'}, "
                  f"chat_id={'set' if self.chat_id else 'missing'})")
            return False
        result = self.api.send_message(self.chat_id, text, parse_mode)
        return result is not None and not isinstance(result, dict)

    def send_plain(self, text: str) -> bool:
        """Send plain text (no markdown)."""
        if not self.api or not self.chat_id:
            return False
        result = self.api.send_message(self.chat_id, text, parse_mode="")
        return result is not None

    def send_cycle_report(self, result) -> bool:
        """Send a formatted cycle report."""
        try:
            msg = self._fmt.cycle_report(result)
            ok = self.send_text(msg)
            if not ok:
                # Fallback to plain text
                msg = self._fmt.cycle_report_plain(result)
                return self.send_plain(msg)
            return ok
        except Exception as e:
            return self.send_plain(f"Cycle {getattr(result, 'cycle', '?')} complete ({e})")

    def send_buy_alert(self, value_play: dict) -> bool:
        """Send a BUY signal alert."""
        try:
            msg = self._fmt.buy_alert(value_play)
            ok = self.send_text(msg)
            if not ok:
                msg = self._fmt.buy_alert_plain(value_play)
                return self.send_plain(msg)
            return ok
        except Exception as e:
            scan = value_play.get("scan", {})
            return self.send_plain(f"BUY: {scan.get('ticker','?')} ({e})")

    def send_status(self) -> bool:
        """Send system status."""
        try:
            llm_info = {}
            ralph_summary = {}
            kalshi_status = {}

            try:
                from gnosis.reasoning.client import LLMConfig
                cfg = LLMConfig.from_yaml()
                llm_info = {"environment": cfg._environment, "model": cfg.model}
            except Exception:
                pass

            if self.orchestrator:
                try:
                    ralph_summary = self.orchestrator.ralph.get_learning_summary()
                except Exception:
                    pass

            try:
                from gnosis.kalshi.scanner import KalshiAPIClient, _load_env_files
                _load_env_files()
                client = KalshiAPIClient()
                kalshi_status = client.get_exchange_status() or {}
            except Exception:
                pass

            msg = self._fmt.status_report(llm_info, ralph_summary, kalshi_status)
            ok = self.send_text(msg)
            if not ok:
                return self.send_plain(f"Status: LLM={llm_info} Ralph={ralph_summary}")
            return ok
        except Exception as e:
            return self.send_plain(f"Status error: {e}")

    def send_ralph_report(self) -> bool:
        """Send Ralph learning report."""
        try:
            if self.orchestrator:
                summary = self.orchestrator.ralph.get_learning_summary()
            else:
                from gnosis.ralph.learner import RalphLearner
                summary = RalphLearner().get_learning_summary()

            msg = self._fmt.ralph_report(summary)
            ok = self.send_text(msg)
            if not ok:
                return self.send_plain(f"Ralph: {json.dumps(summary, indent=2, default=str)[:1000]}")
            return ok
        except Exception as e:
            return self.send_plain(f"Ralph error: {e}")

    # ── Command Handling ──────────────────────────────────
    def _handle_command(self, text: str, chat_id: str):
        """Handle an incoming command."""
        cmd = text.strip().split()[0].lower().split("@")[0]  # strip @botname

        if cmd == "/start" or cmd == "/help":
            msg = self._fmt.help_message()
            self.api.send_message(chat_id, msg, "MarkdownV2")

        elif cmd == "/status":
            self.send_status()

        elif cmd == "/ralph":
            self.send_ralph_report()

        elif cmd == "/scan":
            self.api.send_message(chat_id, "🔍 Running scan\\.\\.\\.", "MarkdownV2")
            self._force_scan(chat_id)

        elif cmd == "/params":
            self._send_params(chat_id)

        else:
            self.api.send_message(chat_id, f"Unknown command: {text.split()[0]}\nTry /help", "")

    def _force_scan(self, chat_id: str):
        """Force a scan cycle and report results."""
        try:
            if self.orchestrator:
                result = self.orchestrator.run_cycle()
                self._last_cycle_result = result
                self.send_cycle_report(result)

                # Send BUY alerts
                for vp in result.to_dict().get("value_plays", []):
                    if vp.get("recommendation") == "BUY":
                        self.send_buy_alert(vp)
            else:
                self.api.send_message(chat_id, "No orchestrator configured", "")
        except Exception as e:
            self.api.send_message(chat_id, f"Scan error: {e}", "")

    def _send_params(self, chat_id: str):
        """Send current hyperparameters."""
        try:
            if self.orchestrator:
                params = self.orchestrator.ralph.get_params()
                d = params.to_dict()
            else:
                from gnosis.ralph.hyperparams import HyperParams
                d = HyperParams().to_dict()

            lines = ["Current Hyperparameters:\n"]
            for k, v in sorted(d.items()):
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")
            self.api.send_message(chat_id, "\n".join(lines), "")
        except Exception as e:
            self.api.send_message(chat_id, f"Params error: {e}", "")

    # ── Orchestrator Callback ─────────────────────────────
    def _on_cycle(self, result):
        """Called after each orchestrator cycle."""
        self._last_cycle_result = result
        d = result.to_dict() if hasattr(result, "to_dict") else result

        # Send BUY alerts
        if self.notify_on_buy:
            for vp in d.get("value_plays", []):
                if vp.get("recommendation") == "BUY":
                    try:
                        self.send_buy_alert(vp)
                    except Exception:
                        pass

        # Optionally send full cycle report
        if self.notify_on_cycle:
            try:
                self.send_cycle_report(result)
            except Exception:
                pass

    # ── Background Scanner ────────────────────────────────
    def _scan_loop(self):
        """Background thread: runs orchestrator continuously."""
        if not self.orchestrator:
            return

        try:
            self.orchestrator.run_loop(
                interval_s=self.auto_scan_interval,
                on_cycle=self._on_cycle,
            )
        except Exception as e:
            try:
                self.send_plain(f"Scanner stopped: {e}")
            except Exception:
                pass

    # ── Main Loop ─────────────────────────────────────────
    def run(self):
        """Start the bot (blocking). Handles commands + runs scanner."""
        if not self.api:
            print("[TG] ERROR: TELEGRAM_BOT_TOKEN not set")
            print("[TG] Set it in .env or pass token= to TelegramBot()")
            return

        # Verify token
        me = self.api.get_me()
        if not me:
            print("[TG] ERROR: Invalid bot token")
            return

        bot_name = me.get("username", "?")
        print(f"[TG] Bot started: @{bot_name}")

        if self.chat_id:
            print(f"[TG] Chat ID: {self.chat_id}")
            self.send_plain(f"ClawdBot online. Type /help for commands.")
        else:
            print("[TG] WARNING: No TELEGRAM_CHAT_ID set.")
            print("[TG] Send any message to the bot to discover your chat ID.")

        # Start background scanner
        if self.orchestrator:
            self._scan_thread = threading.Thread(
                target=self._scan_loop,
                daemon=True,
                name="orchestrator-scanner",
            )
            self._scan_thread.start()
            print(f"[TG] Scanner running (interval: {self.auto_scan_interval}s)")

        # Long-polling loop
        self._running = True
        print("[TG] Listening for commands...")

        try:
            while self._running:
                try:
                    updates = self.api.get_updates(
                        offset=self._update_offset,
                        timeout=30,
                    )
                    for update in updates:
                        self._update_offset = update.get("update_id", 0) + 1
                        msg = update.get("message", {})
                        text = msg.get("text", "")
                        from_chat = str(msg.get("chat", {}).get("id", ""))

                        # Auto-discover chat ID
                        if not self.chat_id and from_chat:
                            self.chat_id = from_chat
                            print(f"[TG] Chat ID discovered: {self.chat_id}")
                            self.api.send_message(
                                from_chat,
                                f"Chat ID set: {from_chat}\nAdd TELEGRAM_CHAT_ID={from_chat} to .env",
                                "",
                            )

                        if text.startswith("/"):
                            print(f"[TG] Command: {text} (from {from_chat})")
                            self._handle_command(text, from_chat)

                except Exception as e:
                    # Network hiccup — wait and retry
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\n[TG] Bot stopped.")
            self._running = False

    def stop(self):
        """Stop the bot."""
        self._running = False
