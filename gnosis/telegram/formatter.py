"""
Message Formatter — Phone-friendly Telegram messages.
=======================================================
Converts OrchestratorResult, PipelineResult, and learning
summaries into compact, readable Telegram messages.

Uses Telegram MarkdownV2 formatting for clean mobile display.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _esc(text: str) -> str:
    """Escape special MarkdownV2 characters."""
    # Telegram MarkdownV2 requires escaping these characters
    for ch in r"_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


def _esc_code(text: str) -> str:
    """Escape only backticks for code blocks."""
    return text.replace("`", "'")


class MessageFormatter:
    """Formats trading system data for Telegram display."""

    # ── Orchestrator Cycle Report ────────────────────────
    @staticmethod
    def cycle_report(result) -> str:
        """Format an OrchestratorResult for Telegram.

        Args:
            result: OrchestratorResult (or dict from .to_dict())

        Returns:
            Telegram-formatted string (MarkdownV2)
        """
        d = result if isinstance(result, dict) else result.to_dict()
        lines = []

        now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
        lines.append(f"*ClawdBot × Kalshi* — Cycle {d.get('cycle', '?')}")
        lines.append(f"_{_esc(now_str)}_")

        # Forecast
        fc = d.get("forecast", {})
        if fc and fc.get("current_price"):
            sym = _esc(fc.get("symbol", "?"))
            price = fc.get("current_price", 0)
            pred = fc.get("predicted_price", 0)
            conf = fc.get("confidence", 0)
            regime = _esc(fc.get("regime", "?"))
            arrow = "↑" if fc.get("direction") == "up" else "↓" if fc.get("direction") == "down" else "↔"
            lines.append("")
            lines.append(f"📊 *Forecast*: {sym} {_esc(arrow)} {_esc(f'${pred:,.0f}')}")
            lines.append(f"  Now: {_esc(f'${price:,.0f}')} \\| {_esc(f'{conf:.0%}')} conf \\| {regime}")

        # KPCOFGS
        kp = d.get("kpcofgs", {})
        if kp:
            k = _esc(kp.get("K_label", "?"))
            s = _esc(kp.get("S_label", "?"))
            ent = kp.get("regime_entropy", 0)
            lines.append(f"  KPCOFGS: K\\={k} S\\={s} \\(H\\={_esc(f'{ent:.2f}')}\\)")

        # Kalshi
        buy_count = d.get("buy_count", 0)
        watch_count = d.get("watch_count", 0)
        scan_count = len(d.get("scan_results", []))
        if scan_count > 0:
            lines.append("")
            lines.append(f"🎯 *Kalshi*: {scan_count} opps \\| {buy_count} BUY \\| {watch_count} WATCH")

        # Ralph
        ralph = d.get("ralph", {})
        if ralph:
            m = ralph.get("metrics", {})
            mode = "EXPLORE" if ralph.get("mode") == "explore" else "EXPLOIT"
            lines.append("")
            lines.append(f"🧠 *Ralph*: {_esc(mode)} \\#{ ralph.get('cycle_number', '?')}")
            n_pred = m.get("n_predictions", 0)
            n_res = m.get("n_resolved", 0)
            lines.append(f"  Tracked: {n_pred} \\| Resolved: {n_res}")
            brier = m.get("brier_score")
            hr = m.get("hit_rate")
            if brier is not None and hr is not None:
                lines.append(f"  Brier: {_esc(f'{brier:.4f}')} \\| HR: {_esc(f'{hr:.1%}')}")

        elapsed = d.get("elapsed_ms", 0)
        lines.append(f"\n⏱ {_esc(f'{elapsed:.0f}')}ms")

        return "\n".join(lines)

    # ── BUY Alert ────────────────────────────────────────
    @staticmethod
    def buy_alert(value_play: dict) -> str:
        """Format a BUY recommendation as an urgent alert.

        Args:
            value_play: ValuePlay dict (from .to_dict())

        Returns:
            Telegram-formatted alert string
        """
        scan = value_play.get("scan", {})
        ticker = _esc(scan.get("ticker", "?"))
        side = scan.get("side", "?").upper()
        cost = scan.get("cost_cents", 0)
        edge = scan.get("edge_pct", 0)
        ev = scan.get("ev_cents", 0)
        score = value_play.get("value_score", 0)
        risk = _esc(value_play.get("risk_level", "?"))
        size = value_play.get("suggested_size", 0)
        max_loss = value_play.get("max_loss", 0)
        reasoning = _esc(value_play.get("reasoning", ""))
        expiry = scan.get("minutes_to_expiry")

        lines = [
            f"🟢 *BUY SIGNAL*",
            f"",
            f"*{ticker}*",
            f"  {_esc(side)} @ {cost}c",
            f"  Edge: {_esc(f'{edge:+.1f}')}% \\| EV: {_esc(f'{ev:+.1f}')}c",
            f"  Value: {_esc(f'{score:.1f}')}/10 \\| Risk: {risk}",
            f"  Size: {size} contracts \\({_esc(f'${max_loss:.2f}')} max loss\\)",
        ]

        if reasoning:
            lines.append(f"\n_{reasoning[:200]}_")

        if expiry is not None:
            lines.append(f"\n⏰ Expires: {_esc(f'{expiry:.0f}')} min")

        return "\n".join(lines)

    # ── Status Report ────────────────────────────────────
    @staticmethod
    def status_report(
        llm_info: dict = None,
        ralph_summary: dict = None,
        kalshi_status: dict = None,
    ) -> str:
        """Format system status for /status command.

        Returns:
            Telegram-formatted status string
        """
        lines = ["*System Status*", ""]

        # LLM
        if llm_info:
            env = _esc(llm_info.get("environment", "?"))
            model = _esc(llm_info.get("model", "?"))
            lines.append(f"🤖 *LLM*: {env} \\({model}\\)")

        # Ralph
        if ralph_summary:
            params = ralph_summary.get("params", {})
            tracker = ralph_summary.get("tracker", {})
            metrics = ralph_summary.get("metrics", {})

            cycle = params.get("cycle", 0)
            mode = "EXPLORE" if params.get("is_exploring") else "EXPLOIT"
            best = params.get("best_score", 0)

            lines.append(f"\n🧠 *Ralph Wiggum*")
            lines.append(f"  Cycle: {cycle} \\| {_esc(mode)}")
            lines.append(f"  Best score: {_esc(f'{best:.4f}')}")
            lines.append(f"  Predictions: {tracker.get('total', 0)}")
            lines.append(f"  Resolved: {tracker.get('resolved', 0)}")

            brier = metrics.get("brier_score")
            hr = metrics.get("hit_rate")
            pnl = metrics.get("total_pnl_cents", 0)
            if brier is not None:
                lines.append(f"\n📈 *Performance*")
                lines.append(f"  Brier: {_esc(f'{brier:.4f}')}")
                lines.append(f"  Hit Rate: {_esc(f'{hr:.1%}')}")
                lines.append(f"  PnL: {_esc(f'{pnl:+.0f}')}c")

        # Kalshi
        if kalshi_status:
            active = kalshi_status.get("exchange_active", False)
            trading = kalshi_status.get("trading_active", False)
            status_str = "OPEN" if active else "CLOSED"
            trading_str = "ACTIVE" if trading else "INACTIVE"
            lines.append(f"\n🏛 *Kalshi*: {_esc(status_str)} \\| {_esc(trading_str)}")

        return "\n".join(lines)

    # ── Ralph Learning Report ────────────────────────────
    @staticmethod
    def ralph_report(summary: dict) -> str:
        """Format Ralph learning state for /ralph command."""
        params = summary.get("params", {})
        metrics = summary.get("metrics", {})
        tracker = summary.get("tracker", {})

        lines = ["🧠 *Ralph Wiggum — Learning State*", ""]

        cycle = params.get("cycle", 0)
        mode = "EXPLORE" if params.get("is_exploring") else "EXPLOIT"
        best = params.get("best_score", 0)
        hist = params.get("history_size", 0)

        lines.append(f"Cycle: {cycle}")
        lines.append(f"Mode: {_esc(mode)}")
        lines.append(f"Best score: {_esc(f'{best:.4f}')}")
        lines.append(f"History: {hist} snapshots")
        lines.append(f"Predictions: {tracker.get('total', 0)}")
        lines.append(f"Resolved: {tracker.get('resolved', 0)}")

        brier = metrics.get("brier_score")
        hr = metrics.get("hit_rate")
        if brier is not None and hr is not None:
            pnl = metrics.get("total_pnl_cents", 0)
            lines.append(f"\n📊 *Metrics*")
            lines.append(f"Brier: {_esc(f'{brier:.4f}')}")
            lines.append(f"Hit Rate: {_esc(f'{hr:.1%}')}")
            lines.append(f"PnL: {_esc(f'{pnl:+.0f}')}c")
            lines.append(f"Kalshi trades: {metrics.get('n_kalshi_trades', 0)}")

        # Best params
        bp = params.get("best_params", {})
        if bp:
            lines.append(f"\n⚙️ *Best Params*")
            for k in ["min_edge_pct", "min_ev_cents", "kelly_fraction",
                       "confidence_threshold", "forecast_weight"]:
                v = bp.get(k)
                if v is not None:
                    lines.append(f"  {_esc(k)}: {_esc(f'{v:.3f}' if isinstance(v, float) else str(v))}")

        return "\n".join(lines)

    # ── Help ─────────────────────────────────────────────
    @staticmethod
    def help_message() -> str:
        """Format the /help command response."""
        return (
            "*ClawdBot Trading System*\n"
            "\n"
            "Commands:\n"
            "/scan — Run a scan cycle now\n"
            "/status — System status\n"
            "/ralph — Ralph learning state\n"
            "/params — Current hyperparameters\n"
            "/help — This message\n"
            "\n"
            "_Alerts are sent automatically when BUY signals appear\\._"
        )

    # ── Error ────────────────────────────────────────────
    @staticmethod
    def error_message(error: str) -> str:
        """Format an error message."""
        return f"⚠️ *Error*\n{_esc(error[:500])}"

    # ── Plain text fallback ──────────────────────────────
    @staticmethod
    def cycle_report_plain(result) -> str:
        """Format cycle report as plain text (no markdown)."""
        d = result if isinstance(result, dict) else result.to_dict()
        lines = []

        now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
        lines.append(f"ClawdBot × Kalshi — Cycle {d.get('cycle', '?')}")
        lines.append(now_str)

        fc = d.get("forecast", {})
        if fc and fc.get("current_price"):
            arrow = "↑" if fc.get("direction") == "up" else "↓"
            lines.append(f"\nForecast: {fc.get('symbol','?')} {arrow} ${fc.get('predicted_price',0):,.0f}")
            lines.append(f"Now: ${fc.get('current_price',0):,.0f} | {fc.get('confidence',0):.0%} conf")

        buy_count = d.get("buy_count", 0)
        scan_count = len(d.get("scan_results", []))
        if scan_count:
            lines.append(f"\nKalshi: {scan_count} opps | {buy_count} BUY")

        for vp in d.get("value_plays", [])[:3]:
            scan = vp.get("scan", {})
            rec = vp.get("recommendation", "?")
            ticker = scan.get("ticker", "?")
            edge = scan.get("edge_pct", 0)
            lines.append(f"  {rec} {ticker} (edge {edge:+.1f}%)")

        ralph = d.get("ralph", {})
        if ralph:
            m = ralph.get("metrics", {})
            mode = "EXPLORE" if ralph.get("mode") == "explore" else "EXPLOIT"
            lines.append(f"\nRalph: {mode} #{ralph.get('cycle_number','?')}")

        lines.append(f"\n{d.get('elapsed_ms', 0):.0f}ms")
        return "\n".join(lines)

    @staticmethod
    def buy_alert_plain(value_play: dict) -> str:
        """Format BUY alert as plain text."""
        scan = value_play.get("scan", {})
        lines = [
            f"BUY SIGNAL",
            f"",
            f"{scan.get('ticker', '?')}",
            f"{scan.get('side','?').upper()} @ {scan.get('cost_cents',0)}c",
            f"Edge: {scan.get('edge_pct',0):+.1f}% | EV: {scan.get('ev_cents',0):+.1f}c",
            f"Value: {value_play.get('value_score',0):.1f}/10",
            f"Size: {value_play.get('suggested_size',0)} contracts",
        ]
        reasoning = value_play.get("reasoning", "")
        if reasoning:
            lines.append(f"\n{reasoning[:200]}")
        return "\n".join(lines)
