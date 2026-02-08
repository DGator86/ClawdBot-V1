"""
Kalshi Trading Pipeline — Single-command scan → analyze → report.
================================================================
Orchestrates KalshiScanner and KalshiAnalyzer into a unified pipeline
that can be run with one command from the VPS.

Usage:
    from gnosis.kalshi.pipeline import KalshiPipeline
    pipeline = KalshiPipeline()
    result = pipeline.run()
    result.print_report()
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gnosis.kalshi.scanner import KalshiScanner, ScanResult, KalshiAPIClient, _load_env_files
from gnosis.kalshi.analyzer import KalshiAnalyzer, ValuePlay
from gnosis.reasoning.client import LLMClient, LLMConfig


# ── Pipeline Result ────────────────────────────────────────
@dataclass
class PipelineResult:
    """Complete output of a pipeline run."""
    scan_results: List[ScanResult] = field(default_factory=list)
    value_plays: List[ValuePlay] = field(default_factory=list)
    exchange_status: Dict[str, Any] = field(default_factory=dict)
    balance: Optional[Dict[str, Any]] = None
    positions: List[Dict] = field(default_factory=list)
    llm_environment: str = ""
    llm_model: str = ""
    elapsed_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def buy_plays(self) -> List[ValuePlay]:
        return [p for p in self.value_plays if p.recommendation == "BUY"]

    @property
    def watch_plays(self) -> List[ValuePlay]:
        return [p for p in self.value_plays if p.recommendation == "WATCH"]

    def to_dict(self) -> dict:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange_status": self.exchange_status,
            "llm": {"environment": self.llm_environment, "model": self.llm_model},
            "scan_count": len(self.scan_results),
            "buy_plays": [p.to_dict() for p in self.buy_plays],
            "watch_plays": [p.to_dict() for p in self.watch_plays],
            "all_plays": [p.to_dict() for p in self.value_plays],
            "balance": self.balance,
            "positions": self.positions,
            "elapsed_ms": self.elapsed_ms,
            "errors": self.errors,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def print_report(self, file=None):
        """Print a phone-friendly report."""
        out = file or sys.stdout
        w = 50

        print("=" * w, file=out)
        print("  KALSHI VALUE SCANNER REPORT", file=out)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"  {ts}", file=out)
        print("=" * w, file=out)

        # Exchange status
        ex = self.exchange_status
        print(f"\n  Exchange: {'OPEN' if ex.get('exchange_active') else 'CLOSED'}", file=out)
        print(f"  Trading:  {'ACTIVE' if ex.get('trading_active') else 'INACTIVE'}", file=out)

        # LLM status
        print(f"  LLM:      {self.llm_environment} ({self.llm_model})", file=out)

        # Balance
        if self.balance:
            bal = self.balance.get("balance", 0)
            if isinstance(bal, (int, float)):
                print(f"  Balance:  ${bal / 100:.2f}", file=out)

        # Scan summary
        print(f"\n  Scanned:  {len(self.scan_results)} contracts above threshold", file=out)
        print(f"  BUY:      {len(self.buy_plays)}", file=out)
        print(f"  WATCH:    {len(self.watch_plays)}", file=out)

        # Buy plays
        if self.buy_plays:
            print(f"\n{'─' * w}", file=out)
            print("  🟢 BUY RECOMMENDATIONS", file=out)
            print(f"{'─' * w}", file=out)
            for i, p in enumerate(self.buy_plays, 1):
                print(f"\n  #{i} {p.scan.ticker}", file=out)
                if p.scan.title:
                    print(f"     {p.scan.title}", file=out)
                print(f"     {p.scan.action.upper()} {p.scan.side.upper()} @ {p.scan.cost_cents}c", file=out)
                print(f"     Edge: {p.scan.edge_pct:+.1f}% | EV: {p.scan.ev_cents:+.1f}c", file=out)
                print(f"     Value: {p.value_score:.1f}/10 | Risk: {p.risk_level}", file=out)
                print(f"     Size: {p.suggested_size} contracts (${p.max_loss:.2f} max loss)", file=out)
                if p.reasoning:
                    # Wrap reasoning to ~45 chars
                    words = p.reasoning.split()
                    line = "     "
                    for w_ in words:
                        if len(line) + len(w_) > 48:
                            print(line, file=out)
                            line = "     " + w_
                        else:
                            line += " " + w_ if line.strip() else "     " + w_
                    if line.strip():
                        print(line, file=out)
                if p.scan.minutes_to_expiry is not None:
                    print(f"     Expires: {p.scan.minutes_to_expiry:.0f} min", file=out)

        # Watch plays
        if self.watch_plays:
            print(f"\n{'─' * w}", file=out)
            print("  🟡 WATCH LIST", file=out)
            print(f"{'─' * w}", file=out)
            for p in self.watch_plays:
                print(f"  {p.scan.ticker}: {p.scan.side.upper()} @ {p.scan.cost_cents}c "
                      f"| Edge {p.scan.edge_pct:+.1f}% | Value {p.value_score:.1f}/10", file=out)

        # Current positions
        if self.positions:
            print(f"\n{'─' * w}", file=out)
            print("  📊 CURRENT POSITIONS", file=out)
            print(f"{'─' * w}", file=out)
            for pos in self.positions:
                ticker = pos.get("ticker", "?")
                qty = pos.get("total_traded", pos.get("quantity", 0))
                side = "YES" if pos.get("yes_amount", 0) > 0 else "NO"
                print(f"  {ticker}: {qty}x {side}", file=out)

        # Errors
        if self.errors:
            print(f"\n{'─' * w}", file=out)
            print("  ⚠️  ERRORS", file=out)
            for err in self.errors:
                print(f"  - {err}", file=out)

        print(f"\n{'=' * w}", file=out)
        print(f"  Pipeline completed in {self.elapsed_ms:.0f}ms", file=out)
        print("=" * w, file=out)


# ── Pipeline ───────────────────────────────────────────────
class KalshiPipeline:
    """
    Complete Kalshi trading pipeline: scan → analyze → report.

    Usage:
        pipeline = KalshiPipeline()
        result = pipeline.run()
        result.print_report()  # Phone-friendly output
    """

    def __init__(
        self,
        series: List[str] = None,
        top_n: int = 5,
        min_edge_pct: float = 3.0,
        min_ev_cents: float = 1.0,
        show_balance: bool = True,
        show_positions: bool = True,
    ):
        self.series = series or ["KXBTC", "KXETH"]
        self.top_n = top_n
        self.min_edge_pct = min_edge_pct
        self.min_ev_cents = min_ev_cents
        self.show_balance = show_balance
        self.show_positions = show_positions

    def run(self) -> PipelineResult:
        """Execute the full pipeline."""
        t0 = time.time()
        result = PipelineResult()

        # Step 1: Initialize
        print("\n[1/4] Initializing Kalshi scanner...")
        _load_env_files()
        try:
            client = KalshiAPIClient()
            scanner = KalshiScanner(
                client=client,
                min_edge_pct=self.min_edge_pct,
                min_ev_cents=self.min_ev_cents,
            )
            print(f"  Connected (key: {client.key_id[:12]}...)")
        except Exception as e:
            result.errors.append(f"Kalshi init failed: {e}")
            result.elapsed_ms = (time.time() - t0) * 1000
            return result

        # Step 2: Exchange status + optional balance/positions
        print("\n[2/4] Checking exchange status...")
        status = client.get_exchange_status()
        result.exchange_status = status or {}
        if not status or not status.get("exchange_active"):
            print("  Exchange is CLOSED")
            result.elapsed_ms = (time.time() - t0) * 1000
            return result
        print(f"  Exchange: OPEN | Trading: {'ACTIVE' if status.get('trading_active') else 'INACTIVE'}")

        if self.show_balance:
            try:
                result.balance = client.get_balance()
                bal = result.balance.get("balance", 0) if result.balance else 0
                if isinstance(bal, (int, float)):
                    print(f"  Balance: ${bal / 100:.2f}")
            except Exception as e:
                result.errors.append(f"Balance check: {e}")

        if self.show_positions:
            try:
                result.positions = client.get_positions()
                if result.positions:
                    print(f"  Open positions: {len(result.positions)}")
            except Exception as e:
                result.errors.append(f"Positions check: {e}")

        # Step 3: Scan markets
        print(f"\n[3/4] Scanning {', '.join(self.series)} markets...")
        try:
            result.scan_results = scanner.scan(
                series=self.series,
                top_n=self.top_n,
            )
            print(f"  Found {len(result.scan_results)} opportunities above threshold")
            for r in result.scan_results:
                print(f"    {r.summary_line()}")
        except Exception as e:
            result.errors.append(f"Scan failed: {e}")
            result.elapsed_ms = (time.time() - t0) * 1000
            return result

        if not result.scan_results:
            print("  No contracts meet edge threshold")
            result.elapsed_ms = (time.time() - t0) * 1000
            return result

        # Step 4: LLM analysis
        print("\n[4/4] Running LLM value analysis...")
        try:
            llm_cfg = LLMConfig.from_yaml()
            result.llm_environment = llm_cfg._environment
            result.llm_model = llm_cfg.model
            print(f"  LLM: {llm_cfg._environment} ({llm_cfg.model})")

            analyzer = KalshiAnalyzer()
            result.value_plays = analyzer.analyze(result.scan_results)

            buy_count = len(result.buy_plays)
            watch_count = len(result.watch_plays)
            print(f"  Analysis complete: {buy_count} BUY, {watch_count} WATCH")
        except Exception as e:
            result.errors.append(f"LLM analysis failed: {e}")
            # Still return scan results even if LLM fails

        result.elapsed_ms = round((time.time() - t0) * 1000, 1)
        return result
