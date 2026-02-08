"""
Unified Orchestrator — ClawdBot × Yoshi × Kalshi × Ralph.
============================================================
Wires all four systems into one continuous loop:

  ClawdBot (14-paradigm forecast)
  → Yoshi (KPCOFGS regime + walk-forward validation + scoring)
  → Kalshi (market scan + LLM value analysis + edge detection)
  → Ralph (record predictions, resolve outcomes, learn, optimize)

Each cycle:
  1. Ralph provides hyperparams (explore 10% / exploit 90%)
  2. ClawdBot runs 14-paradigm ensemble forecast
  3. Yoshi enriches with KPCOFGS 7-level regime classification
  4. Yoshi optionally validates via walk-forward with purge/embargo
  5. Kalshi scanner finds edge opportunities on binary markets
  6. LLM analyzer filters noise from real edge
  7. Ralph records all predictions and auto-resolves past ones
  8. Ralph computes Brier / hit-rate / PnL / calibration metrics
  9. Metrics feed back to Ralph's hyperparameter optimizer
  10. Repeat

Single-command: `python3 scripts/kalshi-system.py`
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from gnosis.ralph.learner import RalphLearner, LearningConfig, CycleResult
from gnosis.ralph.hyperparams import HyperParams


# ── Orchestrator Config ───────────────────────────────────────
@dataclass
class OrchestratorConfig:
    """Configuration for the unified orchestrator."""
    # Symbols
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    kalshi_series: List[str] = field(default_factory=lambda: ["KXBTC", "KXETH"])

    # Forecast
    horizon_hours: float = 1.0        # 1h for Kalshi hourly markets
    bars_limit: int = 500             # bars to fetch (less = faster)

    # Kalshi scanner
    top_n: int = 5                    # top picks per scan
    use_llm: bool = True              # LLM value analysis (free tier)

    # Ralph learning
    learning: LearningConfig = field(default_factory=LearningConfig)

    # Cycle control
    cycle_interval_s: float = 60.0    # seconds between cycles
    max_cycles: int = 0               # 0 = unlimited

    # Pipeline modes
    enable_forecast: bool = True
    enable_kpcofgs: bool = True
    enable_validation: bool = False   # heavy; skip for speed
    enable_kalshi: bool = True
    enable_ralph: bool = True

    # Verbosity
    verbose: bool = True


# ── Orchestrator Result ───────────────────────────────────────
@dataclass
class OrchestratorResult:
    """Result of one orchestrator cycle."""
    cycle: int = 0
    timestamp: str = ""

    # Forecast
    forecast: Dict[str, Any] = field(default_factory=dict)
    kpcofgs: Dict[str, Any] = field(default_factory=dict)
    kpcofgs_regime: str = "range"

    # Kalshi
    scan_results: List[Dict] = field(default_factory=list)
    value_plays: List[Dict] = field(default_factory=list)
    buy_count: int = 0
    watch_count: int = 0

    # Ralph learning
    ralph_cycle: Optional[CycleResult] = None
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    # Meta
    elapsed_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "forecast": self.forecast,
            "kpcofgs": self.kpcofgs,
            "kpcofgs_regime": self.kpcofgs_regime,
            "scan_results": self.scan_results,
            "buy_count": self.buy_count,
            "watch_count": self.watch_count,
            "ralph": self.ralph_cycle.to_dict() if self.ralph_cycle else {},
            "hyperparams": self.hyperparams,
            "elapsed_ms": self.elapsed_ms,
            "errors": self.errors,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def print_report(self, file=None):
        """Print a phone-friendly report."""
        out = file or sys.stdout
        w = 55
        now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")

        print(f"\n{'='*w}", file=out)
        print(f"  CLAWDBOT × YOSHI × KALSHI — Cycle {self.cycle}", file=out)
        print(f"  {now_str}", file=out)
        print(f"{'='*w}", file=out)

        # Forecast
        fc = self.forecast
        if fc:
            sym = fc.get("symbol", "?")
            price = fc.get("current_price", 0)
            pred = fc.get("predicted_price", 0)
            conf = fc.get("confidence", 0)
            regime = fc.get("regime", "?")
            arrow = "↑" if fc.get("direction") == "up" else "↓" if fc.get("direction") == "down" else "↔"
            print(f"\n  Forecast: {sym} {arrow} ${pred:,.0f}", file=out)
            print(f"  Current:  ${price:,.0f} | Conf: {conf:.0%} | Regime: {regime}", file=out)

        # KPCOFGS
        if self.kpcofgs:
            k = self.kpcofgs.get("K_label", "?")
            s = self.kpcofgs.get("S_label", "?")
            ent = self.kpcofgs.get("regime_entropy", 0)
            print(f"  KPCOFGS:  K={k} S={s} (entropy={ent:.2f})", file=out)

        # Kalshi
        if self.scan_results:
            print(f"\n  Kalshi:   {len(self.scan_results)} opps | "
                  f"{self.buy_count} BUY | {self.watch_count} WATCH", file=out)
            for vp in self.value_plays[:3]:
                scan = vp.get("scan", {})
                ticker = scan.get("ticker", "?")
                rec = vp.get("recommendation", "?")
                score = vp.get("value_score", 0)
                edge = scan.get("edge_pct", 0)
                emoji = {"BUY": "🟢", "WATCH": "🟡"}.get(rec, "🔴")
                print(f"    {emoji} {ticker}: {rec} (v={score:.1f} e={edge:+.1f}%)", file=out)

        # Ralph
        rc = self.ralph_cycle
        if rc:
            m = rc.metrics
            mode = "EXPLORE" if rc.mode == "explore" else "EXPLOIT"
            brier = m.get("brier_score")
            hr = m.get("hit_rate")
            pnl = m.get("total_pnl_cents", 0)
            print(f"\n  Ralph:    {mode} | cycle #{rc.cycle_number}", file=out)
            if brier is not None and hr is not None:
                print(f"  Learning: Brier={brier:.4f} HR={hr:.1%} PnL={pnl:+.0f}c", file=out)
            print(f"  Tracked:  {m.get('n_predictions', 0)} total | "
                  f"{m.get('n_resolved', 0)} resolved", file=out)

        if self.errors:
            print(f"\n  Errors:   {len(self.errors)}", file=out)

        print(f"\n  Elapsed:  {self.elapsed_ms:.0f}ms", file=out)
        print(f"{'='*w}", file=out)


# ── Unified Orchestrator ─────────────────────────────────────
class UnifiedOrchestrator:
    """
    Wires ClawdBot, Yoshi, Kalshi, and Ralph into one loop.

    Usage:
        orch = UnifiedOrchestrator()
        result = orch.run_cycle()
        result.print_report()

        # Or loop:
        orch.run_loop()
    """

    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()

        # Initialize Ralph
        self.ralph = RalphLearner(config=self.config.learning)

        self._cycle_count = 0
        self._last_forecast = None

    def run_cycle(self) -> OrchestratorResult:
        """Run one complete cycle of the pipeline."""
        t0 = time.time()
        self._cycle_count += 1
        result = OrchestratorResult(
            cycle=self._cycle_count,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # 1. Get hyperparams from Ralph
        params = self.ralph.get_params()
        result.hyperparams = params.to_dict()

        if self.config.verbose:
            print(f"\n{'━'*55}")
            print(f"  CYCLE {self._cycle_count} — "
                  f"{'EXPLORE' if self.ralph.param_mgr.is_exploring else 'EXPLOIT'}")
            print(f"{'━'*55}")

        # 2. ClawdBot forecast
        forecast = {}
        kpcofgs = {}
        kpcofgs_regime = "range"

        if self.config.enable_forecast:
            forecast, kpcofgs, kpcofgs_regime = self._run_forecast(params)
            result.forecast = forecast
            result.kpcofgs = kpcofgs
            result.kpcofgs_regime = kpcofgs_regime

        # 3. Kalshi scan + analysis
        scan_results_raw = []
        value_plays_raw = []

        if self.config.enable_kalshi:
            scan_results_raw, value_plays_raw = self._run_kalshi(params, forecast)
            result.scan_results = [
                sr if isinstance(sr, dict) else sr.to_dict() for sr in scan_results_raw
            ]
            result.value_plays = [
                vp if isinstance(vp, dict) else vp.to_dict() for vp in value_plays_raw
            ]
            result.buy_count = sum(
                1 for vp in value_plays_raw
                if (vp.get("recommendation") if isinstance(vp, dict) else getattr(vp, "recommendation", "")) == "BUY"
            )
            result.watch_count = sum(
                1 for vp in value_plays_raw
                if (vp.get("recommendation") if isinstance(vp, dict) else getattr(vp, "recommendation", "")) == "WATCH"
            )

        # 4. Ralph learning cycle
        if self.config.enable_ralph:
            # Get current prices for auto-resolution
            current_prices = {}
            if forecast and forecast.get("current_price"):
                current_prices[forecast.get("symbol", "BTCUSDT")] = forecast["current_price"]

            # Convert to dicts for Ralph
            scan_dicts = [
                sr if isinstance(sr, dict) else sr.to_dict()
                for sr in scan_results_raw
            ]
            vp_dicts = [
                vp if isinstance(vp, dict) else vp.to_dict()
                for vp in value_plays_raw
            ]

            forecasts_list = [forecast] if forecast else []

            try:
                ralph_result = self.ralph.run_cycle(
                    forecasts=forecasts_list,
                    scan_results=scan_dicts,
                    value_plays=vp_dicts,
                    kpcofgs=kpcofgs,
                    current_prices=current_prices,
                )
                result.ralph_cycle = ralph_result
            except Exception as e:
                result.errors.append(f"Ralph cycle: {e}")

        result.elapsed_ms = round((time.time() - t0) * 1000, 1)
        return result

    # ── Forecast Step ─────────────────────────────────────
    def _run_forecast(
        self,
        params: HyperParams,
    ) -> tuple:
        """Run ClawdBot forecast + KPCOFGS enrichment.

        Returns:
            (forecast_dict, kpcofgs_dict, kpcofgs_regime)
        """
        forecast = {}
        kpcofgs = {}
        regime = "range"

        try:
            from gnosis.bridge import (
                snapshot_to_dataframe, classify_kpcofgs, kpcofgs_to_regime,
            )

            if self.config.verbose:
                print(f"\n  [Forecast] Running {self.config.symbols[0]}...")

            # Import forecaster components
            from scripts.forecaster.engine import Forecaster
            from scripts.forecaster.data import fetch_market_snapshot

            snap = fetch_market_snapshot(
                self.config.symbols[0],
                bars_limit=self.config.bars_limit,
            )

            fc = Forecaster(
                mc_iterations=10_000,  # reduced for speed
                mc_steps=24,
                enable_mc=True,
            )
            fc_result = fc.forecast_from_snapshot(
                snap,
                horizon_hours=self.config.horizon_hours,
            )

            forecast = {
                "symbol": fc_result.symbol,
                "current_price": fc_result.current_price,
                "predicted_price": fc_result.predicted_price,
                "direction": fc_result.direction,
                "confidence": fc_result.confidence,
                "volatility": fc_result.volatility,
                "regime": fc_result.regime,
                "var_95": fc_result.mc_summary.get("var_95", 0),
                "horizon_hours": self.config.horizon_hours,
                "gate_decision": fc_result.module_outputs.get("regime_gate", {}),
            }
            self._last_forecast = forecast

            if self.config.verbose:
                arrow = "↑" if forecast["direction"] == "up" else "↓"
                print(f"    ${fc_result.current_price:,.0f} → "
                      f"${fc_result.predicted_price:,.0f} {arrow} "
                      f"({fc_result.confidence:.0%} conf)")

            # KPCOFGS enrichment
            if self.config.enable_kpcofgs:
                bars_df = snapshot_to_dataframe(snap.bars_1h)
                if not bars_df.empty:
                    _, kpcofgs = classify_kpcofgs(bars_df)
                    regime = kpcofgs_to_regime(kpcofgs)
                    if self.config.verbose:
                        print(f"    KPCOFGS: K={kpcofgs.get('K_label', '?')} "
                              f"S={kpcofgs.get('S_label', '?')} → {regime}")

        except Exception as e:
            if self.config.verbose:
                print(f"    Forecast error: {e}")

        return forecast, kpcofgs, regime

    # ── Kalshi Step ───────────────────────────────────────
    def _run_kalshi(
        self,
        params: HyperParams,
        forecast: Dict[str, Any],
    ) -> tuple:
        """Run Kalshi scan + LLM analysis.

        Returns:
            (scan_results, value_plays)
        """
        scan_results = []
        value_plays = []

        try:
            from gnosis.kalshi.scanner import KalshiScanner, KalshiAPIClient, _load_env_files
            from gnosis.kalshi.analyzer import KalshiAnalyzer

            _load_env_files()

            if self.config.verbose:
                print(f"\n  [Kalshi] Scanning {', '.join(self.config.kalshi_series)}...")

            client = KalshiAPIClient()

            # Apply Ralph's hyperparams
            scanner = KalshiScanner(
                client=client,
                min_edge_pct=params.min_edge_pct,
                min_ev_cents=params.min_ev_cents,
                max_contracts=params.max_contracts,
            )

            # Get current prices for model prob
            current_prices = {}
            if forecast and forecast.get("current_price"):
                sym = forecast.get("symbol", "BTCUSDT")
                current_prices[sym] = forecast["current_price"]
                # Map to Kalshi series
                for series in self.config.kalshi_series:
                    from gnosis.kalshi.scanner import _series_to_symbol
                    if _series_to_symbol(series) == sym:
                        current_prices[series] = forecast["current_price"]

            scan_results = scanner.scan(
                series=self.config.kalshi_series,
                top_n=self.config.top_n,
                current_prices=current_prices,
            )

            if self.config.verbose:
                print(f"    Found {len(scan_results)} opportunities")

            # LLM analysis
            if scan_results and self.config.use_llm:
                analyzer = KalshiAnalyzer()
                value_plays = analyzer.analyze(scan_results)

                buy_count = sum(1 for vp in value_plays if vp.recommendation == "BUY")
                watch_count = sum(1 for vp in value_plays if vp.recommendation == "WATCH")

                if self.config.verbose:
                    print(f"    LLM: {buy_count} BUY, {watch_count} WATCH")

        except Exception as e:
            if self.config.verbose:
                print(f"    Kalshi error: {e}")

        return scan_results, value_plays

    # ── Loop ──────────────────────────────────────────────
    def run_loop(
        self,
        max_cycles: int = None,
        interval_s: float = None,
        on_cycle: callable = None,
    ):
        """Run the continuous orchestration loop.

        Args:
            max_cycles: Override config max_cycles
            interval_s: Override config cycle_interval_s
            on_cycle: Optional callback(OrchestratorResult) after each cycle
        """
        max_c = max_cycles or self.config.max_cycles
        interval = interval_s or self.config.cycle_interval_s

        print(f"\n{'='*55}")
        print(f"  UNIFIED SYSTEM — ClawdBot × Yoshi × Kalshi × Ralph")
        print(f"  Interval: {interval}s | Max cycles: {max_c or 'unlimited'}")
        print(f"  Symbols: {self.config.symbols}")
        print(f"  Kalshi: {self.config.kalshi_series}")
        print(f"{'='*55}")

        cycle = 0
        try:
            while True:
                cycle += 1
                if max_c and cycle > max_c:
                    break

                result = self.run_cycle()

                if self.config.verbose:
                    result.print_report()

                if on_cycle:
                    on_cycle(result)

                # Sleep until next cycle
                if max_c and cycle >= max_c:
                    break

                remaining = interval - (result.elapsed_ms / 1000)
                if remaining > 0:
                    if self.config.verbose:
                        print(f"\n  Next cycle in {remaining:.0f}s...")
                    time.sleep(remaining)

        except KeyboardInterrupt:
            print(f"\n\n  System stopped after {cycle} cycles.")
            self.ralph.print_summary()
