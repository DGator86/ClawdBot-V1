"""
Ralph Learner — Continuous Learning Loop.
==========================================
Orchestrates the prediction → resolve → score → optimize cycle.
This is the brain of Ralph Wiggum — it ties together the tracker
(records), hyperparams (optimization), and pipeline (execution).

The learner runs in a loop:
  1. Get current hyperparams (explore or exploit)
  2. Execute pipeline cycle (forecast + scan + trade)
  3. Record predictions
  4. Resolve past predictions that have settled
  5. Compute metrics (Brier, Kelly, calibration, PnL)
  6. Feed metrics back to hyperparameter optimizer
  7. Step to next cycle

The learner also produces a learning_report.json after each cycle
for monitoring.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from gnosis.ralph.tracker import PredictionTracker, PredictionRecord
from gnosis.ralph.hyperparams import HyperParams, HyperParamManager


# ── Learning Config ───────────────────────────────────────────
@dataclass
class LearningConfig:
    """Configuration for the Ralph learning loop."""
    data_dir: str = "data/ralph"
    explore_rate: float = 0.10
    min_predictions_for_optimize: int = 10  # need N resolved before optimizing
    report_every_n: int = 5                 # save learning report every N cycles
    auto_resolve_hours: float = 1.5         # auto-resolve forecasts after this many hours
    verbose: bool = True


# ── Cycle Result ──────────────────────────────────────────────
@dataclass
class CycleResult:
    """Output from one learning cycle."""
    cycle_number: int = 0
    mode: str = "exploit"                   # "explore" or "exploit"
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    predictions_recorded: int = 0
    predictions_resolved: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    elapsed_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number,
            "mode": self.mode,
            "hyperparams": self.hyperparams,
            "predictions_recorded": self.predictions_recorded,
            "predictions_resolved": self.predictions_resolved,
            "metrics": self.metrics,
            "best_score": self.best_score,
            "elapsed_ms": self.elapsed_ms,
            "errors": self.errors,
        }


# ── Ralph Learner ─────────────────────────────────────────────
class RalphLearner:
    """
    The continuous learning engine.

    Ties together:
      - PredictionTracker (records + resolution)
      - HyperParamManager (explore/exploit optimization)
      - Pipeline (whatever produces forecasts and trades)

    Usage:
        learner = RalphLearner()

        # Run one cycle
        result = learner.run_cycle(
            forecasts=[...],      # from ClawdBot
            scan_results=[...],   # from KalshiScanner
            value_plays=[...],    # from KalshiAnalyzer
            current_prices={"BTCUSDT": 69000},
        )

        # Or plug into the orchestrator loop
        for cycle in range(100):
            params = learner.get_params()
            # ... run pipeline with params ...
            learner.record_and_learn(predictions, metrics)
    """

    def __init__(self, config: LearningConfig = None):
        self.config = config or LearningConfig()

        # Core components
        self.tracker = PredictionTracker(data_dir=self.config.data_dir)
        self.param_mgr = HyperParamManager(
            data_dir=self.config.data_dir,
            explore_rate=self.config.explore_rate,
        )

        # Report file
        self.report_dir = Path(self.config.data_dir) / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ── Params ────────────────────────────────────────────
    def get_params(self) -> HyperParams:
        """Get current hyperparameters (after explore/exploit step)."""
        return self.param_mgr.get_current_params()

    def step_params(self) -> HyperParams:
        """Advance to next cycle and return new params."""
        return self.param_mgr.step()

    # ── Full Cycle ────────────────────────────────────────
    def run_cycle(
        self,
        forecasts: List[Dict[str, Any]] = None,
        scan_results: List[Dict[str, Any]] = None,
        value_plays: List[Dict[str, Any]] = None,
        kpcofgs: Dict[str, Any] = None,
        current_prices: Dict[str, float] = None,
    ) -> CycleResult:
        """Run one learning cycle.

        1. Step params (explore/exploit)
        2. Record new predictions
        3. Auto-resolve old predictions
        4. Compute metrics
        5. Feed back to optimizer

        Args:
            forecasts: New forecast dicts from ClawdBot/Yoshi
            scan_results: New Kalshi scan results
            value_plays: LLM-analyzed value plays
            kpcofgs: KPCOFGS regime data
            current_prices: Current prices for auto-resolution

        Returns:
            CycleResult with metrics and state
        """
        t0 = time.time()
        cycle = CycleResult()

        # 1. Step params
        params = self.step_params()
        cycle.cycle_number = self.param_mgr.cycle
        cycle.mode = "explore" if self.param_mgr.is_exploring else "exploit"
        cycle.hyperparams = params.to_dict()

        if self.config.verbose:
            print(f"\n[Ralph] Cycle {cycle.cycle_number} — "
                  f"{'EXPLORE' if cycle.mode == 'explore' else 'EXPLOIT'}")

        # 2. Record new predictions
        n_recorded = 0
        params_snap = params.to_dict()

        for fc in (forecasts or []):
            try:
                self.tracker.record_forecast(fc, kpcofgs, params_snap)
                n_recorded += 1
            except Exception as e:
                cycle.errors.append(f"record_forecast: {e}")

        for sr in (scan_results or []):
            # Match value_play if available
            vp = None
            if value_plays:
                ticker = sr.get("ticker", "")
                for v in value_plays:
                    if v.get("scan", {}).get("ticker") == ticker:
                        vp = v
                        break
            try:
                self.tracker.record_kalshi_trade(sr, vp, params_snap)
                n_recorded += 1
            except Exception as e:
                cycle.errors.append(f"record_kalshi: {e}")

        cycle.predictions_recorded = n_recorded

        # 3. Auto-resolve old predictions
        n_resolved = 0
        if current_prices:
            n_resolved = self._auto_resolve(current_prices)
        cycle.predictions_resolved = n_resolved

        # 4. Compute metrics
        metrics = self.tracker.compute_metrics()
        cycle.metrics = metrics

        # 5. Feed back to optimizer (if enough resolved)
        if metrics.get("n_resolved", 0) >= self.config.min_predictions_for_optimize:
            self.param_mgr.record_performance(params, metrics)
            cycle.best_score = (
                self.param_mgr._best.score if self.param_mgr._best else 0
            )
        else:
            cycle.best_score = (
                self.param_mgr._best.score if self.param_mgr._best else 0
            )

        cycle.elapsed_ms = round((time.time() - t0) * 1000, 1)

        # 6. Save report periodically
        if cycle.cycle_number % self.config.report_every_n == 0:
            self._save_report(cycle)

        if self.config.verbose:
            self._print_cycle(cycle)

        return cycle

    # ── Auto-Resolution ──────────────────────────────────
    def _auto_resolve(self, current_prices: Dict[str, float]) -> int:
        """Auto-resolve predictions whose horizon has elapsed.

        Args:
            current_prices: {symbol: current_price}

        Returns:
            Number of records resolved.
        """
        resolved_count = 0
        now = datetime.now(timezone.utc)

        for rec in self.tracker.get_unresolved():
            # Only auto-resolve price-based forecasts
            if rec.source != "clawdbot":
                continue

            # Check if horizon has elapsed
            try:
                pred_time = datetime.fromisoformat(rec.timestamp)
                if pred_time.tzinfo is None:
                    pred_time = pred_time.replace(tzinfo=timezone.utc)
                elapsed_hours = (now - pred_time).total_seconds() / 3600

                if elapsed_hours >= rec.horizon_hours * self.config.auto_resolve_hours:
                    price = current_prices.get(rec.symbol)
                    if price and price > 0:
                        self.tracker.resolve(rec.id, outcome_price=price)
                        resolved_count += 1
            except Exception:
                continue

        return resolved_count

    def resolve_kalshi(
        self,
        ticker: str,
        settled_yes: bool,
    ) -> int:
        """Resolve Kalshi predictions by ticker.

        Args:
            ticker: Kalshi contract ticker
            settled_yes: Whether YES won

        Returns:
            Number of records resolved
        """
        resolved = 0
        for rec in self.tracker.get_unresolved():
            if rec.kalshi_ticker == ticker:
                self.tracker.resolve(rec.id, kalshi_settled_yes=settled_yes)
                resolved += 1
        return resolved

    # ── Reporting ─────────────────────────────────────────
    def _save_report(self, cycle: CycleResult):
        """Save a learning report JSON."""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": cycle.to_dict(),
            "tracker_stats": {
                "total": self.tracker.total_count,
                "resolved": self.tracker.resolved_count,
            },
            "param_state": self.param_mgr.summary(),
        }
        report_file = self.report_dir / f"cycle_{cycle.cycle_number:05d}.json"
        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception:
            pass

        # Also save latest report
        latest = self.report_dir / "latest.json"
        try:
            with open(latest, "w") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception:
            pass

    def _print_cycle(self, cycle: CycleResult):
        """Print cycle summary."""
        m = cycle.metrics
        print(f"  Recorded: {cycle.predictions_recorded} | "
              f"Resolved: {cycle.predictions_resolved}")
        print(f"  Total tracked: {m.get('n_predictions', 0)} | "
              f"Total resolved: {m.get('n_resolved', 0)}")

        brier = m.get("brier_score")
        hr = m.get("hit_rate")
        pnl = m.get("total_pnl_cents", 0)
        if brier is not None and hr is not None:
            print(f"  Brier: {brier:.4f} | HR: {hr:.1%} | PnL: {pnl:+.0f}c")
        print(f"  Best score: {cycle.best_score:.4f} | "
              f"Elapsed: {cycle.elapsed_ms:.0f}ms")
        if cycle.errors:
            print(f"  Errors: {len(cycle.errors)}")

    # ── State Access ──────────────────────────────────────
    def get_metrics(self, source: str = None) -> Dict[str, Any]:
        """Get current aggregate metrics."""
        return self.tracker.compute_metrics(source)

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get full learning state summary."""
        return {
            "tracker": {
                "total": self.tracker.total_count,
                "resolved": self.tracker.resolved_count,
            },
            "params": self.param_mgr.summary(),
            "metrics": self.tracker.compute_metrics(),
        }

    def print_summary(self):
        """Print a combined summary."""
        self.param_mgr.print_summary()
        m = self.tracker.compute_metrics()
        if m.get("n_resolved", 0) > 0:
            print(f"\n  Performance (n={m['n_resolved']}):")
            print(f"    Brier Score:  {m.get('brier_score', '?')}")
            print(f"    Hit Rate:     {m.get('hit_rate', '?')}")
            print(f"    PnL (cents):  {m.get('total_pnl_cents', 0):+.0f}")
            print(f"    Kalshi Trades: {m.get('n_kalshi_trades', 0)}")
            cal = m.get("calibration", {})
            if cal.get("bins"):
                print(f"    Calibration:")
                for b in cal["bins"]:
                    if b["n"] > 0:
                        print(f"      {b['range']}: pred={b['avg_predicted']:.2f} "
                              f"actual={b['avg_actual']:.2f} "
                              f"err={b['calibration_error']:.3f} (n={b['n']})")
