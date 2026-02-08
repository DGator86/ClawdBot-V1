"""
Prediction Tracker — Records predictions and resolves outcomes.
================================================================
Stores every ClawdBot/Yoshi forecast and Kalshi trade as a
PredictionRecord. When the market outcome becomes known (contract
settles, price horizon elapsed), the record is *resolved* and
Brier score, PnL, and calibration data are computed.

Persistence: JSON-lines file at `data/ralph/predictions.jsonl`.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Prediction Record ─────────────────────────────────────────
@dataclass
class PredictionRecord:
    """One prediction: what we said, what happened."""
    # Identity
    id: str = ""
    timestamp: str = ""             # ISO-8601 UTC
    source: str = ""                # "clawdbot", "yoshi", "kalshi_scanner"
    symbol: str = ""                # e.g. "BTCUSDT"

    # Prediction
    predicted_direction: str = ""   # "up" / "down" / "neutral"
    direction_prob: float = 0.5     # P(up)
    predicted_price: float = 0.0
    current_price: float = 0.0
    horizon_hours: float = 24.0
    regime: str = "range"
    kpcofgs_regime: str = ""

    # Kalshi-specific (optional)
    kalshi_ticker: str = ""
    kalshi_side: str = ""           # "yes" / "no"
    kalshi_cost_cents: int = 0
    kalshi_model_prob: float = 0.0
    kalshi_market_prob: float = 0.0
    kalshi_edge_pct: float = 0.0
    kalshi_ev_cents: float = 0.0
    kalshi_contracts: int = 0

    # Hyperparameters snapshot (for attribution)
    hyperparams_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Outcome (filled after resolution)
    resolved: bool = False
    outcome_price: float = 0.0
    outcome_direction: str = ""     # "up" / "down"
    actual_return: float = 0.0
    brier_score: float = 0.0        # (prob - outcome)^2
    pnl_cents: float = 0.0          # Kalshi-specific PnL
    resolved_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionRecord":
        known_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in known_keys}
        return cls(**filtered)


# ── Tracker ───────────────────────────────────────────────────
class PredictionTracker:
    """
    Stores predictions and resolves them against outcomes.

    Usage:
        tracker = PredictionTracker()
        # Record a forecast
        rec = tracker.record_forecast(forecast_dict, kpcofgs_dict, hyperparams)
        # Later, resolve with actual outcome
        tracker.resolve(rec.id, outcome_price=72000.0)
        # Get performance metrics
        metrics = tracker.compute_metrics()
    """

    DEFAULT_DATA_DIR = "data/ralph"
    PREDICTIONS_FILE = "predictions.jsonl"

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or self.DEFAULT_DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pred_file = self.data_dir / self.PREDICTIONS_FILE

        # In-memory index: id -> PredictionRecord
        self._records: Dict[str, PredictionRecord] = {}
        self._load()

    def _load(self):
        """Load existing records from JSONL file."""
        if not self.pred_file.exists():
            return
        try:
            with open(self.pred_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        rec = PredictionRecord.from_dict(d)
                        if rec.id:
                            self._records[rec.id] = rec
                    except Exception:
                        continue
        except Exception:
            pass

    def _save_record(self, rec: PredictionRecord):
        """Append a record to the JSONL file."""
        try:
            with open(self.pred_file, "a") as f:
                f.write(json.dumps(rec.to_dict(), default=str) + "\n")
        except Exception:
            pass

    def _rewrite_all(self):
        """Rewrite all records (used after resolution updates)."""
        try:
            with open(self.pred_file, "w") as f:
                for rec in self._records.values():
                    f.write(json.dumps(rec.to_dict(), default=str) + "\n")
        except Exception:
            pass

    # ── Recording ────────────────────────────────────────
    def record_forecast(
        self,
        forecast: Dict[str, Any],
        kpcofgs: Dict[str, Any] = None,
        hyperparams: Dict[str, Any] = None,
        source: str = "clawdbot",
    ) -> PredictionRecord:
        """Record a ClawdBot/Yoshi forecast."""
        now = datetime.now(timezone.utc)
        rec_id = f"{source}_{now.strftime('%Y%m%d_%H%M%S')}_{len(self._records)}"

        dir_prob = forecast.get("confidence", 0.5)
        direction = forecast.get("direction", "neutral")
        # Convert confidence + direction to a unified direction_prob
        if direction == "up":
            p_up = 0.5 + dir_prob * 0.5 if dir_prob <= 1 else dir_prob
        elif direction == "down":
            p_up = 0.5 - dir_prob * 0.5 if dir_prob <= 1 else 1.0 - dir_prob
        else:
            p_up = 0.5

        rec = PredictionRecord(
            id=rec_id,
            timestamp=now.isoformat(),
            source=source,
            symbol=forecast.get("symbol", "BTCUSDT"),
            predicted_direction=direction,
            direction_prob=round(p_up, 4),
            predicted_price=forecast.get("predicted_price", 0),
            current_price=forecast.get("current_price", 0),
            horizon_hours=forecast.get("horizon_hours", 24.0),
            regime=forecast.get("regime", "range"),
            kpcofgs_regime=(kpcofgs or {}).get("S_label", ""),
            hyperparams_snapshot=hyperparams or {},
        )
        self._records[rec.id] = rec
        self._save_record(rec)
        return rec

    def record_kalshi_trade(
        self,
        scan_result: Dict[str, Any],
        value_play: Dict[str, Any] = None,
        hyperparams: Dict[str, Any] = None,
    ) -> PredictionRecord:
        """Record a Kalshi trade/opportunity."""
        now = datetime.now(timezone.utc)
        rec_id = f"kalshi_{now.strftime('%Y%m%d_%H%M%S')}_{len(self._records)}"

        vp = value_play or {}
        rec = PredictionRecord(
            id=rec_id,
            timestamp=now.isoformat(),
            source="kalshi_scanner",
            symbol=scan_result.get("series", ""),
            kalshi_ticker=scan_result.get("ticker", ""),
            kalshi_side=scan_result.get("side", ""),
            kalshi_cost_cents=scan_result.get("cost_cents", 0),
            kalshi_model_prob=scan_result.get("model_prob", 0),
            kalshi_market_prob=scan_result.get("market_prob", 0),
            kalshi_edge_pct=scan_result.get("edge_pct", 0),
            kalshi_ev_cents=scan_result.get("ev_cents", 0),
            kalshi_contracts=vp.get("suggested_size", scan_result.get("suggested_contracts", scan_result.get("contracts", 1))),
            direction_prob=scan_result.get("model_prob", 0.5),
            hyperparams_snapshot=hyperparams or {},
        )
        self._records[rec.id] = rec
        self._save_record(rec)
        return rec

    # ── Resolution ───────────────────────────────────────
    def resolve(
        self,
        record_id: str,
        outcome_price: float = None,
        kalshi_settled_yes: bool = None,
    ) -> Optional[PredictionRecord]:
        """Resolve a prediction with the actual outcome.

        For price forecasts: provide outcome_price.
        For Kalshi trades: provide kalshi_settled_yes (True if YES won).
        """
        rec = self._records.get(record_id)
        if not rec or rec.resolved:
            return rec

        now = datetime.now(timezone.utc)
        rec.resolved = True
        rec.resolved_at = now.isoformat()

        if outcome_price is not None and rec.current_price > 0:
            # Price-based resolution
            rec.outcome_price = outcome_price
            actual_return = (outcome_price - rec.current_price) / rec.current_price
            rec.actual_return = round(actual_return, 6)
            rec.outcome_direction = "up" if actual_return > 0 else "down"

            # Brier score: (predicted_prob - actual_binary)^2
            actual_binary = 1.0 if actual_return > 0 else 0.0
            rec.brier_score = round((rec.direction_prob - actual_binary) ** 2, 6)

        elif kalshi_settled_yes is not None:
            # Kalshi binary resolution
            actual_binary = 1.0 if kalshi_settled_yes else 0.0
            rec.outcome_direction = "yes" if kalshi_settled_yes else "no"

            # Brier score
            if rec.kalshi_side == "yes":
                rec.brier_score = round((rec.kalshi_model_prob - actual_binary) ** 2, 6)
            else:
                rec.brier_score = round(((1 - rec.kalshi_model_prob) - (1 - actual_binary)) ** 2, 6)

            # PnL
            if kalshi_settled_yes and rec.kalshi_side == "yes":
                rec.pnl_cents = (100 - rec.kalshi_cost_cents) * rec.kalshi_contracts
            elif not kalshi_settled_yes and rec.kalshi_side == "no":
                rec.pnl_cents = (100 - rec.kalshi_cost_cents) * rec.kalshi_contracts
            else:
                rec.pnl_cents = -rec.kalshi_cost_cents * rec.kalshi_contracts

        self._records[rec.id] = rec
        self._rewrite_all()
        return rec

    # ── Querying ─────────────────────────────────────────
    def get_all(self) -> List[PredictionRecord]:
        return list(self._records.values())

    def get_resolved(self) -> List[PredictionRecord]:
        return [r for r in self._records.values() if r.resolved]

    def get_unresolved(self) -> List[PredictionRecord]:
        return [r for r in self._records.values() if not r.resolved]

    def get_recent(self, n: int = 50) -> List[PredictionRecord]:
        recs = sorted(self._records.values(), key=lambda r: r.timestamp, reverse=True)
        return recs[:n]

    @property
    def total_count(self) -> int:
        return len(self._records)

    @property
    def resolved_count(self) -> int:
        return len(self.get_resolved())

    # ── Performance Metrics ──────────────────────────────
    def compute_metrics(self, source: str = None) -> Dict[str, Any]:
        """Compute aggregate performance metrics.

        Args:
            source: Filter by source ("clawdbot", "kalshi_scanner", etc.)

        Returns:
            Dict with Brier score, hit rate, PnL, Kelly, calibration.
        """
        resolved = self.get_resolved()
        if source:
            resolved = [r for r in resolved if r.source == source]

        n = len(resolved)
        if n == 0:
            return {
                "n_predictions": self.total_count,
                "n_resolved": 0,
                "brier_score": None,
                "hit_rate": None,
                "total_pnl_cents": 0,
                "avg_edge_pct": None,
            }

        brier_scores = [r.brier_score for r in resolved]
        avg_brier = sum(brier_scores) / n

        # Hit rate (direction correct)
        hits = sum(
            1 for r in resolved
            if (r.outcome_direction == "up" and r.direction_prob > 0.5)
            or (r.outcome_direction == "down" and r.direction_prob < 0.5)
            or (r.outcome_direction == "yes" and r.kalshi_side == "yes")
            or (r.outcome_direction == "no" and r.kalshi_side == "no")
        )
        hit_rate = hits / n

        # PnL (Kalshi-specific)
        total_pnl = sum(r.pnl_cents for r in resolved if r.source == "kalshi_scanner")
        kalshi_recs = [r for r in resolved if r.source == "kalshi_scanner"]
        n_kalshi = len(kalshi_recs)

        # Average edge
        edges = [r.kalshi_edge_pct for r in resolved if r.kalshi_edge_pct > 0]
        avg_edge = sum(edges) / len(edges) if edges else 0

        # Calibration bins (simple: split prob into 5 bins)
        calibration = self._compute_calibration(resolved)

        return {
            "n_predictions": self.total_count,
            "n_resolved": n,
            "brier_score": round(avg_brier, 6),
            "hit_rate": round(hit_rate, 4),
            "total_pnl_cents": round(total_pnl, 2),
            "n_kalshi_trades": n_kalshi,
            "avg_edge_pct": round(avg_edge, 2),
            "calibration": calibration,
        }

    def _compute_calibration(self, records: List[PredictionRecord]) -> Dict[str, Any]:
        """Compute simple calibration metrics."""
        if not records:
            return {"n_bins": 0, "bins": []}

        bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        result_bins = []

        for lo, hi in bins:
            in_bin = [
                r for r in records
                if lo <= r.direction_prob < hi
            ]
            if not in_bin:
                result_bins.append({
                    "range": f"{lo:.1f}-{hi:.1f}",
                    "n": 0,
                    "avg_predicted": 0,
                    "avg_actual": 0,
                    "calibration_error": 0,
                })
                continue

            avg_pred = sum(r.direction_prob for r in in_bin) / len(in_bin)
            actual_up = sum(
                1 for r in in_bin
                if r.outcome_direction in ("up", "yes")
            ) / len(in_bin)

            result_bins.append({
                "range": f"{lo:.1f}-{hi:.1f}",
                "n": len(in_bin),
                "avg_predicted": round(avg_pred, 3),
                "avg_actual": round(actual_up, 3),
                "calibration_error": round(abs(avg_pred - actual_up), 3),
            })

        return {"n_bins": len(bins), "bins": result_bins}
