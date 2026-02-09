"""
Hyperparameter Manager — Explore/Exploit for binary options.
==============================================================
Manages the tuneable parameters of the ClawdBot+Yoshi+Kalshi
pipeline and adjusts them via an explore/exploit strategy.

Exploration (10% of cycles): random perturbation of params.
Exploitation (90% of cycles): use the best-performing params.

Optimized parameters:
  - min_edge_pct:        minimum edge % to trade (3-20)
  - min_ev_cents:        minimum EV per contract in cents (0.5-10)
  - kelly_fraction:      fraction of full Kelly to use (0.05-0.5)
  - max_contracts:       max contracts per trade (1-25)
  - confidence_threshold: min LLM confidence to trade (0.3-0.9)
  - stop_loss_pct:       stop loss % for crypto positions (0.01-0.10)
  - position_size_pct:   % of capital per trade (0.01-0.10)
  - forecast_weight:     weight of ClawdBot vs Yoshi (0.3-0.9)
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Default Hyperparameters ───────────────────────────────────
@dataclass
class HyperParams:
    """Tuneable pipeline hyperparameters."""
    # Kalshi-specific
    min_edge_pct: float = 5.0           # minimum edge % to enter
    min_ev_cents: float = 2.0           # minimum EV cents/contract
    kelly_fraction: float = 0.25        # fraction of full Kelly
    max_contracts: int = 10             # max contracts per trade
    confidence_threshold: float = 0.5   # min LLM confidence for BUY

    # Crypto-specific (ClawdBot backtest)
    stop_loss_pct: float = 0.03         # 3% stop loss
    take_profit_pct: float = 0.06       # 6% take profit
    position_size_pct: float = 0.02     # 2% of capital per trade
    min_forecast_edge: float = 0.04     # min |dir_prob - 0.5| to trade

    # Fusion weights (how much to trust ClawdBot vs Yoshi)
    forecast_weight: float = 0.6        # ClawdBot weight (Yoshi = 1 - this)

    # Risk management
    max_daily_trades: int = 20
    max_daily_loss_cents: float = 500   # max $5 daily loss in cents
    cooldown_after_loss: int = 2        # cycles to skip after a loss

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HyperParams":
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ── Parameter Bounds ──────────────────────────────────────────
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "min_edge_pct": (2.0, 25.0),
    "min_ev_cents": (0.5, 15.0),
    "kelly_fraction": (0.05, 0.50),
    "max_contracts": (1, 30),
    "confidence_threshold": (0.2, 0.95),
    "stop_loss_pct": (0.01, 0.10),
    "take_profit_pct": (0.02, 0.20),
    "position_size_pct": (0.005, 0.10),
    "min_forecast_edge": (0.01, 0.15),
    "forecast_weight": (0.2, 0.9),
    "max_daily_trades": (5, 50),
    "max_daily_loss_cents": (100, 2000),
    "cooldown_after_loss": (0, 10),
}


# ── Snapshot: params + their performance ──────────────────────
@dataclass
class ParamSnapshot:
    """A set of hyperparams with their measured performance."""
    params: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    cycle_count: int = 0
    is_exploration: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ParamSnapshot":
        return cls(
            params=d.get("params", {}),
            performance=d.get("performance", {}),
            timestamp=d.get("timestamp", ""),
            cycle_count=d.get("cycle_count", 0),
            is_exploration=d.get("is_exploration", False),
        )

    @property
    def score(self) -> float:
        """Composite optimization score (higher = better).

        Combines Brier score (lower=better), hit rate, PnL, and Sharpe.
        """
        perf = self.performance
        if not perf:
            return 0.0

        brier = perf.get("brier_score", 0.25)
        hit_rate = perf.get("hit_rate", 0.5)
        pnl = perf.get("total_pnl_cents", 0)
        n = perf.get("n_resolved", 0)

        if n < 5:
            return 0.0  # Not enough data

        # Brier: 0 is perfect, 0.25 is random. Invert to score.
        brier_score = max(0, (0.25 - brier) / 0.25)

        # Hit rate bonus (0.5 is random, 1.0 is perfect)
        hr_score = max(0, (hit_rate - 0.5) * 2)

        # PnL (normalize to rough dollars)
        pnl_score = max(-1, min(1, pnl / 1000.0))

        # Sample size bonus (more data = more confident)
        n_bonus = min(1.0, n / 50.0)

        return (
            brier_score * 0.30
            + hr_score * 0.25
            + pnl_score * 0.30
            + n_bonus * 0.15
        )


# ── Hyperparameter Manager ───────────────────────────────────
class HyperParamManager:
    """
    Manages hyperparameter exploration and exploitation.

    Usage:
        mgr = HyperParamManager()
        params = mgr.get_current_params()
        # ... run pipeline with params ...
        mgr.record_performance(params, metrics)
        mgr.step()  # advance to next cycle (maybe explore)
    """

    HISTORY_FILE = "param_history.json"
    EXPLORE_RATE = 0.10  # 10% exploration

    def __init__(
        self,
        data_dir: str = "data/ralph",
        explore_rate: float = None,
        seed: int = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.explore_rate = explore_rate or self.EXPLORE_RATE
        self.rng = random.Random(seed)

        # State
        self._current: HyperParams = HyperParams()
        self._best: Optional[ParamSnapshot] = None
        self._history: List[ParamSnapshot] = []
        self._cycle: int = 0
        self._exploring: bool = False

        self._load()

    def _load(self):
        """Load param history from JSON."""
        hist_file = self.data_dir / self.HISTORY_FILE
        if not hist_file.exists():
            return
        try:
            with open(hist_file) as f:
                data = json.load(f)
            self._cycle = data.get("cycle", 0)
            self._history = [
                ParamSnapshot.from_dict(s) for s in data.get("history", [])
            ]
            if data.get("current"):
                self._current = HyperParams.from_dict(data["current"])
            if data.get("best"):
                self._best = ParamSnapshot.from_dict(data["best"])
        except Exception:
            pass

    def _save(self):
        """Save param history to JSON."""
        hist_file = self.data_dir / self.HISTORY_FILE
        try:
            data = {
                "cycle": self._cycle,
                "current": self._current.to_dict(),
                "best": self._best.to_dict() if self._best else None,
                "history": [s.to_dict() for s in self._history[-200:]],
            }
            with open(hist_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────
    def get_current_params(self) -> HyperParams:
        """Return the current hyperparameters."""
        return self._current

    def get_best_params(self) -> Optional[HyperParams]:
        """Return the historically best-performing params."""
        if self._best:
            return HyperParams.from_dict(self._best.params)
        return self._current

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def is_exploring(self) -> bool:
        return self._exploring

    def record_performance(
        self,
        params: HyperParams,
        metrics: Dict[str, Any],
    ):
        """Record the performance of a param set.

        Args:
            params: The hyperparams that were used.
            metrics: Performance metrics (from PredictionTracker.compute_metrics).
        """
        from datetime import datetime, timezone
        snap = ParamSnapshot(
            params=params.to_dict(),
            performance=metrics,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cycle_count=self._cycle,
            is_exploration=self._exploring,
        )

        self._history.append(snap)

        # Update best if this is better
        if self._best is None or snap.score > self._best.score:
            self._best = snap

        self._save()

    def step(self) -> HyperParams:
        """Advance one cycle: decide explore vs exploit and return params.

        Returns:
            HyperParams for the next cycle.
        """
        self._cycle += 1

        if self.rng.random() < self.explore_rate:
            # EXPLORE: perturb the current best
            self._exploring = True
            base = self._best.params if self._best else self._current.to_dict()
            perturbed = self._perturb(base)
            self._current = HyperParams.from_dict(perturbed)
        else:
            # EXPLOIT: use the best known params
            self._exploring = False
            if self._best:
                self._current = HyperParams.from_dict(self._best.params)

        self._save()
        return self._current

    def _perturb(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly perturb one or two parameters."""
        result = dict(params)
        # Pick 1-3 params to perturb
        n_perturb = self.rng.randint(1, min(3, len(PARAM_BOUNDS)))
        keys = self.rng.sample(list(PARAM_BOUNDS.keys()), n_perturb)

        for key in keys:
            if key not in result:
                continue
            lo, hi = PARAM_BOUNDS[key]
            current = result[key]

            # Gaussian perturbation (10-30% of range)
            range_size = hi - lo
            sigma = range_size * self.rng.uniform(0.10, 0.30)
            new_val = current + self.rng.gauss(0, sigma)

            # Clamp
            new_val = max(lo, min(hi, new_val))

            # Integer params stay integer
            if isinstance(current, int):
                new_val = int(round(new_val))

            result[key] = new_val

        return result

    # ── Reporting ─────────────────────────────────────────
    def summary(self) -> Dict[str, Any]:
        """Return a summary of the learning state."""
        return {
            "cycle": self._cycle,
            "is_exploring": self._exploring,
            "current_params": self._current.to_dict(),
            "best_score": round(self._best.score, 4) if self._best else 0,
            "best_params": self._best.params if self._best else {},
            "history_size": len(self._history),
            "explore_rate": self.explore_rate,
        }

    def print_summary(self):
        """Print a human-readable summary."""
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"  RALPH WIGGUM — Learning State")
        print(f"{'='*50}")
        print(f"  Cycle:          {s['cycle']}")
        print(f"  Mode:           {'EXPLORE' if s['is_exploring'] else 'EXPLOIT'}")
        print(f"  Best Score:     {s['best_score']:.4f}")
        print(f"  History Size:   {s['history_size']}")
        print(f"  Explore Rate:   {s['explore_rate']:.0%}")

        if s['best_params']:
            print(f"\n  Best Params:")
            for k, v in sorted(s['best_params'].items()):
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        print(f"{'='*50}")
