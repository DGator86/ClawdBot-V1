"""
Forecaster Evaluation Framework
=================================
Walk-forward backtesting, calibration analysis, and per-regime metrics.

Evaluation protocol:
  1. Walk-forward (expanding window): train on [0..t], test on t+1
  2. Embargo: 1-bar gap between train and test to prevent leakage
  3. Per-regime and per-vol-bucket breakdown
  4. No lookahead: all features computed on data available at time t

Metrics by target type:
  - Direction:    hit rate, MCC, F1
  - Magnitude:    MAE, MSE, IC (rank correlation)
  - Distribution: pinball loss (quantile), CRPS
  - Tail risk:    Brier score for |y|>k
  - Calibration:  binned reliability (expected vs observed)

Usage:
    from scripts.forecaster.evaluation import WalkForwardEvaluator
    evaluator = WalkForwardEvaluator(forecaster, bars)
    metrics = evaluator.run()
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .schemas import (
    Bar, MarketSnapshot, ModuleOutput, PredictionTargets,
    Regime, EvalMetrics,
)
from .engine import Forecaster, ForecastResult


# ═══════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def hit_rate(predicted_dirs: list[float], actual_returns: list[float]) -> float:
    """
    Direction hit rate.
    predicted_dirs: direction_prob values (>0.5 = bullish)
    actual_returns: realized log-returns
    """
    if not predicted_dirs:
        return 0.0
    correct = sum(
        1 for p, a in zip(predicted_dirs, actual_returns)
        if (p > 0.5 and a > 0) or (p < 0.5 and a < 0) or (p == 0.5 and a == 0)
    )
    return correct / len(predicted_dirs)


def matthews_correlation(predicted_dirs: list[float],
                          actual_returns: list[float]) -> float:
    """Matthews Correlation Coefficient for binary classification."""
    if not predicted_dirs:
        return 0.0
    tp = fp = tn = fn = 0
    for p, a in zip(predicted_dirs, actual_returns):
        pred_up = p > 0.5
        actual_up = a > 0
        if pred_up and actual_up:
            tp += 1
        elif pred_up and not actual_up:
            fp += 1
        elif not pred_up and not actual_up:
            tn += 1
        else:
            fn += 1
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def pinball_loss(quantile_preds: list[float], actuals: list[float],
                  tau: float) -> float:
    """
    Pinball (quantile) loss.
    tau: quantile level (0.1, 0.5, 0.9)
    """
    if not quantile_preds:
        return 0.0
    total = 0.0
    for pred, actual in zip(quantile_preds, actuals):
        error = actual - pred
        if error >= 0:
            total += tau * error
        else:
            total += (tau - 1) * error
    return total / len(quantile_preds)


def crps_empirical(quantile_10: list[float], quantile_50: list[float],
                    quantile_90: list[float], actuals: list[float]) -> float:
    """
    Approximate CRPS from 3 quantile predictions.
    Uses the average pinball loss across quantiles as a proxy.
    """
    if not actuals:
        return 0.0
    pb10 = pinball_loss(quantile_10, actuals, 0.10)
    pb50 = pinball_loss(quantile_50, actuals, 0.50)
    pb90 = pinball_loss(quantile_90, actuals, 0.90)
    return (pb10 + pb50 + pb90) / 3


def brier_score(predicted_probs: list[float], actuals: list[bool]) -> float:
    """Brier score: mean squared error of probability predictions."""
    if not predicted_probs:
        return 0.0
    return sum(
        (p - (1.0 if a else 0.0)) ** 2
        for p, a in zip(predicted_probs, actuals)
    ) / len(predicted_probs)


def information_coefficient(predicted_returns: list[float],
                              actual_returns: list[float]) -> float:
    """
    Information Coefficient: Spearman rank correlation between
    predicted and actual returns.
    """
    if len(predicted_returns) < 3:
        return 0.0
    # Rank correlation
    pred_arr = np.array(predicted_returns)
    actual_arr = np.array(actual_returns)
    # Rank
    pred_ranks = np.argsort(np.argsort(pred_arr)).astype(float)
    actual_ranks = np.argsort(np.argsort(actual_arr)).astype(float)
    # Pearson of ranks
    n = len(pred_ranks)
    mean_p = np.mean(pred_ranks)
    mean_a = np.mean(actual_ranks)
    cov = np.sum((pred_ranks - mean_p) * (actual_ranks - mean_a))
    std_p = np.sqrt(np.sum((pred_ranks - mean_p) ** 2))
    std_a = np.sqrt(np.sum((actual_ranks - mean_a) ** 2))
    if std_p == 0 or std_a == 0:
        return 0.0
    return float(cov / (std_p * std_a))


def calibration_bins(predicted_probs: list[float], actual_outcomes: list[bool],
                      n_bins: int = 10) -> dict:
    """
    Compute calibration bins: for each probability bin,
    what is the expected vs observed frequency?
    """
    bins = {}
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        bin_preds = [p for p in predicted_probs if lo <= p < hi]
        bin_outcomes = [a for p, a in zip(predicted_probs, actual_outcomes)
                        if lo <= p < hi]
        if bin_preds:
            bins[f"{lo:.1f}-{hi:.1f}"] = {
                "count": len(bin_preds),
                "mean_predicted": sum(bin_preds) / len(bin_preds),
                "observed_rate": sum(1 for a in bin_outcomes if a) / len(bin_outcomes),
            }
    return bins


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD EVALUATOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class EvalRecord:
    """Single forecast + outcome pair — captures full pipeline output."""
    timestamp: float = 0.0
    predicted_direction_prob: float = 0.5
    predicted_return: float = 0.0
    predicted_vol: float = 0.0
    predicted_jump_prob: float = 0.0
    predicted_crash_prob: float = 0.0
    predicted_q10: float = 0.0
    predicted_q50: float = 0.0
    predicted_q90: float = 0.0
    actual_return: float = 0.0
    actual_direction_up: bool = False
    actual_jump: bool = False     # |return| > 2*vol
    actual_crash: bool = False    # return < -3*vol
    regime: str = "range"
    vol_bucket: str = "normal"

    # ── Monte Carlo outputs (full pipeline) ────────────────
    mc_ran: bool = False
    mc_var_95: float = 0.0
    mc_var_99: float = 0.0
    mc_cvar_95: float = 0.0
    mc_cvar_99: float = 0.0
    mc_mean_price: float = 0.0
    mc_median_price: float = 0.0
    mc_p5_price: float = 0.0
    mc_p95_price: float = 0.0
    predicted_price: float = 0.0
    actual_price: float = 0.0
    current_price: float = 0.0

    # ── Barrier outputs (Kalshi integration) ────────────────
    barrier_strike: float = 0.0
    barrier_above_prob: float = 0.5
    actual_breached_above: bool = False
    actual_max_price: float = 0.0       # highest price during horizon


class WalkForwardEvaluator:
    """
    Walk-forward backtester for the ensemble forecaster.

    Process:
    1. Start with min_history bars
    2. At each step, forecast next horizon
    3. Record prediction vs actual
    4. Expand window by step_size bars
    5. Repeat until end of data

    Args:
        forecaster: Forecaster instance
        bars: List of hourly Bar objects (full history)
        horizon_hours: Forecast horizon in hours
        min_history: Minimum bars before first forecast
        step_size: Bars to advance between forecasts
        embargo: Bars to skip between train and test (leakage prevention)
    """

    def __init__(self,
                 forecaster: Optional[Forecaster] = None,
                 bars: Optional[list[Bar]] = None,
                 horizon_hours: float = 24,
                 min_history: int = 168,   # 1 week of hourly bars
                 step_size: int = 24,       # forecast every 24 hours
                 embargo: int = 1,
                 barrier_strike: Optional[float] = None):
        # Default: full pipeline with MC enabled (same as production)
        self.forecaster = forecaster or Forecaster(enable_mc=True)
        self.bars = bars or []
        self.horizon_hours = horizon_hours
        self.min_history = min_history
        self.step_size = step_size
        self.embargo = embargo
        self.barrier_strike = barrier_strike
        self.records: list[EvalRecord] = []

    def run(self, symbol: str = "BTCUSDT",
            max_forecasts: int = 500,
            verbose: bool = True) -> EvalMetrics:
        """
        Run walk-forward evaluation.

        Returns:
            EvalMetrics with all computed performance numbers.
        """
        if len(self.bars) < self.min_history + int(self.horizon_hours) + self.embargo:
            if verbose:
                print(f"Not enough bars: {len(self.bars)} < "
                      f"{self.min_history + int(self.horizon_hours) + self.embargo}")
            return EvalMetrics()

        self.records = []
        n_forecasts = 0
        t0 = time.time()

        end_idx = len(self.bars) - int(self.horizon_hours) - self.embargo
        idx = self.min_history

        while idx < end_idx and n_forecasts < max_forecasts:
            # Build snapshot from available history (up to idx)
            history = self.bars[:idx]
            snap = MarketSnapshot(
                symbol=symbol,
                bars_1h=history,
                timestamp=history[-1].timestamp if history else 0,
            )

            # ── Compute barrier strike for this step ──────
            # Auto-derive from current price if not set explicitly
            bar_strike = self.barrier_strike
            current_price = self.bars[idx - 1].close
            if bar_strike is None and current_price > 0:
                # Use nearest round number above current price as
                # a realistic Kalshi-style barrier
                magnitude = 10 ** max(0, int(math.log10(current_price)) - 1)
                bar_strike = math.ceil(current_price / magnitude) * magnitude

            # Forecast — FULL pipeline (all 12 modules + MC)
            try:
                result = self.forecaster.forecast_from_snapshot(
                    snap, self.horizon_hours, barrier_strike=bar_strike
                )
            except Exception as e:
                if verbose:
                    print(f"  Forecast failed at idx={idx}: {e}")
                idx += self.step_size
                continue

            # Compute actual outcome
            future_idx = idx + self.embargo + int(self.horizon_hours)
            if future_idx >= len(self.bars):
                break

            actual_price = self.bars[future_idx].close
            if current_price <= 0 or actual_price <= 0:
                idx += self.step_size
                continue

            actual_return = math.log(actual_price / current_price)

            # Actual max price during horizon (for barrier evaluation)
            horizon_slice = self.bars[idx:future_idx + 1]
            actual_max_price = max(b.high for b in horizon_slice) if horizon_slice else actual_price
            actual_breached = (actual_max_price >= bar_strike) if bar_strike else False

            # Compute vol for threshold-based metrics
            recent_rets = []
            for i in range(max(1, idx - 24), idx):
                if self.bars[i].close > 0 and self.bars[i-1].close > 0:
                    recent_rets.append(
                        math.log(self.bars[i].close / self.bars[i-1].close)
                    )
            recent_vol = float(np.std(recent_rets)) if len(recent_rets) >= 5 else 0.02
            horizon_vol = recent_vol * math.sqrt(self.horizon_hours)

            record = EvalRecord(
                timestamp=history[-1].timestamp,
                predicted_direction_prob=result.targets.direction_prob,
                predicted_return=result.targets.expected_return,
                predicted_vol=result.targets.volatility_forecast,
                predicted_jump_prob=result.targets.jump_prob,
                predicted_crash_prob=result.targets.crash_prob,
                predicted_q10=result.targets.quantile_10,
                predicted_q50=result.targets.quantile_50,
                predicted_q90=result.targets.quantile_90,
                actual_return=actual_return,
                actual_direction_up=(actual_return > 0),
                actual_jump=(abs(actual_return) > 2 * horizon_vol),
                actual_crash=(actual_return < -3 * horizon_vol),
                regime=result.regime,
                vol_bucket=self._vol_bucket(horizon_vol),
                # ── MC outputs ──────────────────────────────
                mc_ran=bool(result.mc_summary),
                mc_var_95=result.var_95,
                mc_var_99=result.var_99,
                mc_cvar_95=result.cvar_95,
                mc_cvar_99=result.cvar_99,
                mc_mean_price=result.mc_summary.get("mc_mean_price", 0),
                mc_median_price=result.mc_summary.get("mc_median_price", 0),
                mc_p5_price=result.mc_summary.get("mc_p5_price", 0),
                mc_p95_price=result.mc_summary.get("mc_p95_price", 0),
                predicted_price=result.predicted_price,
                actual_price=actual_price,
                current_price=current_price,
                # ── Barrier outputs ─────────────────────────
                barrier_strike=bar_strike or 0.0,
                barrier_above_prob=result.barrier_above_prob,
                actual_breached_above=actual_breached,
                actual_max_price=actual_max_price,
            )
            self.records.append(record)
            n_forecasts += 1

            # ── Feed outcome to meta-learner for walk-forward GBM ──
            # The meta-learner accumulates (features, actual_return)
            # pairs and retrains its GBM periodically.  We extract
            # the numeric features from all base module outputs stored
            # in the result, which is exactly the feature set the
            # meta-learner uses for prediction.
            try:
                feat_dict = {}
                for mod_name, mod_data in result.module_outputs.items():
                    if mod_name in ("meta_learner",):
                        continue
                    if isinstance(mod_data, dict):
                        for k, v in mod_data.items():
                            if isinstance(v, (int, float)):
                                feat_dict[f"{mod_name}__{k}"] = float(v)
                if feat_dict:
                    self.forecaster.meta_learner.record_outcome(
                        feat_dict, actual_return)
            except (AttributeError, TypeError):
                pass  # forecaster may not expose meta_learner directly

            mc_tag = " [MC]" if record.mc_ran else ""
            if verbose and n_forecasts % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {n_forecasts} forecasts in {elapsed:.1f}s{mc_tag} "
                      f"(ret: {actual_return:+.4f}, "
                      f"pred: {result.targets.expected_return:+.4f}, "
                      f"regime: {result.regime})")

            idx += self.step_size

        if verbose:
            print(f"\n  Walk-forward complete: {n_forecasts} forecasts "
                  f"in {time.time()-t0:.1f}s")

        return self.compute_metrics()

    def compute_metrics(self) -> EvalMetrics:
        """Compute all evaluation metrics from recorded results."""
        if not self.records:
            return EvalMetrics()

        metrics = EvalMetrics()

        # Extract arrays
        pred_dirs = [r.predicted_direction_prob for r in self.records]
        actual_rets = [r.actual_return for r in self.records]
        pred_rets = [r.predicted_return for r in self.records]
        pred_q10 = [r.predicted_q10 for r in self.records]
        pred_q50 = [r.predicted_q50 for r in self.records]
        pred_q90 = [r.predicted_q90 for r in self.records]
        pred_jumps = [r.predicted_jump_prob for r in self.records]
        pred_crashes = [r.predicted_crash_prob for r in self.records]
        actual_jumps = [r.actual_jump for r in self.records]
        actual_crashes = [r.actual_crash for r in self.records]
        actual_dirs = [r.actual_direction_up for r in self.records]

        # ── Direction metrics ─────────────────────────────
        metrics.hit_rate = hit_rate(pred_dirs, actual_rets)
        metrics.mcc = matthews_correlation(pred_dirs, actual_rets)

        # ── Distribution metrics ──────────────────────────
        metrics.pinball_loss_10 = pinball_loss(pred_q10, actual_rets, 0.10)
        metrics.pinball_loss_50 = pinball_loss(pred_q50, actual_rets, 0.50)
        metrics.pinball_loss_90 = pinball_loss(pred_q90, actual_rets, 0.90)
        metrics.crps = crps_empirical(pred_q10, pred_q50, pred_q90, actual_rets)

        # ── Tail risk metrics ─────────────────────────────
        metrics.brier_score_jump = brier_score(pred_jumps, actual_jumps)
        metrics.brier_score_crash = brier_score(pred_crashes, actual_crashes)

        # ── Return magnitude metrics ──────────────────────
        if pred_rets and actual_rets:
            errors = [p - a for p, a in zip(pred_rets, actual_rets)]
            metrics.mae = sum(abs(e) for e in errors) / len(errors)
            metrics.ic = information_coefficient(pred_rets, actual_rets)

        # ── Calibration ───────────────────────────────────
        metrics.calibration_bins = calibration_bins(pred_dirs, actual_dirs)

        # ══════════════════════════════════════════════════
        # MONTE CARLO METRICS (full pipeline scoring)
        # ══════════════════════════════════════════════════
        mc_records = [r for r in self.records if r.mc_ran]
        if mc_records:
            mc_m = metrics.mc_metrics  # shorthand
            mc_m["n_mc_forecasts"] = len(mc_records)

            # ── VaR accuracy: did actual loss exceed VaR? ──
            # VaR(95) should be breached ~5% of the time
            var95_breaches = sum(
                1 for r in mc_records
                if r.actual_return < r.mc_var_95
            )
            var99_breaches = sum(
                1 for r in mc_records
                if r.actual_return < r.mc_var_99
            )
            n_mc = len(mc_records)
            mc_m["var95_breach_rate"] = var95_breaches / n_mc
            mc_m["var99_breach_rate"] = var99_breaches / n_mc
            mc_m["var95_expected_rate"] = 0.05
            mc_m["var99_expected_rate"] = 0.01
            # How well-calibrated: ideal breach rate = alpha
            mc_m["var95_calibration_gap"] = abs(
                mc_m["var95_breach_rate"] - 0.05
            )
            mc_m["var99_calibration_gap"] = abs(
                mc_m["var99_breach_rate"] - 0.01
            )

            # ── CVaR accuracy: avg loss when VaR breached ──
            var95_breach_losses = [
                r.actual_return for r in mc_records
                if r.actual_return < r.mc_var_95
            ]
            if var95_breach_losses:
                actual_cvar95 = sum(var95_breach_losses) / len(var95_breach_losses)
                pred_cvars = [
                    r.mc_cvar_95 for r in mc_records
                    if r.actual_return < r.mc_var_95
                ]
                mc_m["cvar95_actual"] = actual_cvar95
                mc_m["cvar95_predicted_avg"] = (
                    sum(pred_cvars) / len(pred_cvars) if pred_cvars else 0
                )
                mc_m["cvar95_mae"] = abs(
                    mc_m["cvar95_actual"] - mc_m["cvar95_predicted_avg"]
                )

            # ── MC price distribution accuracy ──────────────
            # Check if actual prices fall within predicted envelopes
            in_5_95 = sum(
                1 for r in mc_records
                if r.mc_p5_price > 0 and r.mc_p5_price <= r.actual_price <= r.mc_p95_price
            )
            mc_m["p5_p95_coverage"] = in_5_95 / n_mc
            mc_m["p5_p95_expected"] = 0.90
            mc_m["p5_p95_gap"] = abs(mc_m["p5_p95_coverage"] - 0.90)

            # MC predicted price MAE
            mc_price_errors = [
                abs(r.mc_mean_price - r.actual_price) / r.current_price
                for r in mc_records
                if r.current_price > 0 and r.mc_mean_price > 0
            ]
            if mc_price_errors:
                mc_m["mc_price_mae_pct"] = sum(mc_price_errors) / len(mc_price_errors)

            # ── Barrier probability (Kalshi) Brier score ────
            barrier_records = [
                r for r in mc_records if r.barrier_strike > 0
            ]
            if barrier_records:
                barrier_preds = [r.barrier_above_prob for r in barrier_records]
                barrier_actuals = [r.actual_breached_above for r in barrier_records]
                mc_m["barrier_brier_score"] = brier_score(
                    barrier_preds, barrier_actuals
                )
                mc_m["barrier_hit_rate"] = sum(
                    1 for p, a in zip(barrier_preds, barrier_actuals)
                    if (p > 0.5) == a
                ) / len(barrier_records)
                mc_m["n_barrier_forecasts"] = len(barrier_records)
                mc_m["barrier_calibration"] = calibration_bins(
                    barrier_preds, barrier_actuals, n_bins=5
                )

        # ── Per-regime breakdown ──────────────────────────
        regimes = set(r.regime for r in self.records)
        for regime in regimes:
            regime_records = [r for r in self.records if r.regime == regime]
            if len(regime_records) < 3:
                continue
            rp_dirs = [r.predicted_direction_prob for r in regime_records]
            ra_rets = [r.actual_return for r in regime_records]
            regime_info = {
                "count": len(regime_records),
                "hit_rate": hit_rate(rp_dirs, ra_rets),
                "mcc": matthews_correlation(rp_dirs, ra_rets),
                "ic": information_coefficient(
                    [r.predicted_return for r in regime_records], ra_rets
                ),
            }
            # Per-regime MC metrics
            mc_regime = [r for r in regime_records if r.mc_ran]
            if mc_regime:
                v95_breach = sum(1 for r in mc_regime if r.actual_return < r.mc_var_95)
                regime_info["mc_var95_breach_rate"] = v95_breach / len(mc_regime)
                b_regime = [r for r in mc_regime if r.barrier_strike > 0]
                if b_regime:
                    regime_info["barrier_hit_rate"] = sum(
                        1 for r in b_regime
                        if (r.barrier_above_prob > 0.5) == r.actual_breached_above
                    ) / len(b_regime)
            metrics.metrics_by_regime[regime] = regime_info
            metrics.hit_rate_by_regime[regime] = hit_rate(rp_dirs, ra_rets)

        # ── Per-vol-bucket breakdown ──────────────────────
        vol_buckets = set(r.vol_bucket for r in self.records)
        for bucket in vol_buckets:
            bucket_records = [r for r in self.records if r.vol_bucket == bucket]
            if len(bucket_records) < 3:
                continue
            bp_dirs = [r.predicted_direction_prob for r in bucket_records]
            ba_rets = [r.actual_return for r in bucket_records]
            metrics.metrics_by_vol_bucket[bucket] = {
                "count": len(bucket_records),
                "hit_rate": hit_rate(bp_dirs, ba_rets),
                "mcc": matthews_correlation(bp_dirs, ba_rets),
            }

        return metrics

    @staticmethod
    def _vol_bucket(vol: float) -> str:
        """Classify volatility into buckets."""
        if vol < 0.01:
            return "low"
        elif vol < 0.03:
            return "normal"
        elif vol < 0.06:
            return "high"
        else:
            return "extreme"


# ═══════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════

def print_eval_report(metrics: EvalMetrics, n_records: int = 0):
    """Pretty-print evaluation results — full pipeline including MC."""
    print(f"\n{'='*64}")
    print(f"  FULL-PIPELINE EVALUATION REPORT (12/12 modules)")
    print(f"{'='*64}")
    if n_records > 0:
        print(f"  Forecast samples:  {n_records}")
    print(f"\n  DIRECTION METRICS:")
    print(f"    Hit Rate:        {metrics.hit_rate*100:>8.2f}%")
    print(f"    MCC:             {metrics.mcc:>8.4f}")
    print(f"\n  DISTRIBUTION METRICS:")
    print(f"    Pinball Q10:     {metrics.pinball_loss_10:>8.6f}")
    print(f"    Pinball Q50:     {metrics.pinball_loss_50:>8.6f}")
    print(f"    Pinball Q90:     {metrics.pinball_loss_90:>8.6f}")
    print(f"    CRPS:            {metrics.crps:>8.6f}")
    print(f"\n  TAIL RISK METRICS:")
    print(f"    Brier (jump):    {metrics.brier_score_jump:>8.6f}")
    print(f"    Brier (crash):   {metrics.brier_score_crash:>8.6f}")
    print(f"\n  RETURN MAGNITUDE:")
    print(f"    MAE:             {metrics.mae:>8.6f}")
    print(f"    IC (rank corr):  {metrics.ic:>8.4f}")

    # ══════════════════════════════════════════════════════
    # MONTE CARLO METRICS
    # ══════════════════════════════════════════════════════
    mc_m = metrics.mc_metrics
    if mc_m:
        n_mc = mc_m.get("n_mc_forecasts", 0)
        print(f"\n{'─'*64}")
        print(f"  MONTE CARLO METRICS ({n_mc} forecasts with MC):")

        # VaR calibration
        if "var95_breach_rate" in mc_m:
            v95_rate = mc_m["var95_breach_rate"]
            v95_gap = mc_m["var95_calibration_gap"]
            v95_q = "GOOD" if v95_gap < 0.03 else "FAIR" if v95_gap < 0.08 else "POOR"
            print(f"    VaR(95%) breach:  {v95_rate*100:>6.1f}% (target: 5.0%) [{v95_q}]")
        if "var99_breach_rate" in mc_m:
            v99_rate = mc_m["var99_breach_rate"]
            v99_gap = mc_m["var99_calibration_gap"]
            v99_q = "GOOD" if v99_gap < 0.02 else "FAIR" if v99_gap < 0.05 else "POOR"
            print(f"    VaR(99%) breach:  {v99_rate*100:>6.1f}% (target: 1.0%) [{v99_q}]")

        # CVaR accuracy
        if "cvar95_mae" in mc_m:
            print(f"    CVaR(95%) MAE:    {mc_m['cvar95_mae']*100:>6.2f}%")
            print(f"    CVaR(95%) pred:   {mc_m['cvar95_predicted_avg']*100:>6.2f}%")
            print(f"    CVaR(95%) actual: {mc_m['cvar95_actual']*100:>6.2f}%")

        # Price envelope coverage
        if "p5_p95_coverage" in mc_m:
            cov = mc_m["p5_p95_coverage"]
            cov_gap = mc_m["p5_p95_gap"]
            cov_q = "GOOD" if cov_gap < 0.05 else "FAIR" if cov_gap < 0.15 else "POOR"
            print(f"    P5-P95 coverage:  {cov*100:>6.1f}% (target: 90.0%) [{cov_q}]")

        # MC price MAE
        if "mc_price_mae_pct" in mc_m:
            print(f"    MC Price MAE:     {mc_m['mc_price_mae_pct']*100:>6.2f}%")

        # Barrier (Kalshi)
        if "barrier_brier_score" in mc_m:
            print(f"\n  BARRIER / KALSHI METRICS ({mc_m.get('n_barrier_forecasts', 0)} contracts):")
            print(f"    Barrier Brier:    {mc_m['barrier_brier_score']:>8.6f}")
            print(f"    Barrier Hit Rate: {mc_m['barrier_hit_rate']*100:>6.1f}%")
            if "barrier_calibration" in mc_m:
                print(f"    Barrier Calibration:")
                for bin_name, info in sorted(mc_m["barrier_calibration"].items()):
                    pred = info["mean_predicted"]
                    obs = info["observed_rate"]
                    gap = abs(pred - obs)
                    q = "GOOD" if gap < 0.1 else "FAIR" if gap < 0.2 else "POOR"
                    print(f"      {bin_name:10s} n={info['count']:3d} "
                          f"pred={pred:.3f} obs={obs:.3f} [{q}]")

    if metrics.metrics_by_regime:
        print(f"\n  PER-REGIME BREAKDOWN:")
        for regime, rm in sorted(metrics.metrics_by_regime.items()):
            line = (f"    {regime:20s} n={rm['count']:3d} "
                    f"HR={rm['hit_rate']*100:.1f}% "
                    f"MCC={rm['mcc']:.3f} "
                    f"IC={rm['ic']:.3f}")
            if "mc_var95_breach_rate" in rm:
                line += f" VaR95br={rm['mc_var95_breach_rate']*100:.0f}%"
            if "barrier_hit_rate" in rm:
                line += f" BarrHR={rm['barrier_hit_rate']*100:.0f}%"
            print(line)

    if metrics.metrics_by_vol_bucket:
        print(f"\n  PER-VOL-BUCKET BREAKDOWN:")
        for bucket, bm in sorted(metrics.metrics_by_vol_bucket.items()):
            print(f"    {bucket:12s} n={bm['count']:3d} "
                  f"HR={bm['hit_rate']*100:.1f}% "
                  f"MCC={bm['mcc']:.3f}")

    if metrics.calibration_bins:
        print(f"\n  DIRECTION CALIBRATION (predicted vs observed):")
        for bin_name, info in sorted(metrics.calibration_bins.items()):
            pred = info["mean_predicted"]
            obs = info["observed_rate"]
            n = info["count"]
            gap = abs(pred - obs)
            quality = "GOOD" if gap < 0.1 else "FAIR" if gap < 0.2 else "POOR"
            print(f"    {bin_name:10s} n={n:3d} "
                  f"pred={pred:.3f} obs={obs:.3f} "
                  f"gap={gap:.3f} [{quality}]")

    print(f"{'='*64}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    """
    Run walk-forward evaluation on historical market data.
    Fetches bars and runs the full evaluation pipeline.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Walk-Forward Evaluation of 12-Paradigm Forecaster"
    )
    parser.add_argument("--symbol", "-s", default="BTCUSDT")
    parser.add_argument("--horizon", "-H", type=float, default=24.0)
    parser.add_argument("--bars", "-b", type=int, default=1000,
                        help="Number of historical bars to fetch")
    parser.add_argument("--step", type=int, default=24,
                        help="Bars between forecasts")
    parser.add_argument("--max-forecasts", type=int, default=100)
    parser.add_argument("--enable-mc", action="store_true", default=True,
                        help="Enable Monte Carlo (default: ON for full pipeline)")
    parser.add_argument("--no-mc", action="store_true",
                        help="Disable Monte Carlo (faster, but partial pipeline)")
    parser.add_argument("--mc-iterations", type=int, default=20_000,
                        help="MC iterations per forecast (default: 20000)")
    parser.add_argument("--barrier", type=float, default=None,
                        help="Fixed barrier strike (auto-derived if not set)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()
    if args.no_mc:
        args.enable_mc = False

    from .data import fetch_ohlcv_bars

    print(f"Fetching {args.bars} bars of {args.symbol} data...")
    bars, source = fetch_ohlcv_bars(args.symbol, args.bars)
    print(f"Got {len(bars)} bars from {source}")

    if len(bars) < 200:
        print("Not enough data for meaningful evaluation. Need 200+ bars.")
        return

    fc = Forecaster(
        enable_mc=args.enable_mc,
        mc_iterations=args.mc_iterations,
    )
    evaluator = WalkForwardEvaluator(
        forecaster=fc,
        bars=bars,
        horizon_hours=args.horizon,
        step_size=args.step,
        barrier_strike=args.barrier,
    )

    mc_label = f"MC ON ({args.mc_iterations:,} iter)" if args.enable_mc else "MC OFF"
    print(f"\nRunning walk-forward evaluation "
          f"(max {args.max_forecasts} forecasts, step={args.step}, {mc_label})...")
    metrics = evaluator.run(
        symbol=args.symbol,
        max_forecasts=args.max_forecasts,
    )

    if args.json:
        import json
        output = {
            "hit_rate": metrics.hit_rate,
            "mcc": metrics.mcc,
            "pinball_10": metrics.pinball_loss_10,
            "pinball_50": metrics.pinball_loss_50,
            "pinball_90": metrics.pinball_loss_90,
            "crps": metrics.crps,
            "brier_jump": metrics.brier_score_jump,
            "brier_crash": metrics.brier_score_crash,
            "mae": metrics.mae,
            "ic": metrics.ic,
            "calibration": metrics.calibration_bins,
            "monte_carlo": metrics.mc_metrics,
            "by_regime": metrics.metrics_by_regime,
            "by_vol_bucket": metrics.metrics_by_vol_bucket,
            "n_records": len(evaluator.records),
            "mc_enabled": args.enable_mc,
            "mc_iterations": args.mc_iterations if args.enable_mc else 0,
        }
        output_str = json.dumps(output, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_str)
            print(f"Results saved to {args.output}")
        else:
            print(output_str)
    else:
        print_eval_report(metrics, len(evaluator.records))


if __name__ == "__main__":
    main()
