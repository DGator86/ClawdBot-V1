"""
Forecaster Diagnostic Suite (Ultimate Enhanced)
Honest assessment of model quality with auto-fix integration.

Diagnostics:
1. Is direction HR statistically better than coin flip?
2. Is the hybrid ML (LightGBM+GRU) helping vs baseline?
3. Where does the model work (regime/vol) and where does it fail?
4. Is the model calibrated (isotonic/Platt)?
5. Are features stable or just noise?
6. Regime gate effectiveness
7. Auto-fix recommendations with confidence

Ultimate-fix enhancements:
  - Hybrid ML comparison (LightGBM+GRU vs plain GBM)
  - Regime gate validation (blocked vs strong regimes)
  - Auto-calibration integration (isotonic + Platt)
  - Health monitoring with drift detection
  - Full auto-fix pipeline on FAIL verdict

Run:
    python3 -m scripts.forecaster.diagnose --bars 2000 --forecasts 75
    python3 -m scripts.forecaster.diagnose --bars 2000 --auto-fix
Forecaster Diagnostic Suite
Honest assessment of model quality. Answers:

1. Is direction HR statistically better than coin flip?
2. Is the GBM helping or hurting vs the baseline?
3. Where does the model work (regime/vol) and where does it fail?
4. Is the model calibrated or just hedging with wide intervals?
5. Are the features stable or is importance just noise?

Run:
    python3 -m scripts.forecaster.diagnose --bars 2000 --forecasts 75
"""
from __future__ import annotations
import math
import time
import json
import argparse
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

from .engine import Forecaster
from .evaluation import WalkForwardEvaluator, EvalRecord
from .data import fetch_ohlcv_bars
from .schemas import MarketSnapshot, Bar
from .auto_fix import AutoFixPipeline, CalibrationSuite, HealthMonitor, HealthStatus
from .regime_gate import RegimeGate, GateDecision, DEFAULT_REGIME_PROFILES


# ─── Data structures ─────────────────────────────────────────

@dataclass
class DiagnosticReport:
    """Full diagnostic output."""
    # Sample info
    n_bars: int = 0
    n_forecasts: int = 0
    date_range: str = ""
    elapsed_s: float = 0.0

    # A/B: baseline vs GBM
    baseline_hr: float = 0.0
    baseline_mcc: float = 0.0
    gbm_hr: float = 0.0
    gbm_mcc: float = 0.0
    gbm_helped: bool = False

    # Statistical significance
    hr_vs_coinflip_pvalue: float = 1.0
    hr_significant: bool = False  # p < 0.05
    direction_is_random: bool = True

    # Rolling HR (windows of 10)
    rolling_hrs: list[float] = field(default_factory=list)
    rolling_trend: str = "unknown"  # improving, degrading, flat, erratic

    # Per-regime
    regime_stats: dict = field(default_factory=dict)

    # Calibration
    calibration_bins: dict = field(default_factory=dict)
    calibration_quality: str = "unknown"

    # Predicted vs actual
    pred_actual_corr: float = 0.0  # Pearson correlation
    pred_actual_rank_corr: float = 0.0  # Spearman

    # GBM stability
    feature_importances: list[tuple[str, float]] = field(default_factory=list)

    # MC quality
    p5_p95_coverage: float = 0.0
    mc_price_mae_pct: float = 0.0

    # Ultimate-fix: enhanced diagnostics
    hybrid_ml_hr: float = 0.0
    hybrid_ml_mcc: float = 0.0
    hybrid_ml_helped: bool = False
    regime_gate_stats: dict = field(default_factory=dict)
    auto_fix_applied: bool = False
    auto_fix_report: dict = field(default_factory=dict)
    health_status: dict = field(default_factory=dict)
    calibration_ece: float = 0.0

    # Verdict
    verdict: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.floating):
                d[k] = float(v)
            elif isinstance(v, np.integer):
                d[k] = int(v)
            else:
                d[k] = v
        return d


# ─── Diagnostic functions ────────────────────────────────────

def binomial_test_pvalue(hits: int, n: int, p0: float = 0.5) -> float:
    """Two-sided binomial test: is hits/n different from p0?
    Uses normal approximation for n >= 20."""
    if n < 5:
        return 1.0
    # Normal approximation
    expected = n * p0
    std = math.sqrt(n * p0 * (1 - p0))
    if std == 0:
        return 1.0
    z = abs(hits - expected) / std
    # Two-sided p-value from z-score (no scipy needed)
    # Approximation: P(Z > z) ≈ erfc(z/sqrt(2))/2
    p_one_tail = 0.5 * math.erfc(z / math.sqrt(2))
    return 2 * p_one_tail


def rolling_hit_rate(records: list[EvalRecord], window: int = 10) -> list[float]:
    """Compute rolling direction HR over windows of size `window`."""
    hrs = []
    for i in range(0, len(records) - window + 1):
        chunk = records[i:i + window]
        correct = sum(1 for r in chunk
                      if (r.predicted_direction_prob > 0.5) == r.actual_direction_up)
        hrs.append(correct / window)
    return hrs


def classify_trend(values: list[float]) -> str:
    """Classify a series as improving, degrading, flat, or erratic."""
    if len(values) < 3:
        return "insufficient_data"
    arr = np.array(values)
    x = np.arange(len(arr))
    # Linear regression slope
    slope = np.polyfit(x, arr, 1)[0]
    std = float(np.std(arr))

    if abs(slope) < 0.005:
        return "flat"
    if std > 0.15:
        return "erratic"
    if slope > 0.005:
        return "improving"
    return "degrading"


def compute_calibration(records: list[EvalRecord]) -> dict:
    """Binned calibration analysis."""
    bins = {}
    for r in records:
        p = r.predicted_direction_prob
        if p < 0.3:
            b = "0.0-0.3"
        elif p < 0.4:
            b = "0.3-0.4"
        elif p < 0.5:
            b = "0.4-0.5"
        elif p < 0.6:
            b = "0.5-0.6"
        elif p < 0.7:
            b = "0.6-0.7"
        else:
            b = "0.7-1.0"

        if b not in bins:
            bins[b] = {"preds": [], "actuals": []}
        bins[b]["preds"].append(p)
        bins[b]["actuals"].append(1.0 if r.actual_direction_up else 0.0)

    result = {}
    for b, data in sorted(bins.items()):
        n = len(data["preds"])
        mean_pred = np.mean(data["preds"])
        obs_rate = np.mean(data["actuals"])
        gap = abs(obs_rate - mean_pred)
        result[b] = {
            "n": n,
            "mean_predicted": round(float(mean_pred), 4),
            "observed_rate": round(float(obs_rate), 4),
            "gap": round(float(gap), 4),
            "quality": "GOOD" if gap < 0.1 else ("FAIR" if gap < 0.2 else "POOR"),
        }
    return result


def spearman_corr(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation."""
    if len(x) < 5:
        return 0.0
    from scipy.stats import spearmanr
    try:
        corr, _ = spearmanr(x, y)
        return float(corr) if not math.isnan(corr) else 0.0
    except Exception:
        # Fallback: manual rank correlation
        n = len(x)
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        d = rx - ry
        return float(1 - 6 * np.sum(d**2) / (n * (n**2 - 1)))


# ─── Main diagnostic runner ──────────────────────────────────

def run_diagnosis(n_bars: int = 2000,
                  max_forecasts: int = 75,
                  verbose: bool = True) -> DiagnosticReport:
    """Run full diagnostic suite."""
    report = DiagnosticReport()
    t0 = time.time()

    # ── Fetch data ────────────────────────────────────────
    if verbose:
        print(f"\n{'='*65}")
        print(f"  FORECASTER DIAGNOSTIC — {n_bars} bars, up to {max_forecasts} forecasts")
        print(f"{'='*65}")
        print("\nFetching data...")

    bars, source = fetch_ohlcv_bars("BTCUSDT", limit=n_bars)
    report.n_bars = len(bars)
    if verbose:
        print(f"  Got {len(bars)} bars from {source}")
    if len(bars) < 200:
        report.verdict = "INSUFFICIENT DATA"
        return report

    report.date_range = f"{bars[0].timestamp} → {bars[-1].timestamp}"

    # ── Run A: BASELINE (no hybrid ML, no regime gate) ────
    if verbose:
        print(f"\n--- TEST A: Baseline (weighted average, no hybrid ML) ---")

    fc_base = Forecaster(
        enable_mc=True, enable_hybrid_ml=False,
        enable_regime_gate=False, enable_auto_fix=False,
    )
    # ── Run A: BASELINE (no GBM — fresh forecaster) ───────
    if verbose:
        print(f"\n--- TEST A: Baseline (weighted average, no GBM) ---")

    fc_base = Forecaster(enable_mc=True)
    ev_base = WalkForwardEvaluator(
        forecaster=fc_base, bars=bars,
        horizon_hours=24, min_history=168, step_size=24,
    )
    metrics_base = ev_base.run(max_forecasts=max_forecasts, verbose=verbose)
    records_base = ev_base.records

    report.baseline_hr = metrics_base.hit_rate
    report.baseline_mcc = metrics_base.mcc

    if verbose:
        print(f"  Baseline: HR={metrics_base.hit_rate:.1%}, "
              f"MCC={metrics_base.mcc:.4f}, "
              f"n={len(records_base)}")

    # ── Run B: WITH Hybrid ML + Regime Gate + Auto-Fix ────
    if verbose:
        print(f"\n--- TEST B: Hybrid ML + Regime Gate + Auto-Calibration ---")

    fc_gbm = Forecaster(
        enable_mc=True, enable_hybrid_ml=True,
        enable_regime_gate=True, enable_auto_fix=True,
    )
    # ── Run B: WITH GBM ───────────────────────────────────
    # The GBM trains walk-forward during the evaluation, so
    # early forecasts use baseline, later ones use GBM.
    # This IS the real comparison — same data, same order.
    if verbose:
        print(f"\n--- TEST B: With GBM meta-learner ---")

    fc_gbm = Forecaster(enable_mc=True)
    ev_gbm = WalkForwardEvaluator(
        forecaster=fc_gbm, bars=bars,
        horizon_hours=24, min_history=168, step_size=24,
    )
    metrics_gbm = ev_gbm.run(max_forecasts=max_forecasts, verbose=verbose)
    records_gbm = ev_gbm.records

    report.n_forecasts = len(records_gbm)
    report.gbm_hr = metrics_gbm.hit_rate
    report.gbm_mcc = metrics_gbm.mcc
    report.gbm_helped = (metrics_gbm.mcc > metrics_base.mcc)

    # Track hybrid ML stats
    report.hybrid_ml_hr = metrics_gbm.hit_rate
    report.hybrid_ml_mcc = metrics_gbm.mcc
    report.hybrid_ml_helped = report.gbm_helped

    gbm_trained = fc_gbm.meta_learner._dir_model is not None
    gbm_samples = len(fc_gbm.meta_learner._history)

    if verbose:
        print(f"  Hybrid ML: HR={metrics_gbm.hit_rate:.1%}, "
        print(f"  GBM: HR={metrics_gbm.hit_rate:.1%}, "
              f"MCC={metrics_gbm.mcc:.4f}, "
              f"trained={gbm_trained}, samples={gbm_samples}")
        delta = metrics_gbm.hit_rate - metrics_base.hit_rate
        print(f"  Delta: HR {delta:+.1%}, "
              f"Hybrid ML {'HELPED' if report.gbm_helped else 'HURT'}")

    # ── Regime gate stats ─────────────────────────────────
    gate_info = fc_gbm.regime_gate
    for regime_name, hits in gate_info._live_hits.items():
        if hits:
            report.regime_gate_stats[regime_name] = {
                "n": len(hits),
                "hit_rate": round(sum(hits) / len(hits), 4),
            }
    if verbose and report.regime_gate_stats:
        print(f"\n  Regime Gate Live Stats:")
        for rname, rstat in report.regime_gate_stats.items():
            print(f"    {rname:20s}: HR={rstat['hit_rate']:.1%} (n={rstat['n']})")

    # ── Health monitor status ─────────────────────────────
    health = fc_gbm.auto_fix.monitor.check_health()
    report.health_status = {
        "is_healthy": health.is_healthy,
        "rolling_hr_10": round(health.rolling_hr_10, 4),
        "rolling_hr_20": round(health.rolling_hr_20, 4),
        "mcc_last_20": round(health.mcc_last_20, 4),
        "coverage": round(health.coverage_p5_p95, 4),
        "drift_detected": health.concept_drift_detected,
        "needs_retrain": health.needs_retrain,
        "alerts": health.alerts,
    }
    if verbose:
        print(f"\n  Health Monitor:")
        print(f"    Healthy:  {health.is_healthy}")
        print(f"    HR(10):   {health.rolling_hr_10:.1%}")
        print(f"    HR(20):   {health.rolling_hr_20:.1%}")
        print(f"    MCC(20):  {health.mcc_last_20:.4f}")
        print(f"    Drift:    {health.concept_drift_detected}")
        for alert in health.alerts:
            print(f"    ALERT:    {alert}")

    # Use enhanced records for remaining diagnostics
    # Use GBM records for remaining diagnostics
    records = records_gbm

    # ── Statistical test: is HR different from coin flip? ──
    if verbose:
        print(f"\n--- STATISTICAL SIGNIFICANCE ---")

    hits = sum(1 for r in records
               if (r.predicted_direction_prob > 0.5) == r.actual_direction_up)
    n = len(records)
    report.hr_vs_coinflip_pvalue = binomial_test_pvalue(hits, n, 0.5)
    report.hr_significant = report.hr_vs_coinflip_pvalue < 0.05
    report.direction_is_random = not report.hr_significant

    if verbose:
        print(f"  Hits: {hits}/{n} = {hits/n:.1%}")
        print(f"  Binomial test p-value: {report.hr_vs_coinflip_pvalue:.4f}")
        print(f"  Significant (p<0.05): {'YES' if report.hr_significant else 'NO'}")
        print(f"  Direction is random:   {'YES ⚠️' if report.direction_is_random else 'NO ✓'}")

    # ── Rolling HR ────────────────────────────────────────
    if verbose:
        print(f"\n--- ROLLING DIRECTION HR (window=10) ---")

    report.rolling_hrs = rolling_hit_rate(records, window=10)
    report.rolling_trend = classify_trend(report.rolling_hrs)

    if verbose:
        for i, hr in enumerate(report.rolling_hrs):
            bar = "█" * int(hr * 40)
            marker = " ←coin" if abs(hr - 0.5) < 0.05 else ""
            print(f"  [{i*1+1:2d}-{i*1+10:2d}] {hr:.0%} {bar}{marker}")
        print(f"  Trend: {report.rolling_trend}")

    # ── Per-regime breakdown ──────────────────────────────
    if verbose:
        print(f"\n--- PER-REGIME BREAKDOWN ---")

    regime_groups: dict[str, list[EvalRecord]] = {}
    for r in records:
        rg = r.regime or "unknown"
        regime_groups.setdefault(rg, []).append(r)

    for regime, recs in sorted(regime_groups.items(),
                                key=lambda x: -len(x[1])):
        n_r = len(recs)
        hits_r = sum(1 for r in recs
                     if (r.predicted_direction_prob > 0.5) == r.actual_direction_up)
        hr_r = hits_r / n_r if n_r > 0 else 0
        avg_conf = np.mean([abs(r.predicted_direction_prob - 0.5) for r in recs])
        report.regime_stats[regime] = {
            "n": n_r,
            "hit_rate": round(hr_r, 4),
            "avg_confidence": round(float(avg_conf), 4),
            "pct_of_total": round(n_r / len(records), 4),
        }
        if verbose:
            verdict = "✓" if hr_r > 0.5 else ("~" if hr_r > 0.45 else "✗")
            print(f"  {regime:20s}: HR={hr_r:.1%} ({hits_r}/{n_r})  "
                  f"conf={avg_conf:.3f}  {verdict}")

    # ── Calibration ───────────────────────────────────────
    if verbose:
        print(f"\n--- CALIBRATION RELIABILITY ---")

    report.calibration_bins = compute_calibration(records)
    n_good = sum(1 for b in report.calibration_bins.values()
                 if b["quality"] == "GOOD")
    n_bins = len(report.calibration_bins)
    report.calibration_quality = (
        "GOOD" if n_good >= n_bins * 0.6
        else ("FAIR" if n_good >= n_bins * 0.3 else "POOR")
    )

    if verbose:
        for bin_name, b in report.calibration_bins.items():
            print(f"  {bin_name:>10s}: n={b['n']:3d}  "
                  f"pred={b['mean_predicted']:.3f}  "
                  f"obs={b['observed_rate']:.3f}  "
                  f"gap={b['gap']:.3f} [{b['quality']}]")
        print(f"  Overall calibration: {report.calibration_quality}")

    # ── Predicted vs actual correlation ───────────────────
    if verbose:
        print(f"\n--- PREDICTED vs ACTUAL RETURNS ---")

    pred_rets = [r.predicted_return for r in records]
    actual_rets = [r.actual_return for r in records]

    if len(pred_rets) >= 5:
        report.pred_actual_corr = float(np.corrcoef(pred_rets, actual_rets)[0, 1])
        if math.isnan(report.pred_actual_corr):
            report.pred_actual_corr = 0.0
        report.pred_actual_rank_corr = spearman_corr(pred_rets, actual_rets)

    if verbose:
        print(f"  Pearson r:  {report.pred_actual_corr:.4f}")
        print(f"  Spearman ρ: {report.pred_actual_rank_corr:.4f}")
        if report.pred_actual_corr < 0:
            print(f"  ⚠️  Negative correlation = predictions are anti-signal")
        elif report.pred_actual_corr < 0.05:
            print(f"  ⚠️  Near-zero correlation = no predictive power")
        else:
            print(f"  ✓  Positive correlation = some signal")

    # ── MC quality ────────────────────────────────────────
    mc = metrics_gbm.mc_metrics or {}
    report.p5_p95_coverage = mc.get("p5_p95_coverage", 0)
    report.mc_price_mae_pct = mc.get("mc_price_mae_pct", 0)

    # ── Feature importance (hybrid ML or GBM) ────────────
    if verbose:
        print(f"\n--- FEATURE IMPORTANCE ---")

    # Try hybrid predictor first, then meta-learner
    hybrid = fc_gbm.hybrid_predictor
    if hybrid.is_trained:
        report.feature_importances = hybrid.get_feature_importances()[:15]
        if verbose:
            print(f"  Source: HybridPredictor (LightGBM + temporal)")
            for name, score in report.feature_importances[:10]:
                bar = "█" * int(min(score, 50))
                print(f"  {name:45s}  {score:6.1f} {bar}")
    elif gbm_trained and fc_gbm.meta_learner._feature_names:
        imp = fc_gbm.meta_learner._dir_model.feature_importance(
            importance_type="gain")
        pairs = sorted(zip(fc_gbm.meta_learner._feature_names, imp),
                       key=lambda x: -x[1])
        report.feature_importances = [(n, float(s)) for n, s in pairs[:15]]
        if verbose:
            print(f"  Source: MetaLearner GBM")
            for name, score in report.feature_importances[:10]:
                bar = "█" * int(min(score, 50))
                print(f"  {name:45s}  {score:6.1f} {bar}")
            zero_imp = sum(1 for _, s in pairs if s == 0)
            print(f"  Zero-importance features: {zero_imp}/{len(pairs)}")
    else:
        if verbose:
            print(f"  No ML model trained (insufficient samples)")

    # ── AUTO-FIX (when enabled) ───────────────────────────
    auto_fix_report = fc_gbm.auto_fix.check_and_fix()
    report.auto_fix_applied = auto_fix_report.retrain_triggered or auto_fix_report.calibration_applied
    report.auto_fix_report = {
        "calibration_applied": auto_fix_report.calibration_applied,
        "calibration_method": auto_fix_report.calibration_method,
        "retrain_triggered": auto_fix_report.retrain_triggered,
        "confidence_scalar_adjusted": auto_fix_report.confidence_scalar_adjusted,
        "new_confidence_scalar": auto_fix_report.new_confidence_scalar,
        "actions_taken": auto_fix_report.actions_taken,
        "verdict": auto_fix_report.verdict,
    }
    if verbose and auto_fix_report.actions_taken:
        print(f"\n--- AUTO-FIX PIPELINE ---")
        print(f"  Status: {auto_fix_report.verdict}")
        for action in auto_fix_report.actions_taken:
            print(f"  ACTION: {action}")
            print(f"  GBM not trained (insufficient samples)")

    # ── VERDICT ───────────────────────────────────────────
    report.elapsed_s = time.time() - t0
    report.verdict, report.recommendations = _render_verdict(report)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  VERDICT: {report.verdict}")
        print(f"{'='*65}")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        print(f"\n  Elapsed: {report.elapsed_s:.1f}s")

    return report


def _render_verdict(r: DiagnosticReport) -> tuple[str, list[str]]:
    """Generate honest verdict and recommendations."""
    recs = []
    problems = 0

    # Direction
    if r.direction_is_random:
        problems += 2
        recs.append(
            f"Direction HR ({r.gbm_hr:.1%}) is NOT statistically different "
            f"from coin flip (p={r.hr_vs_coinflip_pvalue:.3f}). "
            f"The model has no directional edge on this sample."
        )
    elif r.gbm_hr < 0.50:
        problems += 1
        recs.append(
            f"Direction HR ({r.gbm_hr:.1%}) is below 50% but may be "
            f"significant. Check if inverting predictions helps."
        )

    # GBM
    if not r.gbm_helped:
        problems += 1
        recs.append(
            f"GBM hurt performance (baseline MCC={r.baseline_mcc:.3f} "
            f"vs GBM MCC={r.gbm_mcc:.3f}). The GBM may be overfitting "
            f"on {r.n_forecasts} samples — consider disabling until "
            f"more walk-forward data accumulates."
        )
    else:
        recs.append(
            f"GBM improved MCC from {r.baseline_mcc:.3f} to "
            f"{r.gbm_mcc:.3f}. Keep it, but monitor on new data."
        )

    # Correlation
    if r.pred_actual_corr < 0:
        problems += 1
        recs.append(
            f"Predicted returns are ANTI-correlated with actual returns "
            f"(r={r.pred_actual_corr:.3f}). The return predictor is "
            f"systematically wrong on magnitude/direction. Consider "
            f"flattening return predictions toward zero."
        )
    elif r.pred_actual_corr < 0.05:
        problems += 1
        recs.append(
            f"Predicted returns have near-zero correlation with actuals "
            f"(r={r.pred_actual_corr:.3f}). Return predictions are noise."
        )

    # Calibration
    if r.calibration_quality == "POOR":
        problems += 1
        recs.append(
            "Calibration is POOR — predicted probabilities don't match "
            "observed frequencies. Need more training data for isotonic "
            "regression or stronger static calibration."
        )

    # Regime
    best_regime = max(r.regime_stats.items(),
                      key=lambda x: x[1]["hit_rate"],
                      default=("none", {"hit_rate": 0, "n": 0}))
    worst_regime = min(r.regime_stats.items(),
                       key=lambda x: x[1]["hit_rate"],
                       default=("none", {"hit_rate": 0, "n": 0}))
    if best_regime[1]["hit_rate"] > 0.55 and best_regime[1]["n"] >= 5:
        recs.append(
            f"Best regime: {best_regime[0]} "
            f"(HR={best_regime[1]['hit_rate']:.1%}, "
            f"n={best_regime[1]['n']}). Consider only trading in "
            f"this regime."
        )
    if worst_regime[1]["hit_rate"] < 0.35 and worst_regime[1]["n"] >= 5:
        recs.append(
            f"Worst regime: {worst_regime[0]} "
            f"(HR={worst_regime[1]['hit_rate']:.1%}, "
            f"n={worst_regime[1]['n']}). Model is anti-predictive here."
        )

    # Rolling trend
    if r.rolling_trend == "degrading":
        problems += 1
        recs.append(
            "Rolling HR is DEGRADING over time — model may be "
            "concept-drifting or overfitting to early patterns."
        )
    elif r.rolling_trend == "improving":
        recs.append(
            "Rolling HR is improving — the hybrid ML may be learning. "
            "More data could help."
        )

    # Hybrid ML specific
    if r.hybrid_ml_helped:
        recs.append(
            f"Hybrid ML (LightGBM+GRU) improved performance: "
            f"HR={r.hybrid_ml_hr:.1%}, MCC={r.hybrid_ml_mcc:.4f}. "
            f"Keep enabled."
        )
    elif r.hybrid_ml_hr > 0:
        recs.append(
            f"Hybrid ML did not help (HR={r.hybrid_ml_hr:.1%}). "
            f"Consider disabling until more data."
        )

    # Health monitoring
    if r.health_status.get("drift_detected"):
        problems += 1
        recs.append(
            "Concept drift DETECTED by health monitor. "
            "Auto-retrain should be triggered."
        )
    if r.health_status.get("needs_retrain"):
        recs.append(
            "Health monitor recommends RETRAIN due to degraded performance."
        )

    # Auto-fix
    if r.auto_fix_applied:
        recs.append(
            f"Auto-fix pipeline applied: {r.auto_fix_report.get('verdict', 'unknown')}. "
            f"Actions: {', '.join(r.auto_fix_report.get('actions_taken', []))}"
        )

    # MC
    if r.p5_p95_coverage > 0:
        if abs(r.p5_p95_coverage - 0.90) < 0.05:
            recs.append(
                f"MC P5-P95 coverage ({r.p5_p95_coverage:.1%}) is well-"
                f"calibrated around the 90% target."
            )
        elif r.p5_p95_coverage > 0.97:
            recs.append(
                f"MC intervals are too wide (coverage={r.p5_p95_coverage:.1%}). "
                f"The model is hiding behind fat tails."
            )

    # Overall
    if problems >= 3:
        verdict = "POOR — No reliable signal detected"
    elif problems >= 2:
        verdict = "WEAK — Some issues, needs investigation"
    elif problems >= 1:
        verdict = "MIXED — Partial signal, room for improvement"
    else:
        verdict = "PROMISING — Signal detected, continue development"

    return verdict, recs


# ─── CLI ──────────────────────────────────────────────────────

def full_diagnostics_and_fix(
    n_bars: int = 2000,
    max_forecasts: int = 75,
    auto_fix: bool = True,
    verbose: bool = True,
) -> DiagnosticReport:
    """
    Run full diagnostics with auto-fix.
    Called by clawdbot.service ExecStartPre for health check on boot.

    Returns DiagnosticReport with verdict and any actions taken.
    """
    report = run_diagnosis(n_bars=n_bars, max_forecasts=max_forecasts, verbose=verbose)

    if auto_fix and report.verdict.startswith(("POOR", "WEAK")):
        if verbose:
            print(f"\n  AUTO-FIX: Verdict is {report.verdict}, running fixes...")
        # The auto-fix is already integrated into the Forecaster,
        # so the report already contains any auto-fix actions.
        # Additional fix: write a recovery marker file
        try:
            import os
            marker_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "diagnostics_report.json",
            )
            os.makedirs(os.path.dirname(marker_path), exist_ok=True)
            with open(marker_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            if verbose:
                print(f"  Diagnostic report saved to {marker_path}")
        except Exception as e:
            if verbose:
                print(f"  Could not save report: {e}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Forecaster Diagnostic Suite (Ultimate Enhanced)")
        description="Forecaster Diagnostic Suite (14-Paradigm)")
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--forecasts", type=int, default=75)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--auto-fix", action="store_true",
                        help="Run full diagnostics with auto-fix pipeline")
    args = parser.parse_args()

    if args.auto_fix:
        report = full_diagnostics_and_fix(
            n_bars=args.bars,
            max_forecasts=args.forecasts,
            verbose=not args.json,
        )
    else:
        report = run_diagnosis(
            n_bars=args.bars,
            max_forecasts=args.forecasts,
            verbose=not args.json,
        )

    if args.json or args.output:
        data = report.to_dict()
        json_str = json.dumps(data, indent=2, default=str)
        if args.output:
            with open(args.output, "w") as f:
                f.write(json_str)
            print(f"Saved to {args.output}")
        else:
            print(json_str)


if __name__ == "__main__":
    main()
