"""
ClawdBot ↔ Yoshi Bridge — Unified Integration Layer
=====================================================
Connects the ClawdBot-V1 14-paradigm ensemble forecaster with Yoshi's
walk-forward harness, KPCOFGS regime classifier, backtest engine,
and Kalshi execution layer.

Architecture:
  ClawdBot engine.py (fast forecast, 500ms)
  → Bridge (this file)
    → Yoshi KPCOFGS regime enrichment
    → Yoshi walk-forward validation with purge/embargo
    → Yoshi scoring (pinball, coverage, sharpness, CRPS)
    → Yoshi backtest (PnL, Sharpe, drawdown)
    → Yoshi Kalshi scanner + execution
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# CLAWDBOT IMPORTS (lazy to avoid circular)
# ═══════════════════════════════════════════════════════════════
def _get_forecaster():
    from scripts.forecaster.engine import Forecaster, ForecastResult
    return Forecaster, ForecastResult


def _get_schemas():
    from scripts.forecaster.schemas import (
        Bar, MarketSnapshot, PredictionTargets, Regime
    )
    return Bar, MarketSnapshot, PredictionTargets, Regime


# ═══════════════════════════════════════════════════════════════
# YOSHI IMPORTS (safe — all pure-python, no heavy deps)
# ═══════════════════════════════════════════════════════════════
from gnosis.regimes.kpcofgs import KPCOFGSClassifier


# ═══════════════════════════════════════════════════════════════
# KPCOFGS ADAPTER: enrich ClawdBot snapshot with 7-level regimes
# ═══════════════════════════════════════════════════════════════

DEFAULT_REGIMES_CONFIG = {
    "K": {"labels": ["K_TRENDING", "K_MEAN_REVERTING", "K_BALANCED"]},
    "P": {"labels": ["P_VOL_EXPANDING", "P_VOL_CONTRACTING", "P_VOL_STABLE"]},
    "C": {"labels": ["C_BUY_FLOW_DOMINANT", "C_SELL_FLOW_DOMINANT", "C_FLOW_NEUTRAL"]},
    "O": {"labels": ["O_BREAKOUT", "O_BREAKDOWN", "O_RANGE", "O_SWEEP_REVERT"]},
    "F": {"labels": ["F_ACCEL", "F_DECEL", "F_STALL", "F_REVERSAL"]},
    "G": {"labels": ["G_TREND_CONT", "G_TREND_EXH", "G_MR_BOUNCE",
                      "G_MR_FADE", "G_BO_HOLD", "G_BO_FAIL"]},
    "S": {"labels": [
        "S_TC_PULLBACK_RESUME", "S_TC_ACCEL_BREAK", "S_TX_TOPPING_ROLL",
        "S_MR_OVERSHOOT_SNAPBACK", "S_MR_GRIND_BACK",
        "S_BO_LEVEL_BREAK_HOLD", "S_BO_LEVEL_BREAK_FAIL",
        "S_RANGE_EDGE_FADE", "S_RANGE_MID_MEANREV",
        "S_SWEEP_UP_REVERT", "S_SWEEP_DOWN_REVERT", "S_UNCERTAIN",
    ]},
}


def snapshot_to_dataframe(bars: list, recent_trades: list = None) -> pd.DataFrame:
    """Convert ClawdBot Bar list to pandas DataFrame for KPCOFGS.

    Computes the minimum feature set KPCOFGS needs:
      returns, realized_vol, ofi, range_pct
    """
    if not bars:
        return pd.DataFrame()

    rows = []
    for b in bars:
        rows.append({
            "timestamp": getattr(b, "timestamp", 0),
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Returns
    df["returns"] = df["close"].pct_change().fillna(0)

    # Realized vol (20-bar rolling)
    df["realized_vol"] = df["returns"].rolling(20, min_periods=5).std().fillna(
        df["returns"].std() if len(df) > 1 else 0.01
    )

    # Range percent
    df["range_pct"] = ((df["high"] - df["low"]) / df["close"]).fillna(0)

    # OFI (proxy: close position in range * volume direction)
    range_size = df["high"] - df["low"]
    mid_range = (df["high"] + df["low"]) / 2
    df["ofi"] = np.where(
        range_size > 0,
        (df["close"] - mid_range) / (range_size + 1e-10) * df["volume"],
        0,
    )

    # Symbol (required by some downstream code)
    df["symbol"] = "BTCUSDT"

    return df


def classify_kpcofgs(df: pd.DataFrame,
                     config: dict = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run KPCOFGS classification on a bars DataFrame.

    Returns:
        (enriched_df, summary_dict)
        enriched_df has columns: K_label, P_label, ..., S_label, regime_entropy
        summary_dict has latest labels and probabilities.
    """
    config = config or DEFAULT_REGIMES_CONFIG
    classifier = KPCOFGSClassifier(config)

    classified = classifier.classify(df)

    # Extract latest row summary
    if classified.empty:
        return classified, {}

    latest = classified.iloc[-1]
    summary = {}
    for level in ["K", "P", "C", "O", "F", "G", "S"]:
        label_col = f"{level}_label"
        pmax_col = f"{level}_pmax"
        entropy_col = f"{level}_entropy"
        if label_col in classified.columns:
            summary[f"{level}_label"] = str(latest.get(label_col, "?"))
        if pmax_col in classified.columns:
            summary[f"{level}_pmax"] = round(float(latest.get(pmax_col, 0)), 4)
        if entropy_col in classified.columns:
            summary[f"{level}_entropy"] = round(float(latest.get(entropy_col, 0)), 4)

    if "regime_entropy" in classified.columns:
        summary["regime_entropy"] = round(float(latest.get("regime_entropy", 0)), 4)

    return classified, summary


# ═══════════════════════════════════════════════════════════════
# KPCOFGS → CLAWDBOT REGIME MAPPING
# ═══════════════════════════════════════════════════════════════
# Map KPCOFGS scenario-level labels to ClawdBot's 6-state Regime enum.

_KPCOFGS_TO_CLAWDBOT = {
    "S_TC_PULLBACK_RESUME": "trend_up",
    "S_TC_ACCEL_BREAK": "trend_up",
    "S_TX_TOPPING_ROLL": "trend_down",
    "S_MR_OVERSHOOT_SNAPBACK": "post_jump",
    "S_MR_GRIND_BACK": "range",
    "S_BO_LEVEL_BREAK_HOLD": "vol_expansion",
    "S_BO_LEVEL_BREAK_FAIL": "range",
    "S_RANGE_EDGE_FADE": "range",
    "S_RANGE_MID_MEANREV": "range",
    "S_SWEEP_UP_REVERT": "post_jump",
    "S_SWEEP_DOWN_REVERT": "post_jump",
    "S_UNCERTAIN": "range",
}

_KPCOFGS_G_TO_CLAWDBOT = {
    "G_TREND_CONT": "trend_up",     # direction depends on K, but "trending"
    "G_TREND_EXH": "trend_down",    # exhaustion → reversal
    "G_MR_BOUNCE": "range",
    "G_MR_FADE": "range",
    "G_BO_HOLD": "vol_expansion",
    "G_BO_FAIL": "range",
}


def kpcofgs_to_regime(summary: Dict[str, Any]) -> str:
    """Map KPCOFGS summary to ClawdBot regime string."""
    s_label = summary.get("S_label", "S_UNCERTAIN")
    if s_label in _KPCOFGS_TO_CLAWDBOT:
        return _KPCOFGS_TO_CLAWDBOT[s_label]

    # Fall back to G-level
    g_label = summary.get("G_label", "")
    if g_label in _KPCOFGS_G_TO_CLAWDBOT:
        regime = _KPCOFGS_G_TO_CLAWDBOT[g_label]
        # Refine trend direction using K-level
        k_label = summary.get("K_label", "K_BALANCED")
        if regime == "trend_up" and k_label == "K_MEAN_REVERTING":
            return "range"
        return regime

    return "range"


# ═══════════════════════════════════════════════════════════════
# YOSHI SCORING ADAPTER
# ═══════════════════════════════════════════════════════════════

def score_forecast_series(
    forecasts: List[Dict[str, Any]],
    actuals: List[float],
) -> Dict[str, float]:
    """Score a series of ClawdBot forecasts using Yoshi metrics.

    Args:
        forecasts: list of dicts with keys:
            expected_return, quantile_10, quantile_50, quantile_90,
            direction_prob, volatility_forecast
        actuals: list of actual log-returns (same length)

    Returns:
        Dict with pinball_05, pinball_50, pinball_95, coverage_90,
        sharpness, crps, mae, hit_rate, n_samples
    """
    n = min(len(forecasts), len(actuals))
    if n == 0:
        return {"n_samples": 0}

    y_true = np.array(actuals[:n], dtype=float)
    q05 = np.array([f.get("quantile_05", f.get("quantile_10", 0)) for f in forecasts[:n]])
    q50 = np.array([f.get("quantile_50", 0) for f in forecasts[:n]])
    q95 = np.array([f.get("quantile_95", f.get("quantile_90", 0)) for f in forecasts[:n]])
    dir_probs = np.array([f.get("direction_prob", 0.5) for f in forecasts[:n]])

    # Pinball losses
    def _pinball(y, yhat, q):
        err = y - yhat
        return float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))

    # Coverage
    in_interval = (y_true >= q05) & (y_true <= q95)
    cov = float(np.mean(in_interval))

    # Sharpness
    sharp = float(np.mean(q95 - q05))

    # CRPS approximation
    crps = (_pinball(y_true, q05, 0.05) +
            _pinball(y_true, q50, 0.50) +
            _pinball(y_true, q95, 0.95)) / 3.0

    # Hit rate
    actual_up = y_true > 0
    predicted_up = dir_probs > 0.5
    hits = int(np.sum(actual_up == predicted_up))
    hr = hits / n if n > 0 else 0

    # Binomial p-value (scipy >= 1.7 uses binomtest)
    try:
        from scipy.stats import binomtest
        p_val = float(binomtest(hits, n, 0.5).pvalue)
    except ImportError:
        try:
            from scipy.stats import binom_test
            p_val = float(binom_test(hits, n, 0.5))
        except Exception:
            p_val = 1.0
    except Exception:
        p_val = 1.0

    return {
        "pinball_05": round(_pinball(y_true, q05, 0.05), 6),
        "pinball_50": round(_pinball(y_true, q50, 0.50), 6),
        "pinball_95": round(_pinball(y_true, q95, 0.95), 6),
        "coverage_90": round(cov, 4),
        "sharpness": round(sharp, 6),
        "crps": round(crps, 6),
        "mae": round(float(np.mean(np.abs(y_true - q50))), 6),
        "hit_rate": round(hr, 4),
        "hit_rate_p_value": round(p_val, 4),
        "n_samples": n,
    }


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION WITH PURGE/EMBARGO
# ═══════════════════════════════════════════════════════════════

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    n_outer_folds: int = 5
    train_bars: int = 500
    val_bars: int = 100
    test_bars: int = 100
    purge_bars: int = 24    # 24h for 1h bars
    embargo_bars: int = 12  # 12h buffer
    horizon_bars: int = 24  # forecast horizon in bars


@dataclass
class FoldResult:
    fold_idx: int
    train_size: int
    test_size: int
    hit_rate: float
    pinball_50: float
    coverage_90: float
    sharpness: float
    crps: float
    regime_breakdown: Dict[str, Dict] = field(default_factory=dict)


def run_walk_forward(
    bars: list,
    forecaster_fn,
    config: WalkForwardConfig = None,
    verbose: bool = False,
) -> Tuple[List[FoldResult], Dict[str, float]]:
    """Run walk-forward validation on ClawdBot forecaster with purge/embargo.

    Args:
        bars: list of Bar objects (ClawdBot schema)
        forecaster_fn: callable(bars_window) -> dict with forecast fields
        config: WalkForwardConfig
        verbose: print progress

    Returns:
        (fold_results, aggregate_metrics)
    """
    config = config or WalkForwardConfig()
    n_bars = len(bars)
    fold_size = config.train_bars + config.purge_bars + config.val_bars
    min_bars = fold_size + config.test_bars + config.embargo_bars

    if n_bars < min_bars:
        if verbose:
            print(f"  Not enough bars for walk-forward: {n_bars} < {min_bars}")
        return [], {"error": "insufficient_data", "n_bars": n_bars, "min_bars": min_bars}

    # Generate folds
    folds = []
    step = max(1, (n_bars - min_bars) // max(1, config.n_outer_folds - 1))

    for fold_idx in range(config.n_outer_folds):
        train_start = fold_idx * step
        train_end = train_start + config.train_bars
        # Purge gap
        test_start = train_end + config.purge_bars + config.embargo_bars
        test_end = min(test_start + config.test_bars, n_bars - config.horizon_bars)

        if test_end <= test_start or test_start >= n_bars:
            break

        folds.append((fold_idx, train_start, train_end, test_start, test_end))

    results = []
    all_forecasts = []
    all_actuals = []

    for fold_idx, train_start, train_end, test_start, test_end in folds:
        fold_forecasts = []
        fold_actuals = []

        for t in range(test_start, test_end):
            # Use all bars up to time t (expanding window within fold)
            window_bars = bars[train_start:t]
            if len(window_bars) < 50:
                continue

            try:
                fc = forecaster_fn(window_bars)
            except Exception as e:
                if verbose:
                    print(f"  Fold {fold_idx} bar {t}: forecast error: {e}")
                continue

            # Actual return (horizon bars ahead)
            horizon_end = min(t + config.horizon_bars, n_bars - 1)
            actual_return = math.log(bars[horizon_end].close / bars[t].close)

            fold_forecasts.append(fc)
            fold_actuals.append(actual_return)

        if not fold_forecasts:
            continue

        # Score this fold
        scores = score_forecast_series(fold_forecasts, fold_actuals)

        results.append(FoldResult(
            fold_idx=fold_idx,
            train_size=train_end - train_start,
            test_size=len(fold_forecasts),
            hit_rate=scores.get("hit_rate", 0),
            pinball_50=scores.get("pinball_50", 0),
            coverage_90=scores.get("coverage_90", 0),
            sharpness=scores.get("sharpness", 0),
            crps=scores.get("crps", 0),
        ))

        all_forecasts.extend(fold_forecasts)
        all_actuals.extend(fold_actuals)

        if verbose:
            print(f"  Fold {fold_idx}: n={len(fold_forecasts)}, "
                  f"HR={scores.get('hit_rate', 0):.1%}, "
                  f"cov={scores.get('coverage_90', 0):.1%}, "
                  f"CRPS={scores.get('crps', 0):.6f}")

    # Aggregate
    if all_forecasts:
        agg = score_forecast_series(all_forecasts, all_actuals)
    else:
        agg = {"n_samples": 0, "error": "no_forecasts"}

    agg["n_folds"] = len(results)
    agg["fold_hit_rates"] = [r.hit_rate for r in results]

    return results, agg


# ═══════════════════════════════════════════════════════════════
# BACKTEST ADAPTER
# ═══════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    """Simplified backtest configuration."""
    initial_capital: float = 10_000.0
    position_size_pct: float = 0.02   # 2% of capital per trade
    stop_loss_pct: float = 0.03       # 3% stop loss
    take_profit_pct: float = 0.06     # 6% take profit (2:1 R:R)
    fee_pct: float = 0.001            # 0.1% per trade
    min_ev_edge: float = 0.04         # minimum |dir_prob - 0.50|
    max_open_positions: int = 1


@dataclass
class Trade:
    """Single trade record."""
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    side: str              # "long" or "short"
    pnl: float
    pnl_pct: float
    regime: str
    direction_prob: float
    exit_reason: str       # "take_profit", "stop_loss", "horizon", "signal_flip"


@dataclass
class BacktestResult:
    """Backtest output."""
    trades: List[Trade]
    equity_curve: List[float]
    stats: Dict[str, float]


def run_backtest(
    bars: list,
    forecaster_fn,
    config: BacktestConfig = None,
    horizon_bars: int = 24,
    verbose: bool = False,
) -> BacktestResult:
    """Run a simple backtest using ClawdBot forecasts.

    Uses decision-at-t, fill-at-t+1 model (no lookahead).

    Args:
        bars: list of Bar objects
        forecaster_fn: callable(bars_up_to_t) -> dict with direction_prob, regime, etc.
        config: BacktestConfig
        horizon_bars: forecast horizon in bars
        verbose: print trades

    Returns:
        BacktestResult with trades, equity curve, and stats
    """
    config = config or BacktestConfig()
    n = len(bars)

    capital = config.initial_capital
    equity = [capital]
    trades: List[Trade] = []
    position = None  # (entry_bar, entry_price, side, target, stop)

    # Start after warmup
    warmup = 200

    for t in range(warmup, n - 1):
        price_t = bars[t].close
        price_t1 = bars[t + 1].close  # fill at t+1

        # Check exit conditions for open position
        if position is not None:
            entry_bar, entry_price, side, tp_price, sl_price, regime, dprob = position
            bars_held = t - entry_bar

            # Check stop/take-profit on t+1 price
            exit_reason = None
            if side == "long":
                if price_t1 >= tp_price:
                    exit_reason = "take_profit"
                elif price_t1 <= sl_price:
                    exit_reason = "stop_loss"
                elif bars_held >= horizon_bars:
                    exit_reason = "horizon"
            else:  # short
                if price_t1 <= tp_price:
                    exit_reason = "take_profit"
                elif price_t1 >= sl_price:
                    exit_reason = "stop_loss"
                elif bars_held >= horizon_bars:
                    exit_reason = "horizon"

            if exit_reason:
                if side == "long":
                    pnl_pct = (price_t1 - entry_price) / entry_price - config.fee_pct
                else:
                    pnl_pct = (entry_price - price_t1) / entry_price - config.fee_pct
                pnl = capital * config.position_size_pct * pnl_pct
                capital += pnl

                trades.append(Trade(
                    entry_bar=entry_bar, exit_bar=t + 1,
                    entry_price=entry_price, exit_price=price_t1,
                    side=side, pnl=pnl, pnl_pct=pnl_pct,
                    regime=regime, direction_prob=dprob,
                    exit_reason=exit_reason,
                ))
                position = None

                if verbose:
                    print(f"  EXIT {exit_reason}: bar {t+1} @ ${price_t1:.2f}, "
                          f"PnL {pnl_pct:+.2%} (${pnl:+.2f})")

        # Generate forecast if no position
        if position is None:
            window_bars = bars[max(0, t - 500):t + 1]
            if len(window_bars) < 50:
                equity.append(capital)
                continue

            try:
                fc = forecaster_fn(window_bars)
            except Exception:
                equity.append(capital)
                continue

            dir_prob = fc.get("direction_prob", 0.5)
            ev_edge = abs(dir_prob - 0.50)

            if ev_edge >= config.min_ev_edge:
                side = "long" if dir_prob > 0.5 else "short"
                entry_price = price_t1  # fill at t+1
                if side == "long":
                    tp = entry_price * (1 + config.take_profit_pct)
                    sl = entry_price * (1 - config.stop_loss_pct)
                else:
                    tp = entry_price * (1 - config.take_profit_pct)
                    sl = entry_price * (1 + config.stop_loss_pct)

                regime = fc.get("regime", "range")
                position = (t + 1, entry_price, side, tp, sl, regime, dir_prob)

                if verbose:
                    print(f"  ENTRY {side}: bar {t+1} @ ${entry_price:.2f}, "
                          f"dir={dir_prob:.3f}, regime={regime}")

        equity.append(capital)

    # Close any remaining position
    if position is not None:
        entry_bar, entry_price, side, _, _, regime, dprob = position
        exit_price = bars[-1].close
        if side == "long":
            pnl_pct = (exit_price - entry_price) / entry_price - config.fee_pct
        else:
            pnl_pct = (entry_price - exit_price) / entry_price - config.fee_pct
        pnl = capital * config.position_size_pct * pnl_pct
        capital += pnl
        trades.append(Trade(
            entry_bar=len(bars) - 1, exit_bar=len(bars) - 1,
            entry_price=entry_price, exit_price=exit_price,
            side=side, pnl=pnl, pnl_pct=pnl_pct,
            regime=regime, direction_prob=dprob,
            exit_reason="close",
        ))
        equity.append(capital)

    # Compute stats
    stats = _compute_backtest_stats(trades, equity, config.initial_capital)

    return BacktestResult(trades=trades, equity_curve=equity, stats=stats)


def _compute_backtest_stats(
    trades: List[Trade],
    equity: List[float],
    initial_capital: float,
) -> Dict[str, float]:
    """Compute performance statistics."""
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "n_trades": 0, "total_return_pct": 0, "sharpe": 0,
            "max_drawdown_pct": 0, "win_rate": 0,
        }

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    total_pnl = sum(t.pnl for t in trades)
    total_return = total_pnl / initial_capital

    # Sharpe (annualized, assuming hourly bars, ~8760 bars/year)
    returns = [t.pnl_pct for t in trades]
    avg_ret = np.mean(returns) if returns else 0
    std_ret = np.std(returns) if len(returns) > 1 else 1
    trades_per_year = 8760 / max(1, len(equity) / max(1, n_trades))
    sharpe = (avg_ret / (std_ret + 1e-10)) * math.sqrt(trades_per_year) if std_ret > 0 else 0

    # Max drawdown
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-10)
    max_dd = float(np.min(dd))

    # Per-regime stats
    regime_stats = {}
    for t in trades:
        if t.regime not in regime_stats:
            regime_stats[t.regime] = {"n": 0, "wins": 0, "pnl": 0}
        regime_stats[t.regime]["n"] += 1
        regime_stats[t.regime]["wins"] += int(t.pnl > 0)
        regime_stats[t.regime]["pnl"] += t.pnl

    return {
        "n_trades": n_trades,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": round(len(wins) / n_trades, 4) if n_trades else 0,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return * 100, 2),
        "avg_pnl_per_trade": round(total_pnl / n_trades, 2) if n_trades else 0,
        "avg_win": round(np.mean([t.pnl for t in wins]), 2) if wins else 0,
        "avg_loss": round(np.mean([t.pnl for t in losses]), 2) if losses else 0,
        "profit_factor": round(
            abs(sum(t.pnl for t in wins)) / (abs(sum(t.pnl for t in losses)) + 1e-10), 2
        ) if losses else float("inf"),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "final_equity": round(equity[-1], 2) if equity else initial_capital,
        "per_regime": regime_stats,
    }


# ═══════════════════════════════════════════════════════════════
# KALSHI SCANNER ADAPTER
# ═══════════════════════════════════════════════════════════════

def scan_kalshi_opportunities(
    forecast_result,
    min_edge: float = 0.10,
) -> List[Dict]:
    """Scan for Kalshi opportunities using ClawdBot forecast + ArbitrageDetector.

    Args:
        forecast_result: ForecastResult from ClawdBot engine
        min_edge: minimum model-vs-market probability edge

    Returns:
        List of opportunity dicts
    """
    from scripts.forecaster.regime_gate import ArbitrageDetector

    detector = ArbitrageDetector()
    opportunities = []

    # Check barrier probabilities from forecast
    targets = forecast_result.targets
    if targets.barrier_strike > 0:
        model_prob = targets.barrier_above_prob
        # Estimate market prob from quantiles (rough)
        market_prob = 0.50  # placeholder; real impl needs Kalshi API

        opp = detector.check_model_edge_arb(
            model_prob=model_prob,
            market_prob=market_prob,
            yes_ask=int(market_prob * 100),
            no_ask=int((1 - market_prob) * 100),
            ticker=f"{forecast_result.symbol}_barrier_{targets.barrier_strike}",
            min_edge=min_edge,
        )
        if opp:
            opportunities.append(asdict(opp))

    return opportunities


# ═══════════════════════════════════════════════════════════════
# UNIFIED FORECAST + VALIDATE PIPELINE
# ═══════════════════════════════════════════════════════════════

@dataclass
class UnifiedResult:
    """Complete result from forecast + KPCOFGS + validation + LLM reasoning."""
    # ClawdBot forecast
    forecast: Dict[str, Any] = field(default_factory=dict)

    # KPCOFGS enrichment
    kpcofgs: Dict[str, Any] = field(default_factory=dict)
    kpcofgs_regime: str = "range"

    # Yoshi scoring (if validation run)
    validation: Dict[str, Any] = field(default_factory=dict)

    # Backtest (if backtest run)
    backtest: Dict[str, Any] = field(default_factory=dict)

    # Kalshi opportunities
    opportunities: List[Dict] = field(default_factory=list)

    # LLM Reasoning (if reasoning run)
    reasoning: Dict[str, Any] = field(default_factory=dict)

    # Timing
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def run_unified(
    symbol: str = "BTCUSDT",
    horizon_hours: float = 24.0,
    bars_limit: int = 2000,
    mode: str = "forecast",  # "forecast", "validate", "backtest", "full", "reason"
    mc_iterations: int = 50_000,
    verbose: bool = True,
    reasoning_mode: str = "auto",  # "auto", "full", "regime", "trade", "risk", "extrapolation", "critique", "off"
    risk_budget_usd: float = 500.0,
    max_leverage: float = 2.0,
) -> UnifiedResult:
    """Run the unified ClawdBot + Yoshi + LLM pipeline.

    Modes:
        forecast:  ClawdBot 14-paradigm + KPCOFGS enrichment
        validate:  + walk-forward validation with purge/embargo
        backtest:  + backtest with PnL/Sharpe
        full:      forecast + validate + backtest + Kalshi scan + LLM reasoning
        reason:    forecast + KPCOFGS + LLM reasoning (skip heavy validation)

    Reasoning modes:
        auto:          Engine picks best mode based on data
        full:          Complete analysis with trade suggestion
        regime:        Deep KPCOFGS regime analysis
        trade:         Specific trade plan with entry/exit/sizing
        risk:          Risk assessment and hedging
        extrapolation: Forward-looking scenarios
        critique:      Self-critique of forecast weaknesses
        off:           Skip LLM reasoning
    """
    t0 = time.time()
    result = UnifiedResult()

    Forecaster, ForecastResult = _get_forecaster()
    Bar, MarketSnapshot, PredictionTargets, Regime = _get_schemas()

    # ── Step 1: Fetch data ─────────────────────────────
    if verbose:
        print(f"{'='*60}")
        print(f"UNIFIED PIPELINE: {symbol} | {horizon_hours}h | mode={mode}")
        print(f"{'='*60}")
        print(f"\n[1/5] Fetching {bars_limit} bars from Coinbase...")

    from scripts.forecaster.data import fetch_market_snapshot
    snap = fetch_market_snapshot(symbol, bars_limit=bars_limit)

    if verbose:
        print(f"  Got {len(snap.bars_1h)} bars, current price: ${snap.current_price:,.2f}")

    # ── Step 2: ClawdBot forecast ──────────────────────
    if verbose:
        print(f"\n[2/5] Running 14-paradigm ensemble forecast...")

    fc = Forecaster(
        mc_iterations=mc_iterations,
        mc_steps=48,
        enable_mc=True,
    )
    fc_result = fc.forecast_from_snapshot(snap, horizon_hours=horizon_hours)

    result.forecast = {
        "symbol": fc_result.symbol,
        "current_price": fc_result.current_price,
        "predicted_price": fc_result.predicted_price,
        "direction": fc_result.direction,
        "confidence": fc_result.confidence,
        "volatility": fc_result.volatility,
        "regime": fc_result.regime,
        "modules_run": len([k for k, v in fc_result.module_outputs.items()
                            if v.get("confidence", 0) > 0]),
        "var_95": fc_result.mc_summary.get("var_95", 0),
        "var_99": fc_result.mc_summary.get("var_99", 0),
        "price_q05": fc_result.price_q05,
        "price_q50": fc_result.price_q50,
        "price_q95": fc_result.price_q95,
        "gating_weights": fc_result.gating_weights,
        "gate_decision": fc_result.module_outputs.get("regime_gate", {}),
        "elapsed_ms": fc_result.elapsed_ms,
    }

    if verbose:
        d = result.forecast
        arrow = "↑" if d["direction"] == "up" else "↓" if d["direction"] == "down" else "↔"
        print(f"  {d['symbol']} {arrow} ${d['predicted_price']:,.2f} "
              f"({d['confidence']*100:.1f}% conf)")
        print(f"  Regime: {d['regime']} | Vol: {d['volatility']:.4f} | "
              f"VaR95: {d['var_95']*100:.2f}%")
        gate = d.get("gate_decision", {})
        print(f"  Gate: {gate.get('action', '?')} | EV: {gate.get('ev_edge', 0):.4f} "
              f"(min: {gate.get('min_ev_required', 0.04):.4f})")

    # ── Step 3: KPCOFGS regime enrichment ──────────────
    if verbose:
        print(f"\n[3/5] Running KPCOFGS 7-level regime classification...")

    bars_df = snapshot_to_dataframe(snap.bars_1h)
    if not bars_df.empty:
        classified_df, kpcofgs_summary = classify_kpcofgs(bars_df)
        result.kpcofgs = kpcofgs_summary
        result.kpcofgs_regime = kpcofgs_to_regime(kpcofgs_summary)

        if verbose:
            print(f"  K: {kpcofgs_summary.get('K_label', '?')} "
                  f"(p={kpcofgs_summary.get('K_pmax', 0):.2f})")
            print(f"  S: {kpcofgs_summary.get('S_label', '?')} "
                  f"(p={kpcofgs_summary.get('S_pmax', 0):.2f})")
            print(f"  Regime entropy: {kpcofgs_summary.get('regime_entropy', 0):.3f}")
            print(f"  Mapped to ClawdBot regime: {result.kpcofgs_regime}")

    # ── Step 4: Walk-forward validation ────────────────
    if mode in ("validate", "full"):
        if verbose:
            print(f"\n[4/5] Running walk-forward validation (purge/embargo)...")

        def _forecast_fn(bar_window):
            """Adapter: bars list → forecast dict."""
            from scripts.forecaster.schemas import MarketSnapshot, Bar
            mini_snap = MarketSnapshot(symbol=symbol)
            mini_snap.bars_1h = bar_window
            mini_result = fc.forecast_from_snapshot(mini_snap, horizon_hours=horizon_hours)
            return {
                "expected_return": mini_result.targets.expected_return,
                "direction_prob": mini_result.targets.direction_prob,
                "quantile_05": mini_result.targets.quantile_10 * 0.7,  # approx Q05
                "quantile_10": mini_result.targets.quantile_10,
                "quantile_50": mini_result.targets.quantile_50,
                "quantile_90": mini_result.targets.quantile_90,
                "quantile_95": mini_result.targets.quantile_90 * 1.3,  # approx Q95
                "volatility_forecast": mini_result.targets.volatility_forecast,
                "regime": mini_result.regime,
            }

        wf_config = WalkForwardConfig(
            n_outer_folds=5,
            train_bars=400,
            val_bars=50,
            test_bars=50,
            purge_bars=24,
            embargo_bars=12,
            horizon_bars=int(horizon_hours),
        )

        folds, agg = run_walk_forward(
            bars=snap.bars_1h,
            forecaster_fn=_forecast_fn,
            config=wf_config,
            verbose=verbose,
        )
        result.validation = agg
    else:
        if verbose:
            print(f"\n[4/5] Walk-forward validation: SKIPPED (mode={mode})")

    # ── Step 5: Backtest ──────────────────────────────
    if mode in ("backtest", "full"):
        if verbose:
            print(f"\n[5/5] Running backtest...")

        def _bt_forecast_fn(bar_window):
            """Adapter for backtest."""
            from scripts.forecaster.schemas import MarketSnapshot
            mini_snap = MarketSnapshot(symbol=symbol)
            mini_snap.bars_1h = bar_window
            mini_result = fc.forecast_from_snapshot(mini_snap, horizon_hours=horizon_hours)
            return {
                "direction_prob": mini_result.targets.direction_prob,
                "regime": mini_result.regime,
                "expected_return": mini_result.targets.expected_return,
            }

        bt_result = run_backtest(
            bars=snap.bars_1h,
            forecaster_fn=_bt_forecast_fn,
            config=BacktestConfig(),
            horizon_bars=int(horizon_hours),
            verbose=verbose,
        )
        result.backtest = bt_result.stats
    else:
        if verbose:
            print(f"\n[5/5] Backtest: SKIPPED (mode={mode})")

    # ── Step 6: LLM Reasoning ────────────────────────────
    enable_reasoning = (
        reasoning_mode != "off"
        and mode in ("forecast", "full", "reason")
    )

    if enable_reasoning:
        if verbose:
            print(f"\n[6/6] Running LLM reasoning ({reasoning_mode} mode)...")

        try:
            from gnosis.reasoning import ReasoningEngine, ReasoningConfig, AnalysisMode, LLMConfig

            # Map string to AnalysisMode enum
            _mode_map = {
                "auto": AnalysisMode.AUTO,
                "full": AnalysisMode.FULL_ANALYSIS,
                "regime": AnalysisMode.REGIME_DEEP_DIVE,
                "trade": AnalysisMode.TRADE_PLAN,
                "risk": AnalysisMode.RISK_ASSESSMENT,
                "extrapolation": AnalysisMode.EXTRAPOLATION,
                "critique": AnalysisMode.SELF_CRITIQUE,
            }
            analysis_mode = _mode_map.get(reasoning_mode, AnalysisMode.AUTO)

            reasoning_config = ReasoningConfig(
                mode=analysis_mode,
                llm_config=LLMConfig.from_yaml(),
                risk_budget_usd=risk_budget_usd,
                max_leverage=max_leverage,
                verbose=verbose,
            )

            engine = ReasoningEngine(reasoning_config)
            reasoning_result = engine.analyze(
                forecast=result.forecast,
                kpcofgs=result.kpcofgs,
                kpcofgs_regime=result.kpcofgs_regime,
                validation=result.validation if result.validation else None,
                backtest=result.backtest if result.backtest else None,
                opportunities=result.opportunities if result.opportunities else None,
                mode=analysis_mode,
            )

            result.reasoning = reasoning_result.to_dict()

            if verbose:
                reasoning_result.print_summary()

        except Exception as e:
            result.reasoning = {"error": str(e)}
            if verbose:
                print(f"  LLM reasoning error: {e}")
    else:
        if verbose and mode not in ("validate", "backtest"):
            print(f"\n[6/6] LLM Reasoning: SKIPPED (reasoning_mode={reasoning_mode})")

    result.elapsed_ms = round((time.time() - t0) * 1000, 1)

    if verbose:
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE in {result.elapsed_ms:.0f}ms")
        if result.validation:
            v = result.validation
            print(f"  Walk-forward: HR={v.get('hit_rate', 0):.1%} "
                  f"(p={v.get('hit_rate_p_value', 1):.4f}), "
                  f"cov={v.get('coverage_90', 0):.1%}, "
                  f"CRPS={v.get('crps', 0):.6f}, "
                  f"folds={v.get('n_folds', 0)}")
        if result.backtest:
            b = result.backtest
            print(f"  Backtest: {b.get('n_trades', 0)} trades, "
                  f"WR={b.get('win_rate', 0):.1%}, "
                  f"PnL=${b.get('total_pnl', 0):+.2f} "
                  f"({b.get('total_return_pct', 0):+.1f}%), "
                  f"Sharpe={b.get('sharpe', 0):.3f}, "
                  f"MaxDD={b.get('max_drawdown_pct', 0):.1f}%")
        if result.reasoning and not result.reasoning.get("error"):
            analysis = result.reasoning.get("analysis", {})
            sq = analysis.get("signal_quality", analysis.get("overall_assessment", "?"))
            ts = analysis.get("trade_suggestion", {})
            action = ts.get("action", "?") if ts else "?"
            print(f"  LLM Reasoning: signal={sq}, action={action}")
        print(f"{'='*60}")

    return result
