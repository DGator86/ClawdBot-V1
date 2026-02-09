#!/usr/bin/env python3
"""
Monte Carlo Simulation Engine for BTCUSDT Price Forecasting
============================================================
Now powered by the 12-paradigm ensemble forecaster with regime-conditioned
jump diffusion Monte Carlo.

Two modes:
  1. --live:  Fetch real market data, run full 12-module ensemble + MC
  2. Default: Use hardcoded prediction (legacy dashboard compatibility)

Outputs JSON results consumed by the web dashboard (index.html).

Usage:
    python3 simulation.py                          # legacy mode (hardcoded)
    python3 simulation.py --live                    # live forecaster mode
    python3 simulation.py --live --iterations 500000
    python3 simulation.py --live --barrier 100000   # with Kalshi barrier
    python3 simulation.py --live --symbol ETHUSDT
"""

import json
import math
import time
import sys
import os
from datetime import datetime, timezone

import numpy as np

# ──────────────────────────────────────────────────────────────
# Legacy Prediction Input (fallback when forecaster unavailable)
# ──────────────────────────────────────────────────────────────
PREDICTION = {
    "symbol": "BTCUSDT",
    "timestamp": "2026-02-06T00:00:00Z",
    "current_price": 63207.00,
    "predicted_price": 60452.47,
    "direction": "Down",
    "confidence": 0.7716,
    "volatility": 0.0272,
    "quantiles": {
        "q05": 57158.91,
        "q50": 60452.47,
        "q95": 63746.02,
    },
}


def run_simulation(
    current_price: float,
    predicted_price: float,
    volatility: float,
    confidence: float,
    n_iterations: int = 100_000,
    n_steps: int = 48,
    dt: float = 1 / 48,
    seed: int = 42,
    # New: forecaster-derived params for regime-conditioned MC
    jump_prob: float = 0.0,
    crash_prob: float = 0.0,
    regime: str = "range",
    quantiles: dict = None,
    # Ultimate-fix: regime-conditioned parameters
    regime_probs: dict = None,
    confidence_scalar: float = 0.70,
) -> dict:
    """
    Run a Monte Carlo simulation using Geometric Brownian Motion
    with optional jump diffusion (when forecaster provides jump parameters).

    The drift (mu) is calibrated from the predicted price so that the
    expected terminal value equals the model's forecast.
    """
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()

    # ── Drift calibration ────────────────────────────────────
    T = 1.0
    log_return = math.log(predicted_price / current_price)
    mu = log_return
    sigma = volatility * confidence_scalar  # scale by confidence

    # ── Regime-conditioned volatility adjustment ─────────────
    # Different regimes have different vol characteristics
    if regime_probs:
        vol_adj = 1.0
        if regime_probs.get("trend_up", 0) > 0.3:
            vol_adj = 0.85  # trends are lower vol
        elif regime_probs.get("cascade_risk", 0) > 0.3:
            vol_adj = 1.5   # cascade = high vol
        elif regime_probs.get("post_jump", 0) > 0.3:
            vol_adj = 1.2   # elevated vol after jumps
        elif regime_probs.get("vol_expansion", 0) > 0.3:
            vol_adj = 1.3   # expanding vol
        sigma *= vol_adj
    sigma = volatility

    # ── Generate price paths ─────────────────────────────────
    Z = rng.standard_normal((n_iterations, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z

    # Jump diffusion (from forecaster ensemble)
    jump_lambda = jump_prob * n_steps
    if jump_lambda > 0.01:
        jump_mu = -0.02 if crash_prob > jump_prob * 0.4 else 0.0
        # Jump sizes use raw vol (not scaled) to preserve tail accuracy
        jump_sigma = volatility * 2
        jump_sigma = sigma * 2
        N_jumps = rng.poisson(jump_lambda * dt, (n_iterations, n_steps))
        J_sizes = rng.normal(jump_mu, jump_sigma, (n_iterations, n_steps))
        increments += N_jumps * J_sizes

    # Cumulative sum of log-returns -> price paths
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_iterations, 1)), log_paths])
    paths = current_price * np.exp(log_paths)

    terminal_prices = paths[:, -1]

    # ── Core statistics ──────────────────────────────────────
    mean_terminal = float(np.mean(terminal_prices))
    median_terminal = float(np.median(terminal_prices))
    std_terminal = float(np.std(terminal_prices))

    percentiles_list = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_values = {
        f"p{p}": float(np.percentile(terminal_prices, p))
        for p in percentiles_list
    }

    # Direction alignment
    if predicted_price < current_price:
        paths_aligned = float(np.mean(terminal_prices < current_price))
    else:
        paths_aligned = float(np.mean(terminal_prices > current_price))

    # Quantile range hit rate
    q05 = (quantiles or {}).get("q05", PREDICTION["quantiles"]["q05"])
    q95 = (quantiles or {}).get("q95", PREDICTION["quantiles"]["q95"])
    in_range = float(np.mean((terminal_prices >= q05) & (terminal_prices <= q95)))

    # ── Risk metrics ─────────────────────────────────────────
    returns = terminal_prices / current_price - 1.0

    var_95 = float(np.percentile(returns, 5))
    var_99 = float(np.percentile(returns, 1))
    cvar_95 = float(np.mean(returns[returns <= var_95]))
    cvar_99 = float(np.mean(returns[returns <= var_99]))

    max_drawdowns = (np.min(paths, axis=1) / current_price - 1.0)
    avg_max_drawdown = float(np.mean(max_drawdowns))
    worst_drawdown = float(np.min(max_drawdowns))

    mean_return = float(np.mean(returns))
    sharpe = mean_return / float(np.std(returns)) if np.std(returns) > 0 else 0.0

    # ── Distribution shape ───────────────────────────────────
    hist_counts, hist_edges = np.histogram(terminal_prices, bins=200)
    histogram_data = {
        "counts": hist_counts.tolist(),
        "edges": hist_edges.tolist(),
    }

    # ── Sampled paths for visualisation (50 paths) ───────────
    sample_idx = rng.choice(
        n_iterations, size=min(50, n_iterations), replace=False
    )
    sampled_paths = paths[sample_idx].tolist()

    # ── Percentile envelope (for fan chart) ──────────────────
    envelope = {}
    for p in [5, 10, 25, 50, 75, 90, 95]:
        envelope[f"p{p}"] = np.percentile(paths, p, axis=0).tolist()

    # ── Convergence check (running mean) ─────────────────────
    batch_size = max(1, n_iterations // 100)
    convergence = []
    for i in range(batch_size, n_iterations + 1, batch_size):
        convergence.append({
            "n": i,
            "mean": float(np.mean(terminal_prices[:i])),
            "std": float(np.std(terminal_prices[:i])),
        })

    elapsed = time.perf_counter() - t_start

    # ── Accuracy metrics ─────────────────────────────────────
    prediction_error = abs(mean_terminal - predicted_price) / current_price
    mean_accuracy = 1.0 - prediction_error
    mc_confidence = paths_aligned * mean_accuracy

    # MC-derived jump stats
    mc_returns = np.log(terminal_prices / current_price)
    mc_jump_prob = float(np.mean(np.abs(mc_returns) > 2 * sigma))
    mc_crash_prob = float(np.mean(mc_returns < -3 * sigma))

    model_name = "Jump Diffusion GBM (12-module ensemble)" if jump_lambda > 0.01 \
        else "Geometric Brownian Motion (GBM)"

    return {
        "meta": {
            "symbol": PREDICTION.get("symbol", "BTCUSDT"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iterations": n_iterations,
            "steps": n_steps,
            "elapsed_seconds": round(elapsed, 3),
            "seed": seed,
            "model": model_name,
            "regime": regime,
            "jump_diffusion": jump_lambda > 0.01,
        },
        "input": {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "direction": PREDICTION.get("direction", "Flat"),
            "confidence": confidence,
            "volatility": volatility,
            "quantiles": quantiles or PREDICTION.get("quantiles", {}),
            "jump_prob": jump_prob,
            "crash_prob": crash_prob,
        },
        "terminal": {
            "mean": round(mean_terminal, 2),
            "median": round(median_terminal, 2),
            "std": round(std_terminal, 2),
            "min": round(float(np.min(terminal_prices)), 2),
            "max": round(float(np.max(terminal_prices)), 2),
            "percentiles": {k: round(v, 2) for k, v in pct_values.items()},
        },
        "validation": {
            "paths_aligned_pct": round(paths_aligned * 100, 2),
            "quantile_range_hit_pct": round(in_range * 100, 2),
            "mean_accuracy": round(mean_accuracy * 100, 4),
            "mc_confidence": round(mc_confidence * 100, 2),
            "validated": mc_confidence > 0.5,
            "prediction_error_pct": round(prediction_error * 100, 4),
            "mc_jump_prob": round(mc_jump_prob, 4),
            "mc_crash_prob": round(mc_crash_prob, 4),
        },
        "risk": {
            "var_95_pct": round(var_95 * 100, 4),
            "var_99_pct": round(var_99 * 100, 4),
            "cvar_95_pct": round(cvar_95 * 100, 4),
            "cvar_99_pct": round(cvar_99 * 100, 4),
            "avg_max_drawdown_pct": round(avg_max_drawdown * 100, 4),
            "worst_drawdown_pct": round(worst_drawdown * 100, 4),
            "sharpe_ratio": round(sharpe, 4),
            "mean_return_pct": round(mean_return * 100, 4),
            "std_return_pct": round(float(np.std(returns)) * 100, 4),
        },
        "histogram": histogram_data,
        "sampled_paths": sampled_paths,
        "envelope": envelope,
        "convergence": convergence,
    }


def run_live_simulation(
    symbol: str = "BTCUSDT",
    horizon_hours: float = 24.0,
    n_iterations: int = 100_000,
    n_steps: int = 48,
    barrier_strike: float = None,
    seed: int = 42,
    output_path: str = None,
) -> dict:
    """
    Run Monte Carlo powered by the 12-paradigm ensemble forecaster.

    Fetches live OHLCV data, runs all 12 modules + regime detection,
    then uses the ensemble output to parameterize jump-diffusion MC.
    """
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from scripts.forecaster.engine import Forecaster

    print(f"Running 12-paradigm ensemble forecast for {symbol} "
          f"(horizon={horizon_hours}h)...")

    fc = Forecaster(
        mc_iterations=n_iterations,
        mc_steps=n_steps,
        mc_seed=seed,
        enable_mc=True,
    )

    result = fc.forecast(
        symbol=symbol,
        horizon_hours=horizon_hours,
        barrier_strike=barrier_strike,
    )

    print(f"  Ensemble: {result.modules_run}/12 modules in {result.elapsed_ms:.0f}ms")
    print(f"  Regime: {result.regime} (conf scalar: {result.confidence_scalar:.2f})")

    # Update global PREDICTION for legacy compatibility
    global PREDICTION
    PREDICTION = {
        "symbol": symbol,
        "timestamp": result.timestamp,
        "current_price": result.current_price,
        "predicted_price": result.predicted_price,
        "direction": result.direction,
        "confidence": result.confidence,
        "volatility": result.volatility,
        "quantiles": {
            "q05": result.price_q05,
            "q50": result.price_q50,
            "q95": result.price_q95,
        },
    }

    # Run standalone MC with forecaster params (for dashboard output format)
    mc_results = run_simulation(
        current_price=result.current_price,
        predicted_price=result.predicted_price,
        volatility=result.volatility,
        confidence=result.confidence,
        n_iterations=n_iterations,
        n_steps=n_steps,
        seed=seed,
        jump_prob=result.jump_prob,
        crash_prob=result.crash_prob,
        regime=result.regime,
        quantiles=PREDICTION["quantiles"],
    )

    # Enrich with forecaster ensemble data
    mc_results["ensemble"] = {
        "regime": result.regime,
        "regime_probs": result.regime_probs,
        "confidence_scalar": result.confidence_scalar,
        "modules_run": result.modules_run,
        "elapsed_ms": result.elapsed_ms,
        "gating_weights": result.gating_weights,
        "direction": result.direction,
        "direction_confidence": result.confidence,
        "predicted_price": result.predicted_price,
        "jump_prob": result.jump_prob,
        "crash_prob": result.crash_prob,
        # MC-derived risk metrics from the ensemble
        "mc_var_95": result.var_95,
        "mc_var_99": result.var_99,
        "mc_cvar_95": result.cvar_95,
        "mc_cvar_99": result.cvar_99,
        # Quantile prices
        "price_q05": result.price_q05,
        "price_q10": result.price_q10,
        "price_q25": result.price_q25,
        "price_q50": result.price_q50,
        "price_q75": result.price_q75,
        "price_q90": result.price_q90,
        "price_q95": result.price_q95,
        # Module-level detail
        "module_outputs": result.module_outputs,
    }

    # Add barrier data if available
    if barrier_strike and barrier_strike > 0:
        mc_results["barrier"] = {
            "strike": barrier_strike,
            "above_prob": result.barrier_above_prob,
            "below_prob": result.barrier_below_prob,
        }

    # Save results
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(mc_results, f, indent=2)

    return mc_results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Monte Carlo BTCUSDT Simulation (with optional 12-paradigm ensemble)"
    )
    parser.add_argument("--iterations", "-n", type=int, default=100_000)
    parser.add_argument("--steps", "-s", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", type=str, default=None)
    # New: live forecaster mode
    parser.add_argument("--live", action="store_true",
                        help="Use live data + 12-paradigm ensemble forecaster")
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Symbol to forecast (default: BTCUSDT)")
    parser.add_argument("--horizon", type=float, default=24.0,
                        help="Forecast horizon in hours (default: 24)")
    parser.add_argument("--barrier", type=float, default=None,
                        help="Barrier strike for Kalshi probability")
    args = parser.parse_args()

    if args.live:
        # ── Live mode: full 12-paradigm ensemble ──────────────
        print(f"LIVE MODE: 12-paradigm ensemble + MC "
              f"({args.iterations:,} iterations)")
        results = run_live_simulation(
            symbol=args.symbol,
            horizon_hours=args.horizon,
            n_iterations=args.iterations,
            n_steps=args.steps,
            barrier_strike=args.barrier,
            seed=args.seed,
            output_path=args.output,
        )
    else:
        # ── Legacy mode: hardcoded prediction ─────────────────
        print(f"LEGACY MODE: {args.iterations:,} iterations, {args.steps} steps")
        print("  (Use --live for 12-paradigm ensemble)")
        results = run_simulation(
            current_price=PREDICTION["current_price"],
            predicted_price=PREDICTION["predicted_price"],
            volatility=PREDICTION["volatility"],
            confidence=PREDICTION["confidence"],
            n_iterations=args.iterations,
            n_steps=args.steps,
            seed=args.seed,
        )

        out_path = args.output or os.path.join(
            os.path.dirname(__file__), "results.json"
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # Print summary
    t = results["terminal"]
    v = results["validation"]
    r = results["risk"]
    m = results["meta"]

    print(f"\n{'='*60}")
    print(f"  MONTE CARLO SIMULATION -- {m['symbol']}")
    print(f"{'='*60}")
    print(f"  Iterations:       {m['iterations']:>12,}")
    print(f"  Elapsed:          {m['elapsed_seconds']:>12.3f}s")
    print(f"  Model:            {m['model']}")
    if m.get("regime"):
        print(f"  Regime:           {m['regime']:>12}")
    if m.get("jump_diffusion"):
        print(f"  Jump Diffusion:   {'ENABLED':>12}")
    print(f"{'---'*20}")
    i = results["input"]
    print(f"  Current Price:    ${i['current_price']:>12,.2f}")
    print(f"  Predicted Price:  ${i['predicted_price']:>12,.2f}")
    arrow = "DOWN" if i["direction"] == "Down" else "UP" if i["direction"] == "Up" else "FLAT"
    print(f"  Direction:        {arrow:>12}")
    print(f"{'---'*20}")
    print(f"  MC Mean:          ${t['mean']:>12,.2f}")
    print(f"  MC Median:        ${t['median']:>12,.2f}")
    print(f"  MC Std Dev:       ${t['std']:>12,.2f}")
    print(f"{'---'*20}")
    print(f"  Paths Aligned:    {v['paths_aligned_pct']:>11.2f}%")
    print(f"  MC Confidence:    {v['mc_confidence']:>11.2f}%")
    print(f"  Validated:        {'YES' if v['validated'] else 'NO':>12}")
    if "mc_jump_prob" in v:
        print(f"  MC Jump Prob:     {v['mc_jump_prob']*100:>11.2f}%")
        print(f"  MC Crash Prob:    {v['mc_crash_prob']*100:>11.2f}%")
    print(f"{'---'*20}")
    print(f"  VaR (95%):        {r['var_95_pct']:>11.4f}%")
    print(f"  VaR (99%):        {r['var_99_pct']:>11.4f}%")
    print(f"  CVaR (95%):       {r['cvar_95_pct']:>11.4f}%")
    print(f"  CVaR (99%):       {r['cvar_99_pct']:>11.4f}%")
    print(f"  Sharpe Ratio:     {r['sharpe_ratio']:>12.4f}")
    print(f"{'='*60}")

    # Print ensemble details if available
    if "ensemble" in results:
        e = results["ensemble"]
        print(f"\n  ENSEMBLE DETAILS:")
        print(f"    Modules:    {e['modules_run']}/12")
        print(f"    Regime:     {e['regime']}")
        print(f"    Jump Prob:  {e['jump_prob']*100:.2f}%")
        print(f"    Crash Prob: {e['crash_prob']*100:.2f}%")
        print(f"    MC VaR95:   {e['mc_var_95']*100:.3f}%")
        print(f"    MC CVaR95:  {e['mc_cvar_95']*100:.3f}%")

    if "barrier" in results:
        b = results["barrier"]
        print(f"\n  BARRIER @ ${b['strike']:,.2f}:")
        print(f"    P(above):   {b['above_prob']*100:.2f}%")
        print(f"    P(below):   {b['below_prob']*100:.2f}%")

    out_path = args.output or os.path.join(
        os.path.dirname(__file__), "results.json"
    )
    print(f"\n  Results saved to: {out_path}")

    return results


if __name__ == "__main__":
    main()
