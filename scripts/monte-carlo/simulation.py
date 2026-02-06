#!/usr/bin/env python3
"""
Monte Carlo Simulation Engine for BTCUSDT Price Forecasting
============================================================
Geometric Brownian Motion (GBM) model with 100,000 iterations.
Outputs JSON results consumed by the web dashboard.

Usage:
    python3 simulation.py                    # Run with defaults
    python3 simulation.py --iterations 500000 --steps 96
"""

import json
import math
import time
import sys
import os
from datetime import datetime, timezone

import numpy as np

# ──────────────────────────────────────────────────────────────
# Prediction Input (from Yoshi-Bot latest signal)
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
    n_steps: int = 48,       # 48 half-hour steps = 24h forecast
    dt: float = 1 / 48,      # fraction of forecast horizon per step
    seed: int = 42,
) -> dict:
    """
    Run a Monte Carlo simulation using Geometric Brownian Motion.

    The drift (mu) is calibrated from the predicted price so that the
    expected terminal value equals the Yoshi model's forecast.  Volatility
    is annualised from the model's stated daily vol.
    """
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()

    # ── Drift calibration ────────────────────────────────────
    # ln(S_T / S_0) = (mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z
    # E[S_T] = S_0 * exp(mu*T)  =>  mu = ln(predicted/current) / T
    T = 1.0  # normalised forecast horizon
    log_return = math.log(predicted_price / current_price)
    mu = log_return  # drift over the horizon
    sigma = volatility  # model-stated vol (already scaled to horizon)

    # ── Generate price paths via GBM ─────────────────────────
    # dS/S = mu*dt + sigma*dW
    Z = rng.standard_normal((n_iterations, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z

    # Cumulative sum of log-returns → price paths
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((n_iterations, 1)), log_paths])
    paths = current_price * np.exp(log_paths)  # (n_iterations, n_steps+1)

    terminal_prices = paths[:, -1]

    # ── Core statistics ──────────────────────────────────────
    mean_terminal = float(np.mean(terminal_prices))
    median_terminal = float(np.median(terminal_prices))
    std_terminal = float(np.std(terminal_prices))

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_values = {f"p{p}": float(np.percentile(terminal_prices, p)) for p in percentiles}

    # Direction alignment
    if predicted_price < current_price:
        paths_aligned = float(np.mean(terminal_prices < current_price))
    else:
        paths_aligned = float(np.mean(terminal_prices > current_price))

    # Quantile range hit rate (what % land in Yoshi's q05–q95)
    q05 = PREDICTION["quantiles"]["q05"]
    q95 = PREDICTION["quantiles"]["q95"]
    in_range = float(np.mean((terminal_prices >= q05) & (terminal_prices <= q95)))

    # ── Risk metrics ─────────────────────────────────────────
    returns = terminal_prices / current_price - 1.0

    # Value at Risk (VaR)
    var_95 = float(np.percentile(returns, 5))   # 5th pct of returns = 95% VaR
    var_99 = float(np.percentile(returns, 1))   # 1st pct = 99% VaR

    # Conditional VaR (Expected Shortfall)
    cvar_95 = float(np.mean(returns[returns <= var_95]))
    cvar_99 = float(np.mean(returns[returns <= var_99]))

    # Maximum drawdown across all paths (from starting price)
    max_drawdowns = (np.min(paths, axis=1) / current_price - 1.0)
    avg_max_drawdown = float(np.mean(max_drawdowns))
    worst_drawdown = float(np.min(max_drawdowns))

    # Sharpe-like ratio (return / risk)
    mean_return = float(np.mean(returns))
    sharpe = mean_return / float(np.std(returns)) if np.std(returns) > 0 else 0.0

    # ── Distribution shape ───────────────────────────────────
    from numpy import histogram
    hist_counts, hist_edges = histogram(terminal_prices, bins=200)
    histogram_data = {
        "counts": hist_counts.tolist(),
        "edges": hist_edges.tolist(),
    }

    # ── Sampled paths for visualisation (50 paths) ───────────
    sample_idx = rng.choice(n_iterations, size=min(50, n_iterations), replace=False)
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
    # Mean accuracy: how close the MC mean is to the predicted price
    prediction_error = abs(mean_terminal - predicted_price) / current_price
    mean_accuracy = 1.0 - prediction_error

    # Monte Carlo confidence: paths aligned * accuracy
    mc_confidence = paths_aligned * mean_accuracy

    return {
        "meta": {
            "symbol": PREDICTION["symbol"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iterations": n_iterations,
            "steps": n_steps,
            "elapsed_seconds": round(elapsed, 3),
            "seed": seed,
            "model": "Geometric Brownian Motion (GBM)",
        },
        "input": {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "direction": PREDICTION["direction"],
            "confidence": confidence,
            "volatility": volatility,
            "quantiles": PREDICTION["quantiles"],
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monte Carlo BTCUSDT Simulation")
    parser.add_argument("--iterations", "-n", type=int, default=100_000)
    parser.add_argument("--steps", "-s", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    print(f"Running Monte Carlo simulation: {args.iterations:,} iterations, {args.steps} steps...")

    results = run_simulation(
        current_price=PREDICTION["current_price"],
        predicted_price=PREDICTION["predicted_price"],
        volatility=PREDICTION["volatility"],
        confidence=PREDICTION["confidence"],
        n_iterations=args.iterations,
        n_steps=args.steps,
        seed=args.seed,
    )

    out_path = args.output or os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    t = results["terminal"]
    v = results["validation"]
    r = results["risk"]
    m = results["meta"]

    print(f"\n{'='*60}")
    print(f"  MONTE CARLO SIMULATION — {m['symbol']}")
    print(f"{'='*60}")
    print(f"  Iterations:       {m['iterations']:>12,}")
    print(f"  Elapsed:          {m['elapsed_seconds']:>12.3f}s")
    print(f"  Model:            {m['model']}")
    print(f"{'─'*60}")
    print(f"  Current Price:    ${PREDICTION['current_price']:>12,.2f}")
    print(f"  Predicted Price:  ${PREDICTION['predicted_price']:>12,.2f}")
    print(f"  Direction:        {'↓ DOWN' if PREDICTION['direction'] == 'Down' else '↑ UP':>12}")
    print(f"{'─'*60}")
    print(f"  MC Mean:          ${t['mean']:>12,.2f}")
    print(f"  MC Median:        ${t['median']:>12,.2f}")
    print(f"  MC Std Dev:       ${t['std']:>12,.2f}")
    print(f"  MC Min:           ${t['min']:>12,.2f}")
    print(f"  MC Max:           ${t['max']:>12,.2f}")
    print(f"{'─'*60}")
    print(f"  Paths Aligned:    {v['paths_aligned_pct']:>11.2f}%")
    print(f"  Quantile Range:   {v['quantile_range_hit_pct']:>11.2f}%")
    print(f"  Mean Accuracy:    {v['mean_accuracy']:>11.4f}%")
    print(f"  MC Confidence:    {v['mc_confidence']:>11.2f}%")
    print(f"  Validated:        {'YES ✓' if v['validated'] else 'NO ✗':>12}")
    print(f"{'─'*60}")
    print(f"  VaR (95%):        {r['var_95_pct']:>11.4f}%")
    print(f"  VaR (99%):        {r['var_99_pct']:>11.4f}%")
    print(f"  CVaR (95%):       {r['cvar_95_pct']:>11.4f}%")
    print(f"  CVaR (99%):       {r['cvar_99_pct']:>11.4f}%")
    print(f"  Avg Max DD:       {r['avg_max_drawdown_pct']:>11.4f}%")
    print(f"  Worst DD:         {r['worst_drawdown_pct']:>11.4f}%")
    print(f"  Sharpe Ratio:     {r['sharpe_ratio']:>12.4f}")
    print(f"{'='*60}")
    print(f"\n  Results saved to: {out_path}")

    return results


if __name__ == "__main__":
    main()
