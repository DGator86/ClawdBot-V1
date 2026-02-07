"""
TensorTrade-Inspired RL Trading Environment
=============================================
A Gymnasium-compatible trading environment powered by the 12-paradigm
ensemble forecaster. Incorporates key lessons from TensorTrade research:

Key TensorTrade Findings Incorporated:
  1. PBR (Position-Based Returns) reward — dense per-step signal
  2. Commission-aware training — overtrading is the #1 killer
  3. Scale-invariant features — only ratios, z-scores, bounded values
  4. Confidence-gated actions — only trade when forecaster has edge
  5. Walk-forward only — no random splits for time series

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │                    ForecastTradingEnv                         │
  │                                                              │
  │  Observer ────> Agent ────> ActionScheme ────> Portfolio      │
  │  (12-module     (RL       (Confidence-     (USD + BTC       │
  │   forecaster)    policy)   Gated BSH)       wallets)        │
  │      ^                                          │            │
  │      └────────── RewardScheme <─────────────────┘            │
  │                  (Commission-                                │
  │                   Aware PBR)                                 │
  │                                                              │
  │  DataFeed ────> Exchange ────> Broker ────> Trades           │
  │  (paginated       (sim)        (exec)       (log)           │
  │   OHLCV)                                                    │
  └──────────────────────────────────────────────────────────────┘

Usage:
    from scripts.forecaster.rl_env import ForecastTradingEnv

    # Create environment
    env = ForecastTradingEnv(
        bars=bars,             # list of Bar objects
        initial_balance=10000,
        commission=0.001,      # 0.1% per trade
        horizon_hours=24,
        confidence_threshold=0.55,
    )

    # Standard Gymnasium loop
    obs, info = env.reset()
    for _ in range(1000):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Or run the built-in evaluator
    from scripts.forecaster.rl_env import evaluate_forecaster_as_trader
    results = evaluate_forecaster_as_trader(bars, commission=0.001)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .schemas import (
    Bar, MarketSnapshot, PredictionTargets, Regime, ModuleOutput,
)
from .engine import Forecaster, ForecastResult


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO TRACKER
# ═══════════════════════════════════════════════════════════════

@dataclass
class SimplePortfolio:
    """
    Minimal portfolio tracker for backtesting.
    Tracks USD and BTC wallets, net worth, and trade history.
    """
    initial_balance: float = 10_000.0
    commission: float = 0.001  # 0.1% per trade

    # State
    usd_balance: float = 0.0
    btc_balance: float = 0.0
    position: int = 1  # 1 = cash (USD), 0 = long (BTC)
    entry_price: float = 0.0

    # Tracking
    trades: list = field(default_factory=list)
    net_worth_history: list = field(default_factory=list)
    trade_count: int = 0
    total_commission_paid: float = 0.0

    def reset(self, initial_price: float):
        """Reset portfolio to initial state."""
        self.usd_balance = self.initial_balance
        self.btc_balance = 0.0
        self.position = 1  # start in cash
        self.entry_price = 0.0
        self.trades = []
        self.net_worth_history = [self.initial_balance]
        self.trade_count = 0
        self.total_commission_paid = 0.0

    def net_worth(self, current_price: float) -> float:
        """Current net worth in USD."""
        return self.usd_balance + self.btc_balance * current_price

    def execute_action(self, action: int, current_price: float, step: int) -> bool:
        """
        Execute a trading action.

        Actions (TensorTrade BSH style):
            0 = "I want to be LONG (in BTC)"
            1 = "I want to be in CASH (USD)"

        Returns True if a trade was executed.
        """
        if action == self.position:
            # No change — HOLD
            return False

        if current_price <= 0:
            return False

        if action == 0 and self.position == 1:
            # BUY: USD → BTC
            commission_cost = self.usd_balance * self.commission
            self.total_commission_paid += commission_cost
            buy_amount = self.usd_balance - commission_cost
            self.btc_balance = buy_amount / current_price
            self.usd_balance = 0.0
            self.entry_price = current_price
            self.position = 0
            self.trade_count += 1
            self.trades.append({
                "step": step,
                "side": "BUY",
                "price": current_price,
                "btc_qty": self.btc_balance,
                "commission": commission_cost,
                "net_worth": self.net_worth(current_price),
            })
            return True

        elif action == 1 and self.position == 0:
            # SELL: BTC → USD
            sell_value = self.btc_balance * current_price
            commission_cost = sell_value * self.commission
            self.total_commission_paid += commission_cost
            self.usd_balance = sell_value - commission_cost
            self.btc_balance = 0.0
            self.position = 1
            self.trade_count += 1
            pnl = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
            self.trades.append({
                "step": step,
                "side": "SELL",
                "price": current_price,
                "pnl_pct": pnl,
                "commission": commission_cost,
                "net_worth": self.net_worth(current_price),
            })
            return True

        return False

    def update_history(self, current_price: float):
        """Record current net worth."""
        self.net_worth_history.append(self.net_worth(current_price))


# ═══════════════════════════════════════════════════════════════
# OBSERVATION BUILDER (Scale-Invariant Features)
# ═══════════════════════════════════════════════════════════════

def build_observation(
    forecast: ForecastResult,
    portfolio: SimplePortfolio,
    current_price: float,
    bars_window: list[Bar],
) -> np.ndarray:
    """
    Build a scale-invariant observation vector from the forecast result.

    TensorTrade lesson: Only use returns, ratios, normalized values.
    Never raw prices. This generalizes across price levels.

    Returns a flat numpy array of ~25 features, all bounded to [-1, 1]
    or [0, 1] range.
    """
    features = []

    # ── 1. Forecaster direction & confidence (4 features) ──
    # Direction probability: already [0, 1]
    features.append(forecast.targets.direction_prob)
    # Confidence: distance from 0.5 (0 = uncertain, 0.5 = very confident)
    features.append(abs(forecast.targets.direction_prob - 0.5) * 2)
    # Expected return: tanh-bounded
    features.append(math.tanh(forecast.targets.expected_return * 20))
    # Return std / vol: bounded
    features.append(min(1.0, forecast.targets.volatility_forecast * 10))

    # ── 2. Regime features (8 features) ──
    for regime in Regime:
        prob = forecast.regime_probs.get(regime.value, 0.0)
        features.append(prob)

    # ── 3. Risk features (4 features) ──
    features.append(min(1.0, forecast.jump_prob * 5))
    features.append(min(1.0, forecast.crash_prob * 5))
    features.append(math.tanh(forecast.var_95 * 10) if forecast.var_95 else 0.0)
    features.append(min(1.0, forecast.confidence_scalar))

    # ── 4. MC distribution features (3 features) ──
    if forecast.mc_summary:
        mc_mean = forecast.mc_summary.get("mc_mean_return", 0)
        mc_std = forecast.mc_summary.get("mc_return_std", 0)
        mc_skew = forecast.mc_summary.get("mc_skew", 0)
        features.append(math.tanh(mc_mean * 20))
        features.append(min(1.0, mc_std * 10))
        features.append(math.tanh(mc_skew))
    else:
        features.extend([0.0, 0.0, 0.0])

    # ── 5. Barrier features (2 features) ──
    features.append(forecast.barrier_above_prob)
    features.append(forecast.barrier_below_prob)

    # ── 6. Price action features from raw bars (4 features) ──
    if len(bars_window) >= 24:
        closes = [b.close for b in bars_window]
        volumes = [b.volume for b in bars_window]

        # 1h return
        ret_1h = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0
        features.append(math.tanh(ret_1h * 100))

        # 24h return
        ret_24h = (closes[-1] - closes[-24]) / closes[-24] if closes[-24] > 0 else 0
        features.append(math.tanh(ret_24h * 20))

        # Volume ratio (current vs 20-period avg)
        vol_avg = sum(volumes[-20:]) / 20 if volumes[-20:] else 1
        vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1
        features.append(min(2.0, vol_ratio) / 2.0)  # [0, 1]

        # RSI-like bounded momentum
        gains = sum(max(0, closes[i] - closes[i-1]) for i in range(-14, 0))
        losses = sum(max(0, closes[i-1] - closes[i]) for i in range(-14, 0))
        if gains + losses > 0:
            features.append(gains / (gains + losses))  # [0, 1]
        else:
            features.append(0.5)
    else:
        features.extend([0.0, 0.0, 0.5, 0.5])

    # ── 7. Portfolio state features (2 features) ──
    # Current position (binary)
    features.append(float(portfolio.position == 0))  # 1 if long BTC
    # Unrealized PnL (if long)
    if portfolio.position == 0 and portfolio.entry_price > 0 and current_price > 0:
        unrealized = (current_price - portfolio.entry_price) / portfolio.entry_price
        features.append(math.tanh(unrealized * 10))
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# PBR REWARD SCHEME (Commission-Aware)
# ═══════════════════════════════════════════════════════════════

class PBRReward:
    """
    Position-Based Returns reward scheme from TensorTrade.

    R_t = (P_t - P_{t-1}) * position_t - commission_cost

    Key insight: This gives a learning signal EVERY step, not just
    when trades happen. The agent learns the cost of being on the
    wrong side of the market at every timestep.

    Commission awareness:
    - When a trade happens, subtract estimated commission from reward
    - This naturally teaches the agent that trading is expensive
    - Higher training commission → fewer trades → better generalization
    """

    def __init__(self, commission: float = 0.001,
                 trade_penalty: float = 0.0,
                 hold_bonus: float = 0.0):
        self.commission = commission
        self.trade_penalty = trade_penalty
        self.hold_bonus = hold_bonus
        self.prev_price = None
        self.position = -1  # -1 = cash, +1 = long
        self.prev_action = 1  # start in cash (action=1)

    def compute_reward(self, current_price: float, action: int,
                       traded: bool, portfolio_value: float) -> float:
        """
        Compute PBR reward.

        Args:
            current_price: Current asset price
            action: 0=long, 1=cash
            traded: Whether a trade was executed this step
            portfolio_value: Current portfolio net worth
        """
        if self.prev_price is None or self.prev_price <= 0:
            self.prev_price = current_price
            self.position = 1 if action == 0 else -1
            self.prev_action = action
            return 0.0

        # PBR: reward = position * (price_change / prev_price)
        # Normalized by price for scale invariance
        price_change = (current_price - self.prev_price) / self.prev_price
        pbr = self.position * price_change

        # Commission cost on trade
        commission_penalty = 0.0
        if traded:
            commission_penalty = -self.commission * 2  # round-trip approximation
            commission_penalty += self.trade_penalty

        # Hold bonus in flat markets
        hold_reward = 0.0
        if not traded and abs(price_change) < 0.001:
            hold_reward = self.hold_bonus

        reward = pbr + commission_penalty + hold_reward

        # Update state
        self.prev_price = current_price
        self.position = 1 if action == 0 else -1
        self.prev_action = action

        return reward

    def reset(self):
        self.prev_price = None
        self.position = -1
        self.prev_action = 1


# ═══════════════════════════════════════════════════════════════
# CONFIDENCE-GATED ACTION SCHEME
# ═══════════════════════════════════════════════════════════════

class ConfidenceGatedPolicy:
    """
    Uses the forecaster output to make trading decisions with
    confidence gating — only trades when the forecaster has edge.

    TensorTrade found: The #1 problem is overtrading (2000+ trades/month).
    Solution: Only trade when direction_prob deviates significantly from 0.50.

    Action mapping:
        Forecaster says direction_prob > threshold → action 0 (long)
        Forecaster says direction_prob < (1-threshold) → action 1 (cash)
        Otherwise → hold current position

    Additional filters:
        - Minimum holding period between trades
        - Regime-aware gating (don't trade in uncertain regimes)
        - Commission-aware: only trade if expected edge > commission cost
    """

    def __init__(self,
                 confidence_threshold: float = 0.55,
                 min_hold_periods: int = 4,  # minimum 4 hours between trades
                 regime_gate: bool = True):
        self.confidence_threshold = confidence_threshold
        self.min_hold_periods = min_hold_periods
        self.regime_gate = regime_gate
        self.steps_since_trade = 999  # start allowing trades
        self.current_action = 1  # start in cash

    def decide(self, forecast: ForecastResult, commission: float = 0.001) -> int:
        """
        Decide action based on forecaster output.

        Returns:
            0 = go long (BTC)
            1 = go cash (USD)
        """
        dir_prob = forecast.targets.direction_prob
        regime = forecast.regime
        volatility = forecast.volatility

        # Cooldown: don't trade too frequently
        self.steps_since_trade += 1
        if self.steps_since_trade < self.min_hold_periods:
            return self.current_action

        # Regime gating: don't trade in post_jump or cascade_risk
        if self.regime_gate and regime in ("post_jump", "cascade_risk", "illiquid"):
            return self.current_action

        # Edge check: expected profit should meaningfully exceed commission
        # Only block trades where the forecast is extremely uncertain AND
        # the expected move is smaller than commission drag
        expected_move = abs(forecast.targets.expected_return)
        if expected_move < commission * 4 and abs(dir_prob - 0.5) < 0.03:
            # Both tiny expected move AND near-50/50 direction → no edge
            return self.current_action

        # Direction decision with confidence threshold
        new_action = self.current_action

        if dir_prob > self.confidence_threshold:
            new_action = 0  # go long
        elif dir_prob < (1.0 - self.confidence_threshold):
            new_action = 1  # go cash

        # If action changed, reset cooldown
        if new_action != self.current_action:
            self.steps_since_trade = 0
            self.current_action = new_action

        return self.current_action

    def reset(self):
        self.steps_since_trade = 999
        self.current_action = 1


# ═══════════════════════════════════════════════════════════════
# GYMNASIUM-COMPATIBLE TRADING ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

class ForecastTradingEnv:
    """
    Gymnasium-compatible RL trading environment powered by the
    12-paradigm ensemble forecaster.

    Observation space: ~27 scale-invariant features from forecaster output
    Action space: Discrete(2) — 0=Long, 1=Cash (BSH style)
    Reward: PBR (Position-Based Returns) with commission awareness

    This environment can be used with:
    - The built-in ConfidenceGatedPolicy (no RL training needed)
    - Any Gymnasium-compatible RL agent (PPO, DQN, etc.)
    - Ray RLlib for distributed training
    """

    def __init__(self,
                 bars: list[Bar],
                 initial_balance: float = 10_000.0,
                 commission: float = 0.001,
                 horizon_hours: float = 24.0,
                 min_history: int = 168,
                 step_size: int = 1,  # 1 = every bar
                 forecaster: Optional[Forecaster] = None,
                 mc_iterations: int = 5_000,  # smaller for speed in RL
                 confidence_threshold: float = 0.55,
                 max_steps: int = 2000):
        """
        Args:
            bars: OHLCV bar data (hourly)
            initial_balance: Starting USD balance
            commission: Trading commission rate
            horizon_hours: Forecast horizon
            min_history: Minimum bars before trading starts
            step_size: Bars to advance per env step
            forecaster: Optional pre-built forecaster
            mc_iterations: MC iterations per forecast
            confidence_threshold: Threshold for confidence-gated policy
            max_steps: Maximum env steps per episode
        """
        self.bars = bars
        self.initial_balance = initial_balance
        self.commission = commission
        self.horizon_hours = horizon_hours
        self.min_history = min_history
        self.step_size = step_size
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold

        # Build forecaster (lighter MC for RL speed)
        self.forecaster = forecaster or Forecaster(
            mc_iterations=mc_iterations,
            mc_steps=24,
            enable_mc=True,
        )

        # Components
        self.portfolio = SimplePortfolio(
            initial_balance=initial_balance,
            commission=commission,
        )
        self.reward_fn = PBRReward(
            commission=commission,
            trade_penalty=-0.0005,
            hold_bonus=0.00005,
        )
        self.policy = ConfidenceGatedPolicy(
            confidence_threshold=confidence_threshold,
        )

        # State
        self.current_step = 0
        self.current_bar_idx = 0
        self.done = False
        self.last_forecast = None

        # Tracking
        self.episode_rewards = []
        self.episode_actions = []

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset the environment for a new episode."""
        self.current_step = 0
        self.current_bar_idx = self.min_history
        self.done = False

        initial_price = self.bars[self.current_bar_idx].close
        self.portfolio.reset(initial_price)
        self.reward_fn.reset()
        self.policy.reset()

        self.episode_rewards = []
        self.episode_actions = []

        # Generate first observation
        obs, info = self._get_observation()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.

        Args:
            action: 0=Long, 1=Cash

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            return self._get_observation()[0], 0.0, True, False, {}

        # Get current price
        current_price = self.bars[self.current_bar_idx].close

        # Execute action
        traded = self.portfolio.execute_action(
            action, current_price, self.current_step
        )

        # Compute PBR reward
        reward = self.reward_fn.compute_reward(
            current_price, action, traded,
            self.portfolio.net_worth(current_price)
        )

        # Update portfolio history
        self.portfolio.update_history(current_price)

        # Track
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)

        # Advance
        self.current_step += 1
        self.current_bar_idx += self.step_size

        # Check termination
        terminated = False
        truncated = False

        if self.current_bar_idx >= len(self.bars) - 1:
            terminated = True
            self.done = True
        elif self.current_step >= self.max_steps:
            truncated = True
            self.done = True
        elif self.portfolio.net_worth(current_price) < self.initial_balance * 0.5:
            # Blown up: lost 50%
            terminated = True
            self.done = True
            reward -= 0.1  # extra penalty for blowup

        # Get next observation
        obs, info = self._get_observation()
        info["traded"] = traded
        info["net_worth"] = self.portfolio.net_worth(current_price)
        info["trade_count"] = self.portfolio.trade_count
        info["total_commission"] = self.portfolio.total_commission_paid

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> tuple[np.ndarray, dict]:
        """Build observation from current state using forecaster."""
        if self.current_bar_idx >= len(self.bars):
            return np.zeros(27, dtype=np.float32), {}

        # Build snapshot from available history
        history = self.bars[:self.current_bar_idx + 1]
        snap = MarketSnapshot(
            symbol="BTCUSDT",
            bars_1h=history,
            timestamp=history[-1].timestamp if history else 0,
        )

        # Run forecaster (full 12-module pipeline)
        try:
            forecast = self.forecaster.forecast_from_snapshot(
                snap, self.horizon_hours
            )
            self.last_forecast = forecast
        except Exception:
            # Fallback: empty forecast
            forecast = ForecastResult()
            forecast.targets = PredictionTargets()
            forecast.regime_probs = {}
            self.last_forecast = forecast

        current_price = self.bars[self.current_bar_idx].close
        bars_window = history[-max(48, len(history)):]

        obs = build_observation(
            forecast, self.portfolio, current_price, bars_window
        )

        info = {
            "forecast": forecast,
            "regime": forecast.regime,
            "direction_prob": forecast.targets.direction_prob,
            "price": current_price,
        }

        return obs, info

    def get_portfolio_stats(self) -> dict:
        """Get comprehensive portfolio statistics."""
        nw = self.portfolio.net_worth_history
        if len(nw) < 2:
            return {"pnl": 0, "pnl_pct": 0, "trades": 0}

        pnl = nw[-1] - nw[0]
        pnl_pct = (nw[-1] / nw[0] - 1) * 100

        # Drawdown
        peak = nw[0]
        max_dd = 0
        for v in nw:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

        # Returns for Sharpe
        returns = [(nw[i] - nw[i-1]) / nw[i-1] for i in range(1, len(nw)) if nw[i-1] > 0]
        sharpe = 0.0
        if returns and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)  # annualized

        # Buy and hold comparison
        if self.bars and len(self.bars) > self.min_history:
            bh_start = self.bars[self.min_history].close
            bh_end = self.bars[min(self.current_bar_idx, len(self.bars)-1)].close
            bh_pnl_pct = (bh_end / bh_start - 1) * 100 if bh_start > 0 else 0
        else:
            bh_pnl_pct = 0

        return {
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "buy_hold_pnl_pct": round(bh_pnl_pct, 2),
            "vs_buy_hold": round(pnl_pct - bh_pnl_pct, 2),
            "trades": self.portfolio.trade_count,
            "trades_per_day": round(self.portfolio.trade_count / max(1, self.current_step / 24), 2),
            "total_commission": round(self.portfolio.total_commission_paid, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe": round(sharpe, 2),
            "final_net_worth": round(nw[-1], 2),
            "steps": self.current_step,
        }


# ═══════════════════════════════════════════════════════════════
# FORECASTER-AS-TRADER EVALUATOR
# ═══════════════════════════════════════════════════════════════

def evaluate_forecaster_as_trader(
    bars: list[Bar],
    initial_balance: float = 10_000.0,
    commission: float = 0.001,
    horizon_hours: float = 24.0,
    min_history: int = 168,
    forecast_interval: int = 4,  # forecast every 4 hours
    confidence_threshold: float = 0.55,
    min_hold_periods: int = 6,
    mc_iterations: int = 10_000,
    verbose: bool = True,
) -> dict:
    """
    Evaluate the 12-paradigm forecaster as a trading agent.

    Uses the confidence-gated policy to convert forecasts into
    trading decisions. Reports P&L, Sharpe, trade frequency,
    and comparison to buy-and-hold.

    TensorTrade lesson: Always compare to buy-and-hold. If the agent
    can't beat B&H, it hasn't learned anything useful.

    Args:
        bars: Hourly OHLCV bars
        initial_balance: Starting USD balance
        commission: Trading commission (0.001 = 0.1%)
        horizon_hours: Forecast horizon
        min_history: Bars before first forecast
        forecast_interval: Bars between forecasts (saves compute)
        confidence_threshold: Min direction_prob to trade
        min_hold_periods: Min bars between trades
        mc_iterations: MC iterations per forecast
        verbose: Print progress

    Returns:
        dict with full evaluation results
    """
    if len(bars) < min_history + 48:
        return {"error": "Not enough bars", "bars": len(bars)}

    t0 = time.time()

    # Build components
    forecaster = Forecaster(
        mc_iterations=mc_iterations,
        mc_steps=24,
        enable_mc=True,
    )
    portfolio = SimplePortfolio(
        initial_balance=initial_balance,
        commission=commission,
    )
    reward_fn = PBRReward(commission=commission)
    policy = ConfidenceGatedPolicy(
        confidence_threshold=confidence_threshold,
        min_hold_periods=min_hold_periods,
    )

    # Initialize
    initial_price = bars[min_history].close
    portfolio.reset(initial_price)
    reward_fn.reset()

    # Track
    all_rewards = []
    all_forecasts = []
    action_log = []
    last_forecast = None
    forecast_count = 0

    if verbose:
        print(f"Evaluating forecaster as trader: {len(bars)} bars, "
              f"commission={commission*100:.2f}%, "
              f"confidence_threshold={confidence_threshold}")
        print(f"Forecast interval: every {forecast_interval}h, "
              f"min hold: {min_hold_periods}h")

    # Walk through bars
    for idx in range(min_history, len(bars) - 1):
        current_price = bars[idx].close
        step = idx - min_history

        # Run forecast at intervals
        if step % forecast_interval == 0:
            history = bars[:idx + 1]
            snap = MarketSnapshot(
                symbol="BTCUSDT",
                bars_1h=history,
                timestamp=history[-1].timestamp if history else 0,
            )

            try:
                forecast = forecaster.forecast_from_snapshot(
                    snap, horizon_hours
                )
                last_forecast = forecast
                forecast_count += 1

                if verbose and forecast_count % 50 == 0:
                    print(f"  Step {step}: {forecast_count} forecasts, "
                          f"NW=${portfolio.net_worth(current_price):,.0f}, "
                          f"trades={portfolio.trade_count}")

            except Exception as e:
                if verbose:
                    print(f"  Forecast error at step {step}: {e}")
                continue

        if last_forecast is None:
            continue

        # Get action from confidence-gated policy
        action = policy.decide(last_forecast, commission)

        # Execute
        traded = portfolio.execute_action(action, current_price, step)

        # PBR reward
        reward = reward_fn.compute_reward(
            current_price, action, traded,
            portfolio.net_worth(current_price)
        )
        all_rewards.append(reward)
        portfolio.update_history(current_price)

        if traded:
            action_log.append({
                "step": step,
                "action": "BUY" if action == 0 else "SELL",
                "price": current_price,
                "direction_prob": last_forecast.targets.direction_prob,
                "regime": last_forecast.regime,
                "net_worth": portfolio.net_worth(current_price),
            })

    elapsed = time.time() - t0

    # Compute results
    nw = portfolio.net_worth_history
    final_price = bars[-2].close if len(bars) >= 2 else bars[-1].close

    # Buy-and-hold benchmark
    bh_start = bars[min_history].close
    bh_end = final_price
    bh_pnl = initial_balance * (bh_end / bh_start - 1)
    bh_pnl_pct = (bh_end / bh_start - 1) * 100

    # Agent results
    agent_pnl = nw[-1] - initial_balance if nw else 0
    agent_pnl_pct = (nw[-1] / initial_balance - 1) * 100 if nw else 0

    # Max drawdown
    peak = initial_balance
    max_dd = 0
    for v in nw:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    # Sharpe ratio
    returns = [(nw[i] - nw[i-1]) / nw[i-1] for i in range(1, len(nw)) if nw[i-1] > 0]
    sharpe = 0.0
    if returns and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)

    # Win rate on trades
    win_trades = [t for t in portfolio.trades if t.get("pnl_pct", 0) > 0]
    win_rate = len(win_trades) / max(1, len([t for t in portfolio.trades if "pnl_pct" in t]))

    # Commission impact
    gross_pnl = agent_pnl + portfolio.total_commission_paid
    commission_drag = portfolio.total_commission_paid

    results = {
        # Agent performance
        "agent_pnl": round(agent_pnl, 2),
        "agent_pnl_pct": round(agent_pnl_pct, 2),
        "final_net_worth": round(nw[-1], 2) if nw else initial_balance,

        # Buy-and-hold benchmark
        "bh_pnl": round(bh_pnl, 2),
        "bh_pnl_pct": round(bh_pnl_pct, 2),
        "vs_buy_hold": round(agent_pnl_pct - bh_pnl_pct, 2),
        "beats_buy_hold": agent_pnl_pct > bh_pnl_pct,

        # Commission analysis (TensorTrade key insight)
        "gross_pnl": round(gross_pnl, 2),
        "commission_drag": round(commission_drag, 2),
        "commission_pct_of_gross": round(
            commission_drag / abs(gross_pnl) * 100, 1
        ) if abs(gross_pnl) > 0 else 0,

        # Trading statistics
        "total_trades": portfolio.trade_count,
        "trades_per_day": round(
            portfolio.trade_count / max(1, len(all_rewards) / 24), 2
        ),
        "trade_win_rate": round(win_rate * 100, 1),

        # Risk metrics
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 2),

        # Reward analysis
        "total_pbr_reward": round(sum(all_rewards), 4),
        "avg_pbr_reward": round(sum(all_rewards) / max(1, len(all_rewards)), 6),

        # Meta
        "forecast_count": forecast_count,
        "total_bars": len(bars),
        "elapsed_sec": round(elapsed, 1),
        "commission_rate": commission,
        "confidence_threshold": confidence_threshold,

        # Trade log (last 10)
        "recent_trades": action_log[-10:],
    }

    if verbose:
        _print_results(results)

    return results


def _print_results(r: dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 65)
    print("  FORECASTER-AS-TRADER EVALUATION (TensorTrade Framework)")
    print("=" * 65)

    print(f"\n  {'Agent P&L:':<28} ${r['agent_pnl']:>+10,.2f}  ({r['agent_pnl_pct']:+.2f}%)")
    print(f"  {'Buy-and-Hold P&L:':<28} ${r['bh_pnl']:>+10,.2f}  ({r['bh_pnl_pct']:+.2f}%)")
    print(f"  {'vs Buy-and-Hold:':<28} {r['vs_buy_hold']:>+10.2f}pp")
    beat = "YES ✓" if r['beats_buy_hold'] else "NO ✗"
    print(f"  {'Beats B&H:':<28} {beat:>10}")

    print(f"\n  {'COMMISSION ANALYSIS (Key TensorTrade Insight)':}")
    print(f"  {'Gross P&L (before fees):':<28} ${r['gross_pnl']:>+10,.2f}")
    print(f"  {'Commission paid:':<28} ${r['commission_drag']:>10,.2f}")
    print(f"  {'Commission % of gross:':<28} {r['commission_pct_of_gross']:>10.1f}%")

    print(f"\n  {'TRADING STATISTICS':}")
    print(f"  {'Total trades:':<28} {r['total_trades']:>10}")
    print(f"  {'Trades per day:':<28} {r['trades_per_day']:>10.1f}")
    tpd = r['trades_per_day']
    freq = "GOOD (< 5)" if tpd < 5 else "HIGH (> 5)" if tpd < 20 else "EXCESSIVE (> 20)"
    print(f"  {'Trading frequency:':<28} {freq:>10}")
    print(f"  {'Trade win rate:':<28} {r['trade_win_rate']:>10.1f}%")

    print(f"\n  {'RISK METRICS':}")
    print(f"  {'Max drawdown:':<28} {r['max_drawdown_pct']:>10.2f}%")
    print(f"  {'Sharpe ratio:':<28} {r['sharpe_ratio']:>10.2f}")

    print(f"\n  {'META':}")
    print(f"  {'Forecasts run:':<28} {r['forecast_count']:>10}")
    print(f"  {'Time elapsed:':<28} {r['elapsed_sec']:>10.1f}s")
    print(f"  {'Commission rate:':<28} {r['commission_rate']*100:>10.2f}%")
    print(f"  {'Confidence threshold:':<28} {r['confidence_threshold']:>10.2f}")

    print("\n" + "=" * 65)


# ═══════════════════════════════════════════════════════════════
# MULTI-COMMISSION SWEEP (TensorTrade Experiment 7 Replication)
# ═══════════════════════════════════════════════════════════════

def commission_sweep(
    bars: list[Bar],
    commissions: list[float] = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005],
    **kwargs,
) -> list[dict]:
    """
    Replicate TensorTrade's Experiment 7: test across commission levels.

    Key finding from TT: Agent was profitable at 0% commission (+$239)
    but lost money at 0.1% (-$650) due to overtrading.

    This sweep reveals:
    - Direction prediction quality (0% commission result)
    - Commission sensitivity (how quickly profit erodes)
    - Optimal trading frequency for each commission level
    """
    results = []
    show_detail = kwargs.pop("verbose", True)
    for comm in commissions:
        print(f"\n{'='*50}")
        print(f"  Commission: {comm*100:.3f}%")
        print(f"{'='*50}")
        r = evaluate_forecaster_as_trader(
            bars, commission=comm, verbose=show_detail, **kwargs
        )
        r["commission_level"] = comm
        results.append(r)

    # Summary table
    print("\n\n" + "=" * 80)
    print("  COMMISSION SWEEP SUMMARY")
    print("=" * 80)
    print(f"  {'Commission':<12} {'Agent P&L':>12} {'B&H P&L':>12} "
          f"{'vs B&H':>10} {'Trades':>8} {'Trades/Day':>12}")
    print("-" * 80)
    for r in results:
        print(f"  {r['commission_level']*100:>8.3f}%   "
              f"${r['agent_pnl']:>+10,.2f}  "
              f"${r['bh_pnl']:>+10,.2f}  "
              f"{r['vs_buy_hold']:>+8.2f}pp  "
              f"{r['total_trades']:>6}  "
              f"{r['trades_per_day']:>10.1f}")
    print("=" * 80)

    return results


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    """CLI: Run forecaster-as-trader evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate ClawdBot forecaster as a trading agent "
                    "(TensorTrade framework)"
    )
    parser.add_argument("-s", "--symbol", default="BTCUSDT")
    parser.add_argument("--bars", type=int, default=2000,
                        help="Number of hourly bars to fetch")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Trading commission rate (0.001 = 0.1%%)")
    parser.add_argument("--confidence", type=float, default=0.55,
                        help="Confidence threshold for trading")
    parser.add_argument("--min-hold", type=int, default=6,
                        help="Minimum holding period in hours")
    parser.add_argument("--forecast-interval", type=int, default=4,
                        help="Hours between forecasts")
    parser.add_argument("--mc-iterations", type=int, default=10000,
                        help="Monte Carlo iterations per forecast")
    parser.add_argument("--sweep", action="store_true",
                        help="Run commission sweep (TensorTrade Exp 7)")
    parser.add_argument("--balance", type=float, default=10000,
                        help="Initial balance in USD")

    args = parser.parse_args()

    # Fetch data
    from .data import fetch_ohlcv_bars

    print(f"Fetching {args.bars} bars of {args.symbol}...")
    bars, source = fetch_ohlcv_bars(args.symbol, limit=args.bars)
    print(f"Got {len(bars)} bars from {source}")

    if args.sweep:
        commission_sweep(
            bars,
            initial_balance=args.balance,
            confidence_threshold=args.confidence,
            min_hold_periods=args.min_hold,
            forecast_interval=args.forecast_interval,
            mc_iterations=args.mc_iterations,
        )
    else:
        evaluate_forecaster_as_trader(
            bars,
            initial_balance=args.balance,
            commission=args.commission,
            confidence_threshold=args.confidence,
            min_hold_periods=args.min_hold,
            forecast_interval=args.forecast_interval,
            mc_iterations=args.mc_iterations,
        )


if __name__ == "__main__":
    main()
