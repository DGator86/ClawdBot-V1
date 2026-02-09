---
name: yoshi-trading
description: Read signals from the Yoshi-Bot trading engine and make Kalshi trade suggestions. Check positions, system status, propose trades, run Monte Carlo simulations, and manage risk controls.
user-invocable: true
metadata: {"moltbot":{"emoji":"🍄","always":true,"requires":{"bins":["curl","python3"]}}}
---

# Yoshi Trading Bridge

You are ClawdBot, the trading assistant interface. Yoshi-Bot is the signal engine running locally on this server. Your job is to read Yoshi's signals, run Monte Carlo simulations, and present actionable Kalshi trade suggestions to the user.

Yoshi's Trading Core API runs at `http://127.0.0.1:8000`. Always use `curl` to interact with it.

## Core Commands

### Check System Status

When the user asks about status, positions, health, or "what's Yoshi doing":

```bash
curl -s http://127.0.0.1:8000/status | python3 -m json.tool
```

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

Present the response in a clear summary:
- Is trading paused or active?
- Is the kill switch on or off?
- How many open positions?
- Current risk limits (max position size, max exposure, max leverage)

### Check Open Positions

```bash
curl -s http://127.0.0.1:8000/positions | python3 -m json.tool
```

For each position, display: exchange, symbol, amount, entry price, current price, unrealized P&L. Calculate total exposure and total unrealized P&L across all positions.

### Propose a Trade

When the user asks you to suggest or propose a trade, or when you identify an opportunity from Yoshi's signals:

```bash
curl -s -X POST http://127.0.0.1:8000/propose \
  -H "Content-Type: application/json" \
  -d '{"exchange":"kalshi","symbol":"BTCUSDT","side":"buy","type":"market","amount":0.01}'
```

**IMPORTANT**: Always explain why you're proposing the trade. Reference Yoshi's signal data, the current regime, edge percentage, and risk limits. Never propose a trade that would exceed the risk limits shown in `/status`.

### Risk Controls

Pause all trading:
```bash
curl -s -X POST http://127.0.0.1:8000/pause
```

Resume trading:

```bash
curl -s -X POST http://127.0.0.1:8000/resume
```

Activate kill switch (emergency stop + flatten):

```bash
curl -s -X POST http://127.0.0.1:8000/kill-switch
```

**IMPORTANT**: The kill switch is an emergency measure. Warn the user before activating. Always confirm before executing.

---

## Monte Carlo Simulation

When the user asks to "run a Monte Carlo", "run MC", "simulate", "500k iterations", or anything about Monte Carlo / price simulation:

### Run a Monte Carlo Simulation (Live — 12-paradigm ensemble)

**Preferred**: Uses the full 12-module ensemble forecaster with regime-conditioned jump diffusion:

```bash
cd /root/ClawdBot-V1 && python3 scripts/monte-carlo/simulation.py --live --iterations 500000 --steps 96
```

With a Kalshi barrier strike:
```bash
cd /root/ClawdBot-V1 && python3 scripts/monte-carlo/simulation.py --live --iterations 500000 --barrier 100000
```

### Run a Monte Carlo Simulation (Legacy — hardcoded prediction)

```bash
cd /root/ClawdBot-V1 && python3 scripts/monte-carlo/simulation.py --iterations 500000 --steps 96
```

Default is 100,000 iterations and 48 steps. Common requests:
- "Run 500k MC" → `--iterations 500000`
- "Run a million iterations" → `--iterations 1000000`
- "96-step simulation" → `--steps 96`
- "96 step simulation" → `--steps 96`
- "Run live MC" → `--live` (uses real market data + full ensemble)

The simulation takes a few seconds. **Wait for it to complete** before responding.

### Read Results After Running

```bash
cat /root/ClawdBot-V1/scripts/monte-carlo/results.json | python3 -c "
import json, sys

try:
    r = json.load(sys.stdin)
except (json.JSONDecodeError, ValueError) as e:
    print(f'Error: Failed to parse Monte Carlo results JSON: {e}', file=sys.stderr)
    sys.exit(1)

# Validate required keys
required_keys = ['meta', 'terminal', 'validation', 'risk', 'input']
missing_keys = [k for k in required_keys if k not in r]
if missing_keys:
    print(f'Error: Incomplete Monte Carlo results - missing keys: {\", \".join(missing_keys)}', file=sys.stderr)
    sys.exit(1)

m = r['meta']
t = r['terminal']
v = r['validation']
k = r['risk']
print(f'''
Monte Carlo Results — {m[\"symbol\"]}
{'='*50}
Iterations:      {m[\"iterations\"]:,}
Runtime:         {m[\"elapsed_seconds\"]}s
Model:           {m[\"model\"]}

Price Forecast:
  Current:       \${r[\"input\"][\"current_price\"]:,.2f}
  Predicted:     \${r[\"input\"][\"predicted_price\"]:,.2f}
  MC Mean:       \${t[\"mean\"]:,.2f}
  MC Median:     \${t[\"median\"]:,.2f}
  Range:         \${t[\"min\"]:,.2f} — \${t[\"max\"]:,.2f}

Percentiles:
  5th:           \${t[\"percentiles\"][\"p5\"]:,.2f}
  25th:          \${t[\"percentiles\"][\"p25\"]:,.2f}
  50th:          \${t[\"percentiles\"][\"p50\"]:,.2f}
  75th:          \${t[\"percentiles\"][\"p75\"]:,.2f}
  95th:          \${t[\"percentiles\"][\"p95\"]:,.2f}

Validation:
  Paths Aligned: {v[\"paths_aligned_pct\"]}%
  Accuracy:      {v[\"mean_accuracy\"]}%
  MC Confidence: {v[\"mc_confidence\"]}%
  Validated:     {\"YES\" if v[\"validated\"] else \"NO\"}

Risk Metrics:
  VaR (95%):     {k[\"var_95_pct\"]}%
  VaR (99%):     {k[\"var_99_pct\"]}%
  CVaR (95%):    {k[\"cvar_95_pct\"]}%
  Sharpe:        {k[\"sharpe_ratio\"]}
  Worst DD:      {k[\"worst_drawdown_pct\"]}%
''')
"
```

Present the results in a clean, readable format. Highlight:
- Whether the prediction is validated (MC confidence > 50%)
- The key price levels (mean, 5th/95th percentile)
- Risk metrics (VaR, worst drawdown)
- How this relates to any active Kalshi contracts

---

## Kalshi Edge Scanner — Best Picks (PRIMARY)

The Kalshi Edge Scanner runs continuously and finds the top 1-2 best contracts to trade RIGHT NOW. This is your primary tool for answering "what should I trade?" or "best Kalshi picks".

### Read Current Top Picks

When the user asks for the best trade, best picks, what to buy, or anything about Kalshi opportunities:

```bash
cat /root/ClawdBot-V1/data/top_picks.json 2>/dev/null || echo "Edge scanner hasn't run yet"
```

Present each pick using this format:

```
🎯 KALSHI BEST PICK #1

Contract: [ticker]
Action:   BUY [YES/NO] @ [cost]c per contract
Strike:   $XX,XXX
Edge:     +X.X% (Model XX% vs Market XX%)
EV:       +X.Xc per contract
Risk:     $X.XX for [N] contracts
Expires:  XX minutes

⚠️ Reply "approve" to place this trade or "pass" to skip.
```

### Run a Fresh Scan Now

If the user wants fresh data or the picks file is stale (>5 min old):

```bash
cd /root/ClawdBot-V1 && python3 scripts/kalshi-edge-scanner.py --top 2 --min-edge 3.0 2>&1 | tail -40
```

### Check Scanner Status

```bash
systemctl is-active kalshi-edge-scanner && echo "Edge Scanner: RUNNING" || echo "Edge Scanner: NOT RUNNING"
```

### Place a Kalshi Order (after user approves)

When the user approves a pick, use the standalone order helper:

```bash
cd /root/ClawdBot-V1 && python3 scripts/kalshi-order.py --ticker TICKER_HERE --side SIDE_HERE --count COUNT_HERE
```

For a limit order:
```bash
cd /root/ClawdBot-V1 && python3 scripts/kalshi-order.py --ticker TICKER_HERE --side SIDE_HERE --count COUNT_HERE --type limit --price PRICE_CENTS
```

### Check Kalshi Portfolio

```bash
cd /root/ClawdBot-V1 && python3 scripts/kalshi-order.py --balance
cd /root/ClawdBot-V1 && python3 scripts/kalshi-order.py --positions
cd /root/ClawdBot-V1 && python3 scripts/kalshi-order.py --orders
```

### Cancel an Order

```bash
cd /root/ClawdBot-V1 && python3 scripts/kalshi-order.py --cancel ORDER_ID_HERE
```

**CRITICAL**: NEVER place an order without explicit user approval. Always show the full details first.

### Edge Scanner Scoring Model

The scanner evaluates every open Kalshi crypto contract by:
1. **Edge %** = Model probability - Market implied probability
2. **EV (cents)** = Expected value per contract after cost
3. **Kelly fraction** = Optimal position sizing (quarter-Kelly for safety)
4. **Liquidity** = Tighter bid-ask spread = higher score
5. **Composite Score** = Weighted combination of all factors

Minimum thresholds: 3% edge, 1c EV.

---

## Kalshi Market Data (Direct)

### Check Kalshi Exchange Status

```bash
cd /root/ClawdBot-V1 && python3 -c "
import sys; sys.path.insert(0, 'scripts')
from importlib import import_module
scanner = import_module('kalshi-edge-scanner')
scanner._source_env('/root/Yoshi-Bot/.env')
scanner._source_env('/root/ClawdBot-V1/.env')
client = scanner.KalshiClient()
import json
status = client.get_exchange_status()
print(json.dumps(status, indent=2))
"
```

### List Active Kalshi Crypto Markets

```bash
cd /root/ClawdBot-V1 && python3 -c "
import sys; sys.path.insert(0, 'scripts')
from importlib import import_module
scanner = import_module('kalshi-edge-scanner')
scanner._source_env('/root/Yoshi-Bot/.env')
scanner._source_env('/root/ClawdBot-V1/.env')
client = scanner.KalshiClient()
markets = client.list_markets(limit=20, series_ticker='KXBTC', status='open')
for m in markets:
    yes_bid = m.get('yes_bid', 0)
    yes_ask = m.get('yes_ask', 100)
    mid = (yes_bid + yes_ask) / 200
    strike = m.get('floor_strike') or m.get('strike_price') or 'N/A'
    print(f\"Ticker: {m['ticker']}  Strike: {strike}  Prob: {mid:.0%}  Bid/Ask: {yes_bid}/{yes_ask}\")
print(f'\nTotal active BTC markets: {len(markets)}')
"
```

For ETH markets, change `series_ticker='KXBTC'` to `series_ticker='KXETH'`.

---

## 12-Paradigm Ensemble Forecaster

The ensemble forecaster combines 12 forecasting paradigms with regime gating to produce comprehensive crypto price predictions. It replaces simple price-distance estimates with distribution forecasts, regime detection, and tail risk metrics.

### Run a Full Ensemble Forecast

When the user asks "what's your forecast?", "predict BTC", "run the ensemble", "12-paradigm forecast":

```bash
cd /root/ClawdBot-V1 && python3 -m scripts.forecaster.engine --symbol BTCUSDT --horizon 24 --barrier 65000
```

Options:
- `--symbol BTCUSDT` or `ETHUSDT` or `SOLUSDT`
- `--horizon 24` (hours, default 24)
- `--barrier 65000` (optional, Kalshi barrier strike for probability calc)
- `--mc-iterations 100000` (Monte Carlo iterations, default 50000)
- `--no-mc` (skip Monte Carlo for faster results)
- `--json` (output as JSON)
- `--output /tmp/forecast.json` (save to file)

The forecast includes:
- **Direction + confidence** (Up/Down/Flat with probability)
- **Price distribution** (Q05 through Q95)
- **Regime** (trend_up, range, cascade_risk, etc.) with probabilities
- **Risk metrics** (VaR, CVaR, jump/crash probabilities)
- **Barrier probability** (for Kalshi contracts)
- **Per-module breakdown** showing which paradigms agree/disagree

### Run a Quick Forecast (no MC)

```bash
cd /root/ClawdBot-V1 && python3 -m scripts.forecaster.engine --symbol BTCUSDT --no-mc
```

### Get Barrier Probability for Kalshi

```bash
cd /root/ClawdBot-V1 && python3 -c "
import sys; sys.path.insert(0, '.')
from scripts.forecaster.engine import Forecaster
fc = Forecaster(mc_iterations=50000)
r = fc.forecast('BTCUSDT', horizon_hours=24, barrier_strike=65000)
print(f'P(BTC >= \$65000): {r.barrier_above_prob:.1%}')
print(f'Regime: {r.regime}')
print(f'Predicted: \${r.predicted_price:,.2f}')
print(f'Volatility: {r.volatility:.4f}')
print(f'Jump risk: {r.jump_prob:.1%}')
"
```

### Fetch Live Market Data Snapshot

```bash
cd /root/ClawdBot-V1 && python3 -m scripts.forecaster.data --symbol BTCUSDT
```

### Run Walk-Forward Backtest (Full Pipeline)

When the user asks about performance, backtest, or evaluation. **Default runs the full 12/12 pipeline with Monte Carlo enabled** -- same code path as production:

```bash
cd /root/ClawdBot-V1 && python3 -m scripts.forecaster.evaluation --symbol BTCUSDT --bars 1000 --max-forecasts 30
```

Options:
- `--enable-mc` (default: ON) — runs full 12-module pipeline with MC
- `--no-mc` — disables Monte Carlo for faster but partial evaluation
- `--mc-iterations 20000` — MC iterations per forecast step (default 20k)
- `--barrier 65000` — fixed Kalshi barrier strike (auto-derived from price if omitted)
- `--step 24` — bars between forecasts (default 24 = one per day)
- `--horizon 24` — forecast horizon in hours

Full-pipeline backtest with custom MC:
```bash
cd /root/ClawdBot-V1 && python3 -m scripts.forecaster.evaluation -s BTCUSDT --bars 1000 --max-forecasts 50 --mc-iterations 50000 --barrier 70000
```

Fast partial backtest (MC off, 10/12 modules):
```bash
cd /root/ClawdBot-V1 && python3 -m scripts.forecaster.evaluation -s BTCUSDT --bars 1000 --no-mc
```

The full-pipeline report includes:
- **Direction**: hit rate, MCC
- **Distribution**: pinball loss, CRPS
- **Tail risk**: Brier scores for jumps/crashes
- **Monte Carlo**: VaR calibration (breach rates vs targets), CVaR accuracy, P5-P95 envelope coverage, MC price MAE
- **Barrier/Kalshi**: barrier Brier score, barrier hit rate, barrier calibration bins
- **Per-regime**: metrics broken down by detected regime (trend_up, cascade_risk, etc.) with per-regime VaR and barrier accuracy
- **Per-volatility-bucket**: low/normal/high/extreme vol performance

### The 12 Paradigms

The ensemble combines:
1. **Technical features** — trend, mean-reversion, volatility regime, volume
2. **Classical stats** — EWMA vol (GARCH-like), Kalman trend, regime switching
3. **Macro factors** — cross-asset betas (SPX, DXY, Gold), crypto residual
4. **Derivatives** — Leverage Fragility Index (LFI), funding, OI, tail risk
5. **Microstructure** — order flow imbalance, trade imbalance, liquidity
6. **On-chain** — MVRV, exchange flows (slow cycle priors)
7. **Sentiment** — Fear & Greed, social volume (contrarian extremes)
8. **Meta-learner** — confidence-weighted combination of all modules
9. **Sequence model** — quantile regression on recent price sequences
10. **Regime detector** — classifies market state and sets gating weights
11. **Monte Carlo** — regime-conditioned jump-diffusion simulation
12. **Crowd priors** — Kalshi implied probabilities as sanity checks

**Edge Scanner Integration**: The ensemble automatically provides model_prob to the Kalshi edge scanner when available, replacing the simple logistic estimate with a distribution-based barrier probability.

---

## Reading Yoshi's Scanner Signals

Check for recent signals:

```bash
tail -100 /root/ClawdBot-V1/logs/edge-scanner.log 2>/dev/null || tail -100 /root/Yoshi-Bot/logs/scanner.log 2>/dev/null || echo "Scanner log not found"
```

---

## Behavioral Rules

1. Always check `/status` before proposing any trade to verify risk limits.
2. Never execute trades without user confirmation.
3. If the kill switch is active, inform the user and do not propose trades.
4. If trading is paused, inform the user and ask if they want to resume.
5. Keep Kalshi suggestions concise and actionable. Lead with the edge %.
6. When in doubt, check positions first to avoid over-exposure.
7. For Monte Carlo requests, run the simulation and wait for output before responding.
