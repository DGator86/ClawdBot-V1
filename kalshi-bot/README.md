# Kalshi Crypto Trading Bot

A Kalshi hourly crypto trading bot with technical analysis, controlled via Telegram through moltbot.

## Features

- **Technical Analysis**: EMA, RSI, MACD, Bollinger Bands, ATR, Support/Resistance
- **Supported Cryptos**: BTC, ETH, SOL (hourly markets only)
- **Risk Management**: 20% max drawdown kill switch, $20 max trade
- **Manual Confirmation**: All trades require explicit approval
- **Telegram Control**: Full control via Captain Falcon bot

## Safety Limits (Hardcoded)

| Limit | Value |
|-------|-------|
| Max Trade Cost | $20 |
| Max Drawdown | 20% |
| Require Confirmation | Always |
| Allowed Markets | BTC, ETH, SOL hourly |

## Setup

### 1. Configure Credentials

```bash
cd ~/kalshi-bot
nano .env
```

Add your Kalshi API credentials:
```
KALSHI_API_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
...
-----END RSA PRIVATE KEY-----"
```

Lock the file:
```bash
chmod 600 .env
```

### 2. Deploy

```bash
chmod +x deploy.sh
./deploy.sh
```

### 3. Verify

```bash
systemctl status kalshi-bot
curl http://127.0.0.1:3456/health
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/kalshi status` | Portfolio & risk metrics |
| `/kalshi markets` | Available hourly crypto markets |
| `/kalshi analyze ETH` | Technical analysis |
| `/kalshi propose` | Generate trade proposals |
| `/kalshi list` | View active proposals |
| `/kalshi confirm <id>` | Execute a proposal |
| `/kalshi reject <id>` | Discard a proposal |
| `/kalshi kill` | Emergency stop all trading |
| `/kalshi reset RESET-RISK-CONFIRM` | Reset kill switch |
| `/kalshi history` | Recent trades |

**Shortcuts**: `/k s`, `/k m`, `/k a eth`, `/k p`, `/k c abc123`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Account status & risk metrics |
| `/markets` | GET | Available crypto markets |
| `/analyze/:symbol` | GET | Technical analysis |
| `/propose` | POST | Generate proposals |
| `/proposals` | GET | List active proposals |
| `/confirm/:id` | POST | Execute a proposal |
| `/reject/:id` | DELETE | Reject a proposal |
| `/kill` | POST | Trigger kill switch |
| `/reset-kill` | POST | Reset kill switch |
| `/history` | GET | Trade history |

## Technical Analysis

The bot analyzes:

- **Trend**: EMA 9/21/50 alignment, MACD crossovers
- **Velocity**: Rate of change, momentum direction
- **Volatility**: ATR, Bollinger Band width
- **Pinning**: Support/resistance proximity
- **RSI**: Overbought/oversold conditions

## Risk Management

1. **Max Trade Cost**: $20 per trade (hardcoded)
2. **Drawdown Kill Switch**: Triggers at 20% drawdown from high water mark
3. **Manual Confirmation**: Every trade requires `/kalshi confirm <id>`
4. **Market Restrictions**: Only BTC, ETH, SOL hourly markets

## Logs

```bash
# Live logs
journalctl -u kalshi-bot -f

# Last 100 lines
journalctl -u kalshi-bot -n 100
```

## Troubleshooting

### Bot not responding

```bash
systemctl status kalshi-bot
journalctl -u kalshi-bot -n 50
```

### Kill switch active

```bash
curl http://127.0.0.1:3456/status
# Reset via Telegram: /kalshi reset RESET-RISK-CONFIRM
```

### API connection issues

```bash
curl http://127.0.0.1:3456/health
```

## License

MIT
