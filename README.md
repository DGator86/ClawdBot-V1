# ClawdBot

A Telegram AI trading assistant built on [moltbot](https://github.com/moltbot/moltbot). ClawdBot reads signals from [Yoshi-Bot](https://github.com/DGator86/Yoshi-Bot) and presents actionable Kalshi trade suggestions via Telegram.

## Architecture

```
Yoshi-Bot (Signal Engine)              ClawdBot (Telegram Interface)
========================              ============================

kalshi_scanner.py (loop)              moltbot gateway :18789
  | Fetches OHLCV data                  | yoshi-trading skill
  | Runs PriceTimeManifold              | Reads Trading Core API
  | Finds Kalshi edge opportunities     | Formats trade suggestions
  | Writes to scanner.log               | Sends to Telegram
  v                                     v
yoshi-bridge.py                       You (Telegram)
  | Watches scanner.log                 | "What's Yoshi's status?"
  | Parses signals (edge, strike, etc)  | "Show me positions"
  | POSTs to Trading Core /propose      | "approve" / "pass"
  v                                     |
Trading Core API :8000  <---------------+
  /status    /positions
  /propose   /approve/{id}
  /orders    /kill-switch
  /pause     /resume
  /flatten
```

## Prerequisites

- DigitalOcean Droplet or any Linux server (Ubuntu 24.04 recommended)
- Node.js 22+
- Python 3.10+
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- OpenAI API Key (from [platform.openai.com](https://platform.openai.com/))
- Kalshi API credentials (from [kalshi.com/account/api](https://kalshi.com/account/api))
- Yoshi-Bot deployed with Trading Core running on port 8000

## Quick Start

### 1. Connect to Your Droplet

```bash
ssh root@165.245.140.115
```

### 2. Clone the Repository

```bash
git clone https://github.com/DGator86/ClawdBot-V1.git
cd ClawdBot-V1
```

### 3. Configure Environment

```bash
cp .env.example .env
nano .env
# Set TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, KALSHI_KEY_ID, KALSHI_PRIVATE_KEY
```

### 4. Deploy the Bridge

```bash
chmod +x scripts/deploy-bridge.sh
./scripts/deploy-bridge.sh
```

This installs and starts:
- **clawdbot** service (moltbot gateway + Telegram + yoshi-trading skill)
- **yoshi-bridge** service (scanner log watcher -> Trading Core /propose)

The deploy script will also prompt for Kalshi API credentials if not already configured.

### 5. Talk to Your Bot

Open Telegram and message your bot:
- "What's Yoshi's status?"
- "Show me open positions"
- "Any Kalshi signals?"
- "Check Kalshi exchange status"
- "Pause trading"
- "Activate kill switch"

## Services

| Service | Port | Description |
|---------|------|-------------|
| Yoshi Trading Core | 8000 | FastAPI order management, positions, risk controls |
| ClawdBot Gateway | 18789 | Moltbot AI gateway + Telegram channel |
| Yoshi Bridge | — | Log watcher, forwards scanner signals to Trading Core |
| Kalshi Scanner | — | Background signal engine (part of Yoshi-Bot) |
| Kalshi API | — | Prediction market data (RSA-PSS V2 auth) |

## Configuration

### Environment Variables (.env)

| Variable | Description | Required |
|----------|-------------|----------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token | Yes |
| `OPENAI_API_KEY` | OpenAI API key (for ClawdBot reasoning) | Yes |
| `KALSHI_KEY_ID` | Kalshi API Key ID | Yes |
| `KALSHI_PRIVATE_KEY` | Kalshi RSA private key (PEM format) | Yes |
| `ANTHROPIC_API_KEY` | Anthropic API key (alternative provider) | No |
| `DIGITALOCEAN_TOKEN` | DO API token (for VPS health monitoring) | No |
| `TELEGRAM_CHAT_ID` | Telegram chat ID (for alert notifications) | No |

### Moltbot Configuration (~/.clawdbot/moltbot.json)

```json
{
  "agent": {
    "model": "openai/gpt-4o",
    "thinking": "medium",
    "systemPrompt": "You are ClawdBot, a crypto trading assistant..."
  },
  "gateway": {
    "port": 18789,
    "bind": "127.0.0.1"
  },
  "channels": {
    "telegram": {
      "enabled": true,
      "botToken": "YOUR_TELEGRAM_BOT_TOKEN"
    }
  },
  "skills": {
    "load": {
      "extraDirs": ["./skills"]
    }
  }
}
```

### Yoshi-Trading Skill

The `skills/yoshi-trading/SKILL.md` teaches ClawdBot how to:
- Query Yoshi's Trading Core API (status, positions, health)
- Query Kalshi markets directly (exchange status, active markets, series)
- Parse Kalshi scanner signals from logs
- Propose and approve trades through the Trading Core
- Manage risk controls (pause, resume, flatten, kill switch)
- Format Kalshi suggestions with edge %, strike, and action

## Kalshi Integration

ClawdBot connects to Kalshi prediction markets through Yoshi-Bot's signal engine. The integration uses the Kalshi V2 API with RSA-PSS authentication.

### How It Works

1. **Yoshi-Bot** (`kalshi_scanner.py`) continuously scans Kalshi crypto markets (KXBTC, KXETH) for mispriced opportunities using the Price-Time Manifold model with Monte Carlo simulations (2000 sims per timeframe).
2. **Yoshi-Bridge** (`yoshi-bridge.py`) watches the scanner log and forwards qualifying signals (edge >= 5%) to the Trading Core `/propose` endpoint.
3. **ClawdBot** reads the Trading Core API and presents actionable Kalshi suggestions via Telegram, complete with edge %, strike, probability, and recommended action.
4. **You** review and approve/reject trades via Telegram.

### Kalshi API Setup

1. Get your API credentials from [Kalshi Account API](https://kalshi.com/account/api)
2. Add credentials to Yoshi-Bot's `.env`:
   ```bash
   KALSHI_KEY_ID=your_key_id_here
   KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
   ... your key ...
   -----END RSA PRIVATE KEY-----"
   ```
3. Or use a PEM file: store at `~/.kalshi/private_key.pem` (chmod 600)
4. The deploy script (`scripts/deploy-bridge.sh`) will prompt for these during setup

### Supported Kalshi Markets

| Series | Symbol | Description |
|--------|--------|-------------|
| KXBTC | BTCUSDT | Bitcoin hourly price contracts |
| KXETH | ETHUSDT | Ethereum hourly price contracts |

### Kalshi Commands via Telegram

Ask ClawdBot in natural language:
- "Check Kalshi exchange status"
- "Show me active BTC Kalshi markets"
- "What Kalshi signals does Yoshi have?"
- "Is Kalshi API connected?"
- "Show Kalshi market for KXBTC-25FEB06-T100000"

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | View session details and token usage |
| `/new` or `/reset` | Clear session context |
| `/think <level>` | Set reasoning level (low/medium/high) |
| `/verbose on\|off` | Toggle verbose mode |
| `/restart` | Restart the gateway |

Plus natural language queries like:
- "What's Yoshi doing?"
- "Show positions"
- "Propose a BTC trade"
- "Kill switch NOW"

## Managing Services

All services are managed via systemd. Install with:

```bash
bash scripts/services/install.sh --start
```

```bash
# Check all service status
bash scripts/services/install.sh --status

# View logs
sudo journalctl -u clawdbot -f
sudo journalctl -u kalshi-edge-scanner -f
sudo journalctl -u yoshi-bridge -f

# Restart all
bash scripts/services/install.sh --restart

# Restart individual service
sudo systemctl restart clawdbot

# Check Yoshi Trading Core directly
curl -s http://127.0.0.1:8000/status | python3 -m json.tool
```

### Operations Utilities

```bash
# Fix Kalshi PEM keys across all locations
python3 scripts/lib/pem_utils.py

# Unify Telegram tokens (dry-run)
python3 scripts/lib/env_sync.py

# Unify Telegram tokens (apply)
python3 scripts/lib/env_sync.py --apply

# Rebuild moltbot.json with model auto-detection
python3 scripts/rebuild-config.py

# Nuclear gateway fix (kills everything, rebuilds, restarts)
bash scripts/fix-gateway.sh
```

### Troubleshooting

**Bot not responding:**
1. Check if the service is running: `bash scripts/services/install.sh --status`
2. Check logs for errors: `sudo journalctl -u clawdbot -n 100`
3. Verify your tokens: `python3 scripts/lib/env_sync.py`
4. Nuclear fix: `bash scripts/fix-gateway.sh`

**Connection issues:**
1. Ensure port 18789 is open (for local gateway)
2. Check DigitalOcean firewall allows outbound HTTPS (443)
3. Restart: `sudo systemctl restart clawdbot`

**Kalshi API errors:**
1. Fix PEM keys: `python3 scripts/lib/pem_utils.py`
2. Verify credentials: ask ClawdBot "Is Kalshi API connected?"
3. Check exchange status: may be outside trading hours

## Project Structure

```
ClawdBot-V1/
├── .env.example                    # Environment template
├── package.json                    # Node.js project (moltbot dep)
├── config/
│   └── moltbot.example.json        # Moltbot gateway config (with model fallbacks)
├── skills/
│   └── yoshi-trading/
│       └── SKILL.md                # Yoshi-Trading bridge skill
├── scripts/
│   ├── deploy-bridge.sh            # Full bridge deployment (8 steps)
│   ├── fix-gateway.sh              # Nuclear gateway fix (kills stale, rebuilds config)
│   ├── setup-all.sh                # All-in-one VPS setup (curl-friendly)
│   ├── rebuild-config.py           # Rebuild moltbot.json with model selection
│   ├── yoshi-bridge.py             # Scanner log -> Trading Core bridge
│   ├── kalshi-edge-scanner.py      # Continuous Kalshi best-pick finder
│   ├── kalshi-order.py             # Kalshi order placement helper
│   ├── lib/
│   │   ├── pem_utils.py            # Shared PEM key normalization
│   │   └── env_sync.py             # Telegram token unification + env sync
│   ├── services/
│   │   ├── install.sh              # Systemd service installer
│   │   ├── clawdbot.service        # Gateway + Telegram service
│   │   ├── kalshi-edge-scanner.service  # Edge scanner service
│   │   └── yoshi-bridge.service    # Bridge service
│   ├── forecaster/                 # 12-paradigm ensemble forecaster
│   │   ├── engine.py               # Ensemble orchestrator
│   │   ├── modules.py              # All 12 prediction modules
│   │   ├── evaluation.py           # Walk-forward backtester
│   │   ├── data.py                 # OHLCV data fetcher (Coinbase/Kraken)
│   │   ├── schemas.py              # Module interfaces & data types
│   │   └── bridge.py               # Forecaster -> edge scanner bridge
│   └── monte-carlo/
│       ├── simulation.py           # MC engine (legacy + live forecaster modes)
│       ├── index.html              # Web dashboard
│       └── server.py               # Dashboard HTTP server
└── moltbot-setup/
    ├── SETUP-GUIDE.md              # DigitalOcean setup guide
    └── install-moltbot.sh          # DO installer
```

## DigitalOcean Droplet Details

- **Name**: Clawd-Server
- **Public IP**: 165.245.140.115
- **Private IP**: 10.128.0.2
- **Region**: ATL1 (Atlanta)
- **Specs**: 8 GB Memory / 2 Intel vCPUs / 160 GB Disk
- **OS**: Ubuntu 24.04 (LTS) x64
- **VPC Network**: default-atl1 (10.128.0.0/20)

## Security Notes

- Never commit `.env` or files containing API keys or private keys
- `.env`, `.pem`, and `.kalshi/` are all gitignored
- The Trading Core API runs on localhost only (127.0.0.1:8000)
- The moltbot gateway runs on localhost only (127.0.0.1:18789)
- Kalshi private keys should be stored with `chmod 600` permissions
- Use DigitalOcean Cloud Firewalls to restrict access
- Consider enabling DigitalOcean Monitoring for observability
- Trade approvals require explicit user confirmation via Telegram
- Rotate Kalshi API keys immediately if exposed

## License

MIT
