# ClawdBot

A Telegram AI assistant powered by Claude, built on [moltbot](https://github.com/moltbot/moltbot).

## Prerequisites

- DigitalOcean Droplet (Ubuntu 24.04 recommended)
- Node.js 22+
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- Anthropic API Key (from [console.anthropic.com](https://console.anthropic.com/))

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

### 3. Run Deployment Script

```bash
chmod +x scripts/*.sh
./scripts/deploy.sh
```

### 4. Configure Your Bot

```bash
./scripts/setup-telegram.sh
```

Or manually edit the files:

```bash
# Set your API keys
nano .env

# Configure moltbot
nano ~/.clawdbot/moltbot.json
```

### 5. Start the Bot

```bash
# Start as a service (recommended)
sudo systemctl start clawdbot
sudo systemctl enable clawdbot  # auto-start on boot

# Or run manually
source .env && moltbot gateway --port 18789 --verbose
```

### 6. Check Status

```bash
# View logs
sudo journalctl -u clawdbot -f

# Check service status
sudo systemctl status clawdbot
```

## Configuration

### Environment Variables (.env)

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Yes |
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token | Yes |
| `OPENAI_API_KEY` | OpenAI API key (optional) | No |
| `ELEVENLABS_API_KEY` | ElevenLabs API key for voice | No |

### Moltbot Configuration (~/.clawdbot/moltbot.json)

```json
{
  "agent": {
    "model": "anthropic/claude-sonnet-4-20250514",
    "thinking": "medium"
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
  }
}
```

## Telegram Commands

Once running, interact with your bot in Telegram:

| Command | Description |
|---------|-------------|
| `/status` | View session details and token usage |
| `/new` or `/reset` | Clear session context |
| `/think <level>` | Set reasoning level (low/medium/high) |
| `/verbose on\|off` | Toggle verbose mode |
| `/restart` | Restart the gateway |

## Troubleshooting

### Bot not responding

1. Check if the service is running:
   ```bash
   sudo systemctl status clawdbot
   ```

2. Check logs for errors:
   ```bash
   sudo journalctl -u clawdbot -n 100
   ```

3. Verify your tokens are correct in `.env`

### Connection issues

1. Ensure port 18789 is open (for local gateway)
2. Check DigitalOcean firewall allows outbound HTTPS (443)

### Restart the bot

```bash
sudo systemctl restart clawdbot
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

- Never commit `.env` or files containing API keys
- The `.env` file is gitignored by default
- Use DigitalOcean Cloud Firewalls to restrict access
- Consider enabling DigitalOcean Monitoring for observability

## License

MIT
