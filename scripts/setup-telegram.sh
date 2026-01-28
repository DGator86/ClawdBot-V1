#!/bin/bash
set -e

# ClawdBot Telegram Setup Script
# Configures Telegram bot integration

echo "=========================================="
echo "  ClawdBot Telegram Setup"
echo "=========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if .env exists
if [ ! -f .env ]; then
    cp .env.example .env
fi

# Prompt for Telegram token if not set
if ! grep -q "^TELEGRAM_BOT_TOKEN=.\+$" .env 2>/dev/null || grep -q "your_telegram_bot_token_here" .env; then
    echo -e "${YELLOW}Enter your Telegram Bot Token (from @BotFather):${NC}"
    read -r TELEGRAM_TOKEN

    if [ -n "$TELEGRAM_TOKEN" ]; then
        sed -i "s|TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=$TELEGRAM_TOKEN|" .env
        echo -e "${GREEN}Telegram token saved to .env${NC}"
    else
        echo -e "${RED}No token provided, skipping...${NC}"
    fi
fi

# Prompt for Anthropic API key if not set
if ! grep -q "^ANTHROPIC_API_KEY=.\+$" .env 2>/dev/null || grep -q "your_anthropic_api_key_here" .env; then
    echo -e "${YELLOW}Enter your Anthropic API Key (from console.anthropic.com):${NC}"
    read -r ANTHROPIC_KEY

    if [ -n "$ANTHROPIC_KEY" ]; then
        sed -i "s|ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=$ANTHROPIC_KEY|" .env
        echo -e "${GREEN}Anthropic API key saved to .env${NC}"
    else
        echo -e "${RED}No key provided, skipping...${NC}"
    fi
fi

# Update moltbot.json with Telegram config
CONFIG_FILE="$HOME/.clawdbot/moltbot.json"
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}Updating moltbot configuration...${NC}"

    # Load token from .env
    source .env

    # Use jq if available, otherwise use sed
    if command -v jq &> /dev/null; then
        jq --arg token "$TELEGRAM_BOT_TOKEN" '.channels.telegram.botToken = $token' "$CONFIG_FILE" > "$CONFIG_FILE.tmp" && mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
    else
        sed -i "s|\${TELEGRAM_BOT_TOKEN}|$TELEGRAM_BOT_TOKEN|g" "$CONFIG_FILE"
    fi

    echo -e "${GREEN}Configuration updated${NC}"
fi

echo ""
echo -e "${GREEN}Telegram setup complete!${NC}"
echo ""
echo "To start ClawdBot:"
echo "  sudo systemctl start clawdbot"
echo ""
echo "Or run manually:"
echo "  source .env && moltbot gateway --port 18789 --verbose"
