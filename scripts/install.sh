#!/bin/bash
set -e

# ClawdBot One-Line Installer
# Usage: curl -sSL https://raw.githubusercontent.com/DGator86/ClawdBot-V1/claude/setup-aws-vps-Dzt6f/scripts/install.sh | bash

echo "=========================================="
echo "  ClawdBot Installer"
echo "=========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Update system
echo -e "${YELLOW}[1/6] Updating system packages...${NC}"
sudo apt-get update -y
sudo apt-get install -y curl git jq

# Install Node.js 22
echo -e "${YELLOW}[2/6] Installing Node.js 22...${NC}"
if ! command -v node &> /dev/null || [ "$(node -v | cut -d'v' -f2 | cut -d'.' -f1)" -lt 22 ]; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi
echo -e "${GREEN}Node.js $(node -v) installed${NC}"

# Install moltbot
echo -e "${YELLOW}[3/6] Installing moltbot...${NC}"
sudo npm install -g moltbot@latest
echo -e "${GREEN}moltbot installed${NC}"

# Clone repo
echo -e "${YELLOW}[4/6] Cloning ClawdBot...${NC}"
cd ~
if [ -d "ClawdBot-V1" ]; then
    cd ClawdBot-V1
    git pull origin claude/setup-aws-vps-Dzt6f || true
else
    git clone https://github.com/DGator86/ClawdBot-V1.git
    cd ClawdBot-V1
    git checkout claude/setup-aws-vps-Dzt6f
fi

# Setup config
echo -e "${YELLOW}[5/6] Setting up configuration...${NC}"
mkdir -p ~/.clawdbot

# Prompt for credentials
echo ""
echo -e "${YELLOW}Enter your Telegram Bot Token:${NC}"
read -r TELEGRAM_TOKEN

echo -e "${YELLOW}Enter your OpenAI API Key:${NC}"
read -r OPENAI_KEY

# Create .env
cat > .env << EOF
TELEGRAM_BOT_TOKEN=$TELEGRAM_TOKEN
OPENAI_API_KEY=$OPENAI_KEY
EOF
chmod 600 .env

# Create moltbot config
cat > ~/.clawdbot/moltbot.json << EOF
{
  "agent": {
    "model": "openai/gpt-4o",
    "thinking": "medium"
  },
  "gateway": {
    "port": 18789,
    "bind": "127.0.0.1"
  },
  "channels": {
    "telegram": {
      "enabled": true,
      "botToken": "$TELEGRAM_TOKEN"
    }
  },
  "agents": {
    "defaults": {
      "sandbox": {
        "mode": "disabled"
      }
    }
  }
}
EOF

# Install systemd service
echo -e "${YELLOW}[6/6] Installing systemd service...${NC}"
sudo tee /etc/systemd/system/clawdbot.service > /dev/null << EOF
[Unit]
Description=ClawdBot - Telegram AI Assistant
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/ClawdBot-V1
Environment=NODE_ENV=production
Environment=OPENAI_API_KEY=$OPENAI_KEY
Environment=TELEGRAM_BOT_TOKEN=$TELEGRAM_TOKEN
ExecStart=/usr/bin/moltbot gateway --port 18789
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=clawdbot

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable clawdbot

echo ""
echo -e "${GREEN}=========================================="
echo "  Installation Complete!"
echo "==========================================${NC}"
echo ""
echo "Start ClawdBot:"
echo "  sudo systemctl start clawdbot"
echo ""
echo "View logs:"
echo "  sudo journalctl -u clawdbot -f"
echo ""
echo -e "${YELLOW}Starting ClawdBot now...${NC}"
sudo systemctl start clawdbot
sleep 3
sudo systemctl status clawdbot --no-pager
