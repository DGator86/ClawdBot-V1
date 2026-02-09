#!/bin/bash
set -e

echo "=========================================="
echo "  Kalshi Crypto Trading Bot - Deployment"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root${NC}"
  exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
  echo -e "${RED}ERROR: .env file not found!${NC}"
  echo "Create .env with your Kalshi credentials:"
  echo "  cp .env.example .env"
  echo "  nano .env"
  exit 1
fi

# Install dependencies
echo -e "${YELLOW}[1/4] Installing dependencies...${NC}"
npm install

# Copy to deployment location
echo -e "${YELLOW}[2/4] Setting up deployment directory...${NC}"
mkdir -p /root/kalshi-bot
cp -r src /root/kalshi-bot/
cp package.json /root/kalshi-bot/
cp .env /root/kalshi-bot/
chmod 600 /root/kalshi-bot/.env

cd /root/kalshi-bot
npm install --production

# Install systemd service
echo -e "${YELLOW}[3/4] Installing systemd service...${NC}"
cp /root/ClawdBot-V1/kalshi-bot/kalshi-bot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable kalshi-bot

# Start the service
echo -e "${YELLOW}[4/4] Starting Kalshi Bot...${NC}"
systemctl start kalshi-bot

echo ""
echo -e "${GREEN}=========================================="
echo "  Deployment Complete!"
echo "==========================================${NC}"
echo ""
echo "Commands:"
echo "  systemctl status kalshi-bot   - Check status"
echo "  journalctl -u kalshi-bot -f   - View logs"
echo "  systemctl restart kalshi-bot  - Restart"
echo ""
echo "API running at: http://127.0.0.1:3456"
echo ""
echo "Now configure moltbot to use the Kalshi skill."
echo ""
