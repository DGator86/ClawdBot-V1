#!/bin/bash
set -e

# ClawdBot VPS Deployment Script
# Run this on your AWS VPS to set up ClawdBot

echo "=========================================="
echo "  ClawdBot Deployment Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Please do not run this script as root${NC}"
    echo "Run as a regular user with sudo privileges"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo -e "${RED}Cannot detect OS${NC}"
    exit 1
fi

echo -e "${GREEN}Detected OS: $OS${NC}"

# Update system packages
echo -e "${YELLOW}Updating system packages...${NC}"
sudo apt-get update -y

# Install Node.js 22 if not present
if ! command -v node &> /dev/null || [ "$(node -v | cut -d'v' -f2 | cut -d'.' -f1)" -lt 22 ]; then
    echo -e "${YELLOW}Installing Node.js 22...${NC}"
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y nodejs
else
    echo -e "${GREEN}Node.js $(node -v) already installed${NC}"
fi

# Install pnpm
if ! command -v pnpm &> /dev/null; then
    echo -e "${YELLOW}Installing pnpm...${NC}"
    sudo npm install -g pnpm
else
    echo -e "${GREEN}pnpm already installed${NC}"
fi

# Install moltbot globally
echo -e "${YELLOW}Installing moltbot...${NC}"
sudo npm install -g moltbot@latest

# Create config directory
echo -e "${YELLOW}Setting up configuration...${NC}"
mkdir -p ~/.clawdbot

# Check if config exists
if [ ! -f ~/.clawdbot/moltbot.json ]; then
    echo -e "${YELLOW}Creating default configuration...${NC}"
    cp config/moltbot.example.json ~/.clawdbot/moltbot.json
    echo -e "${RED}IMPORTANT: Edit ~/.clawdbot/moltbot.json with your settings${NC}"
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${RED}IMPORTANT: Edit .env with your API keys${NC}"
fi

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Install systemd service
echo -e "${YELLOW}Installing systemd service...${NC}"
sudo cp scripts/clawdbot.service /etc/systemd/system/
sudo sed -i "s|%USER%|$USER|g" /etc/systemd/system/clawdbot.service
sudo sed -i "s|%HOME%|$HOME|g" /etc/systemd/system/clawdbot.service
sudo systemctl daemon-reload

echo ""
echo -e "${GREEN}=========================================="
echo "  Deployment Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Edit your environment variables:"
echo "   nano .env"
echo ""
echo "2. Edit your moltbot configuration:"
echo "   nano ~/.clawdbot/moltbot.json"
echo ""
echo "3. Run the onboarding wizard:"
echo "   moltbot onboard"
echo ""
echo "4. Start the service:"
echo "   sudo systemctl start clawdbot"
echo "   sudo systemctl enable clawdbot  # auto-start on boot"
echo ""
echo "5. Check logs:"
echo "   sudo journalctl -u clawdbot -f"
echo ""
echo -e "${YELLOW}Your VPS IP: $(curl -s ifconfig.me)${NC}"
