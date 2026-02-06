#!/bin/bash
set -e

# ==========================================================
#  ClawdBot + Yoshi Bridge Deployment Script
#  Run this on the Clawd-Server VPS (165.245.140.115)
#
#  Deploys:
#    1. ClawdBot (moltbot gateway) with yoshi-trading skill
#    2. Yoshi-Bridge (scanner log -> Trading Core /propose)
#    3. Systemd services for both
#
#  Prerequisites:
#    - Yoshi-Bot already deployed with Trading Core on :8000
#    - Node.js 22+ installed
#    - .env with TELEGRAM_BOT_TOKEN and OPENAI_API_KEY
# ==========================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Auto-detect paths — check /root first (DO droplet), then /home/root
if [ -d "/root/Yoshi-Bot" ]; then
    _DEFAULT_YOSHI="/root/Yoshi-Bot"
    _DEFAULT_CLAWD="/root/ClawdBot-V1"
elif [ -d "/home/root/Yoshi-Bot" ]; then
    _DEFAULT_YOSHI="/home/root/Yoshi-Bot"
    _DEFAULT_CLAWD="/home/root/ClawdBot-V1"
else
    _DEFAULT_YOSHI="$HOME/Yoshi-Bot"
    _DEFAULT_CLAWD="$HOME/ClawdBot-V1"
fi

CLAWDBOT_DIR="${CLAWDBOT_DIR:-$_DEFAULT_CLAWD}"
YOSHI_DIR="${YOSHI_DIR:-$_DEFAULT_YOSHI}"
CLAWDBOT_REPO="https://github.com/DGator86/ClawdBot-V1.git"
CLAWDBOT_BRANCH="genspark_ai_developer"

echo -e "${CYAN}=========================================="
echo "  ClawdBot + Yoshi Bridge Deployer"
echo -e "==========================================${NC}"
echo ""

# ------- 1. System Check -------
echo -e "${YELLOW}[1/8] System checks...${NC}"

# Verify Yoshi Trading Core is running
if curl -s http://127.0.0.1:8000/health | grep -q "healthy"; then
    echo -e "${GREEN}  Yoshi Trading Core: RUNNING${NC}"
else
    echo -e "${RED}  Yoshi Trading Core: NOT RUNNING on :8000${NC}"
    echo "  Please start Yoshi-Bot first."
    echo "  cd $YOSHI_DIR && source venv/bin/activate"
    echo "  Then start the Trading Core API."
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null || [ "$(node -v | cut -d'v' -f2 | cut -d'.' -f1)" -lt 22 ]; then
    echo -e "${YELLOW}  Installing Node.js 22...${NC}"
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y nodejs
fi
echo -e "${GREEN}  Node.js: $(node -v)${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}  Python3 not found!${NC}"
    exit 1
fi
echo -e "${GREEN}  Python: $(python3 --version)${NC}"

# ------- 2. Clone / Update ClawdBot -------
echo -e "\n${YELLOW}[2/8] Setting up ClawdBot repository...${NC}"

if [ -d "$CLAWDBOT_DIR" ]; then
    echo "  Updating existing repo..."
    cd "$CLAWDBOT_DIR"
    git fetch origin
    git checkout "$CLAWDBOT_BRANCH" 2>/dev/null || git checkout -b "$CLAWDBOT_BRANCH" "origin/$CLAWDBOT_BRANCH" 2>/dev/null || true
    git reset --hard "origin/$CLAWDBOT_BRANCH" 2>/dev/null || git pull origin "$CLAWDBOT_BRANCH" || true
else
    echo "  Cloning ClawdBot..."
    git clone "$CLAWDBOT_REPO" "$CLAWDBOT_DIR"
    cd "$CLAWDBOT_DIR"
    git checkout "$CLAWDBOT_BRANCH"
fi

# Install node deps
if [ -f "$CLAWDBOT_DIR/package.json" ]; then
    cd "$CLAWDBOT_DIR"
    npm install --silent 2>/dev/null
    echo -e "${GREEN}  npm install: done${NC}"
fi
echo -e "${GREEN}  ClawdBot repo: $CLAWDBOT_DIR${NC}"

# ------- 3. Install moltbot -------
echo -e "\n${YELLOW}[3/8] Installing moltbot...${NC}"

if ! command -v moltbot &> /dev/null; then
    npm install -g moltbot@latest
fi
echo -e "${GREEN}  moltbot: $(moltbot --version 2>/dev/null || echo 'installed')${NC}"

# ------- 4. Configure moltbot -------
echo -e "\n${YELLOW}[4/8] Configuring moltbot...${NC}"

mkdir -p ~/.clawdbot

# Create .env if missing
if [ ! -f "$CLAWDBOT_DIR/.env" ]; then
    if [ -f "$CLAWDBOT_DIR/.env.example" ]; then
        cp "$CLAWDBOT_DIR/.env.example" "$CLAWDBOT_DIR/.env"
        echo -e "${RED}  IMPORTANT: Edit $CLAWDBOT_DIR/.env with your API keys${NC}"
    fi
fi

# Copy config
cp "$CLAWDBOT_DIR/config/moltbot.example.json" ~/.clawdbot/moltbot.json

# Inject Telegram token from .env if available
if [ -f "$CLAWDBOT_DIR/.env" ]; then
    source "$CLAWDBOT_DIR/.env"
    if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ "$TELEGRAM_BOT_TOKEN" != "your_telegram_bot_token_here" ]; then
        # Use python3 for safe JSON editing
        python3 -c "
import json
with open('$HOME/.clawdbot/moltbot.json') as f:
    cfg = json.load(f)
cfg['channels']['telegram']['botToken'] = '$TELEGRAM_BOT_TOKEN'
with open('$HOME/.clawdbot/moltbot.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('  Telegram token injected')
"
    fi
fi

# Update skills path to point to ClawdBot's skills directory
python3 -c "
import json
with open('$HOME/.clawdbot/moltbot.json') as f:
    cfg = json.load(f)
cfg.setdefault('skills', {}).setdefault('load', {})['extraDirs'] = ['$CLAWDBOT_DIR/skills']
with open('$HOME/.clawdbot/moltbot.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('  Skills path configured: $CLAWDBOT_DIR/skills')
"

echo -e "${GREEN}  Config: ~/.clawdbot/moltbot.json${NC}"

# ------- 5. Kalshi API Credentials -------
echo -e "\n${YELLOW}[5/8] Checking Kalshi API credentials...${NC}"

# Check if Yoshi-Bot .env has Kalshi credentials
YOSHI_ENV="$YOSHI_DIR/.env"
if [ -f "$YOSHI_ENV" ]; then
    source "$YOSHI_ENV"
fi

if [ -z "$KALSHI_KEY_ID" ] || [ "$KALSHI_KEY_ID" = "your_kalshi_key_id_here" ]; then
    echo -e "${YELLOW}  Kalshi credentials not found in $YOSHI_ENV${NC}"
    echo ""
    read -p "  Enter Kalshi API Key ID (or press Enter to skip): " KALSHI_KEY_ID_INPUT

    if [ -n "$KALSHI_KEY_ID_INPUT" ]; then
        # Append to Yoshi .env
        echo "" >> "$YOSHI_ENV"
        echo "# Kalshi API Credentials" >> "$YOSHI_ENV"
        echo "KALSHI_KEY_ID=$KALSHI_KEY_ID_INPUT" >> "$YOSHI_ENV"

        # Check for private key file or inline
        echo ""
        echo "  For the Kalshi private key, you can either:"
        echo "    1) Provide a path to the PEM file"
        echo "    2) Paste the key inline (will be written to ~/.kalshi/private_key.pem)"
        echo ""
        read -p "  Enter PEM file path (or 'paste' to enter inline, or Enter to skip): " KEY_INPUT

        if [ "$KEY_INPUT" = "paste" ]; then
            mkdir -p ~/.kalshi
            echo "  Paste your RSA private key (end with Ctrl+D on a new line):"
            cat > ~/.kalshi/private_key.pem
            chmod 600 ~/.kalshi/private_key.pem
            # Read the key and write to .env with proper escaping
            KALSHI_KEY_CONTENT=$(cat ~/.kalshi/private_key.pem)
            echo "KALSHI_PRIVATE_KEY=\"$KALSHI_KEY_CONTENT\"" >> "$YOSHI_ENV"
            echo -e "${GREEN}  Private key saved to ~/.kalshi/private_key.pem and added to .env${NC}"
        elif [ -n "$KEY_INPUT" ] && [ -f "$KEY_INPUT" ]; then
            KALSHI_KEY_CONTENT=$(cat "$KEY_INPUT")
            echo "KALSHI_PRIVATE_KEY=\"$KALSHI_KEY_CONTENT\"" >> "$YOSHI_ENV"
            # Also copy to secure location
            mkdir -p ~/.kalshi
            cp "$KEY_INPUT" ~/.kalshi/private_key.pem
            chmod 600 ~/.kalshi/private_key.pem
            echo -e "${GREEN}  Private key loaded from $KEY_INPUT${NC}"
        else
            echo -e "${YELLOW}  Skipped. Add KALSHI_PRIVATE_KEY to $YOSHI_ENV manually.${NC}"
        fi

        # Also add to ClawdBot .env for reference
        if [ -f "$CLAWDBOT_DIR/.env" ]; then
            grep -q "KALSHI_KEY_ID" "$CLAWDBOT_DIR/.env" || {
                echo "" >> "$CLAWDBOT_DIR/.env"
                echo "# Kalshi API (used by Yoshi-Bot, referenced here for completeness)" >> "$CLAWDBOT_DIR/.env"
                echo "KALSHI_KEY_ID=$KALSHI_KEY_ID_INPUT" >> "$CLAWDBOT_DIR/.env"
            }
        fi

        chmod 600 "$YOSHI_ENV"
        echo -e "${GREEN}  Kalshi credentials configured${NC}"
    else
        echo -e "${YELLOW}  Skipped Kalshi setup. Add credentials to $YOSHI_ENV later.${NC}"
    fi
else
    echo -e "${GREEN}  Kalshi Key ID: ${KALSHI_KEY_ID:0:8}... (found)${NC}"

    # Verify Kalshi connectivity
    cd "$YOSHI_DIR" && source venv/bin/activate 2>/dev/null
    python3 -c "
import sys, os, importlib.util
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('kalshi_client', os.path.join('src', 'gnosis', 'utils', 'kalshi_client.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
KalshiClient = mod.KalshiClient
try:
    c = KalshiClient()
    s = c.get_exchange_status()
    if s:
        print('  Kalshi API: CONNECTED')
        print(f'    Exchange active: {s.get(\"exchange_active\", \"?\")}')
        print(f'    Trading active: {s.get(\"trading_active\", \"?\")}')
    else:
        print('  Kalshi API: no status returned')
except Exception as e:
    print(f'  Kalshi API: {e}')
" 2>/dev/null || echo -e "${YELLOW}  Could not verify Kalshi (Python env issue)${NC}"
    cd "$CLAWDBOT_DIR"
fi

# ------- 6. Install yoshi-bridge systemd service -------
echo -e "\n${YELLOW}[6/8] Installing yoshi-bridge service...${NC}"

SCANNER_LOG="$YOSHI_DIR/logs/scanner.log"
mkdir -p "$YOSHI_DIR/logs"

cat > /etc/systemd/system/yoshi-bridge.service << EOF
[Unit]
Description=Yoshi-Bridge — Scanner signals to Trading Core
After=network.target
Wants=clawdbot.service

[Service]
Type=simple
User=root
WorkingDirectory=$CLAWDBOT_DIR
ExecStart=/usr/bin/python3 $CLAWDBOT_DIR/scripts/yoshi-bridge.py --log-path $SCANNER_LOG --poll-interval 30 --min-edge 5.0
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=yoshi-bridge

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}  Service: yoshi-bridge.service${NC}"

# ------- 7. Install clawdbot systemd service -------
echo -e "\n${YELLOW}[7/8] Installing clawdbot service...${NC}"

cat > /etc/systemd/system/clawdbot.service << EOF
[Unit]
Description=ClawdBot — Telegram AI Trading Assistant (moltbot gateway)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$CLAWDBOT_DIR
EnvironmentFile=$CLAWDBOT_DIR/.env
ExecStart=/usr/bin/moltbot gateway --port 18789
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=clawdbot

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}  Service: clawdbot.service${NC}"

# ------- 8. Enable and start services -------
echo -e "\n${YELLOW}[8/8] Starting services...${NC}"

systemctl daemon-reload
systemctl enable yoshi-bridge clawdbot

# Start yoshi-bridge first (it just watches logs)
systemctl restart yoshi-bridge
sleep 2

# Start clawdbot (moltbot gateway + Telegram)
systemctl restart clawdbot
sleep 3

echo ""
echo -e "${CYAN}=========================================="
echo "  Deployment Complete!"
echo -e "==========================================${NC}"
echo ""
echo "Services:"
echo -e "  ${GREEN}clawdbot${NC}      — moltbot gateway on :18789 + Telegram"
echo -e "  ${GREEN}yoshi-bridge${NC}  — scanner log watcher -> Trading Core /propose"
echo ""
echo "Status:"
systemctl status clawdbot --no-pager -l 2>/dev/null | head -5 || true
echo ""
systemctl status yoshi-bridge --no-pager -l 2>/dev/null | head -5 || true
echo ""
echo "Useful commands:"
echo "  sudo journalctl -u clawdbot -f       # ClawdBot logs"
echo "  sudo journalctl -u yoshi-bridge -f   # Bridge logs"
echo "  sudo systemctl restart clawdbot      # Restart ClawdBot"
echo "  sudo systemctl restart yoshi-bridge  # Restart bridge"
echo "  curl -s http://127.0.0.1:8000/status # Yoshi Trading Core"
echo "  moltbot status                       # Moltbot status"
echo ""
echo -e "${CYAN}Architecture:${NC}"
echo "  Yoshi Scanner -> scanner.log -> yoshi-bridge -> Trading Core :8000 /propose"
echo "  ClawdBot (moltbot :18789) -> reads Trading Core :8000 -> Telegram suggestions"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Verify .env has TELEGRAM_BOT_TOKEN and OPENAI_API_KEY"
echo "  2. Send a message to your bot on Telegram"
echo "  3. Try: 'What is Yoshi's status?' or 'Show me positions'"
