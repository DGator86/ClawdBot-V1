#!/bin/bash
# ============================================================
#  Kalshi API Credential Setup for Yoshi-Bot
#  Run on VPS: bash scripts/setup-kalshi.sh
#
#  This script:
#    1. Writes KALSHI_KEY_ID to Yoshi-Bot's .env
#    2. Saves the RSA private key to ~/.kalshi/private_key.pem
#    3. Writes KALSHI_PRIVATE_KEY to Yoshi-Bot's .env
#    4. Verifies Kalshi API connectivity
#    5. Restarts the Kalshi scanner
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Auto-detect Yoshi-Bot location
if [ -n "$YOSHI_DIR" ] && [ -d "$YOSHI_DIR" ]; then
    : # Already set via env
elif [ -d "/root/Yoshi-Bot" ]; then
    YOSHI_DIR="/root/Yoshi-Bot"
elif [ -d "/home/root/Yoshi-Bot" ]; then
    YOSHI_DIR="/home/root/Yoshi-Bot"
else
    YOSHI_DIR="${YOSHI_DIR:-/root/Yoshi-Bot}"
fi
YOSHI_ENV="$YOSHI_DIR/.env"
KALSHI_KEY_DIR="$HOME/.kalshi"
KALSHI_KEY_FILE="$KALSHI_KEY_DIR/private_key.pem"

echo -e "${CYAN}=========================================="
echo "  Kalshi API Setup for Yoshi-Bot"
echo -e "==========================================${NC}"
echo ""

# Check Yoshi-Bot exists
if [ ! -d "$YOSHI_DIR" ]; then
    echo -e "${RED}Error: Yoshi-Bot not found at $YOSHI_DIR${NC}"
    exit 1
fi

# Check .env exists
if [ ! -f "$YOSHI_ENV" ]; then
    echo -e "${RED}Error: $YOSHI_ENV not found${NC}"
    exit 1
fi

# ------- Step 1: Key ID -------
echo -e "${YELLOW}[1/5] Kalshi API Key ID${NC}"

EXISTING_KEY_ID=$(grep "^KALSHI_KEY_ID=" "$YOSHI_ENV" 2>/dev/null | cut -d'=' -f2)
if [ -n "$EXISTING_KEY_ID" ] && [ "$EXISTING_KEY_ID" != "your_kalshi_key_id_here" ]; then
    echo -e "  Found existing: ${GREEN}${EXISTING_KEY_ID:0:12}...${NC}"
    read -p "  Keep existing? (Y/n): " KEEP_KEY
    if [ "$KEEP_KEY" = "n" ] || [ "$KEEP_KEY" = "N" ]; then
        read -p "  Enter new Kalshi API Key ID: " NEW_KEY_ID
        sed -i "s|^KALSHI_KEY_ID=.*|KALSHI_KEY_ID=$NEW_KEY_ID|" "$YOSHI_ENV"
        echo -e "  ${GREEN}Updated Key ID${NC}"
    fi
else
    read -p "  Enter Kalshi API Key ID: " NEW_KEY_ID
    if [ -z "$NEW_KEY_ID" ]; then
        echo -e "${RED}  Key ID is required. Exiting.${NC}"
        exit 1
    fi
    # Remove any existing placeholder and append
    sed -i '/^KALSHI_KEY_ID=/d' "$YOSHI_ENV"
    echo "" >> "$YOSHI_ENV"
    echo "# Kalshi API Credentials" >> "$YOSHI_ENV"
    echo "KALSHI_KEY_ID=$NEW_KEY_ID" >> "$YOSHI_ENV"
    echo -e "  ${GREEN}Key ID saved${NC}"
fi

# ------- Step 2: Private Key File -------
echo -e "\n${YELLOW}[2/5] RSA Private Key${NC}"

mkdir -p "$KALSHI_KEY_DIR"
chmod 700 "$KALSHI_KEY_DIR"

if [ -f "$KALSHI_KEY_FILE" ]; then
    echo -e "  Found existing key at ${GREEN}$KALSHI_KEY_FILE${NC}"
    read -p "  Replace it? (y/N): " REPLACE_KEY
    if [ "$REPLACE_KEY" != "y" ] && [ "$REPLACE_KEY" != "Y" ]; then
        echo "  Keeping existing key."
    else
        echo "  Paste your RSA private key below."
        echo "  (Include -----BEGIN RSA PRIVATE KEY----- and -----END RSA PRIVATE KEY-----)"
        echo "  Press Ctrl+D on a new line when done:"
        echo ""
        cat > "$KALSHI_KEY_FILE"
        chmod 600 "$KALSHI_KEY_FILE"
        echo -e "\n  ${GREEN}Private key saved to $KALSHI_KEY_FILE${NC}"
    fi
else
    echo "  Paste your RSA private key below."
    echo "  (Include -----BEGIN RSA PRIVATE KEY----- and -----END RSA PRIVATE KEY-----)"
    echo "  Press Ctrl+D on a new line when done:"
    echo ""
    cat > "$KALSHI_KEY_FILE"
    chmod 600 "$KALSHI_KEY_FILE"
    echo -e "\n  ${GREEN}Private key saved to $KALSHI_KEY_FILE${NC}"
fi

# ------- Step 3: Write key to .env -------
echo -e "\n${YELLOW}[3/5] Writing private key to .env${NC}"

# Read the key content
KALSHI_KEY_CONTENT=$(cat "$KALSHI_KEY_FILE")

# Remove any existing KALSHI_PRIVATE_KEY block
sed -i '/^KALSHI_PRIVATE_KEY=/d' "$YOSHI_ENV"

# Write with proper quoting (single-line with literal \n)
KALSHI_KEY_ONELINE=$(echo "$KALSHI_KEY_CONTENT" | tr '\n' '|' | sed 's/|$//')
# Actually, python's dotenv can handle multi-line in quotes
echo "KALSHI_PRIVATE_KEY=\"$KALSHI_KEY_CONTENT\"" >> "$YOSHI_ENV"

chmod 600 "$YOSHI_ENV"
echo -e "  ${GREEN}Private key added to $YOSHI_ENV${NC}"

# ------- Step 4: Verify Kalshi API -------
echo -e "\n${YELLOW}[4/5] Verifying Kalshi API connectivity...${NC}"

cd "$YOSHI_DIR"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Source the env to make credentials available
set -a
source "$YOSHI_ENV"
set +a

python3 -c "
import sys, os, importlib.util
sys.path.insert(0, '.')
# Direct file import to avoid __init__.py chain issues
spec = importlib.util.spec_from_file_location('kalshi_client', os.path.join('src', 'gnosis', 'utils', 'kalshi_client.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
KalshiClient = mod.KalshiClient
import json

try:
    client = KalshiClient()
    print('  Key loaded successfully')

    status = client.get_exchange_status()
    if status:
        print(f'  Exchange active: {status.get(\"exchange_active\", \"unknown\")}')
        print(f'  Trading active: {status.get(\"trading_active\", \"unknown\")}')
        print('  \033[0;32mKalshi API: CONNECTED\033[0m')
    else:
        print('  \033[1;33mKalshi API: Connected but exchange returned no status\033[0m')
        print('  (Exchange may be closed - check trading hours)')

    # Try listing BTC markets
    markets = client.list_markets(limit=5, series_ticker='KXBTC', status='open')
    print(f'  Active KXBTC markets: {len(markets)}')
    if markets:
        for m in markets[:3]:
            ticker = m.get('ticker', 'N/A')
            y_bid = m.get('yes_bid', 0)
            y_ask = m.get('yes_ask', 100)
            print(f'    {ticker}  bid/ask: {y_bid}/{y_ask}')

except ValueError as e:
    print(f'  \033[0;31mCredential Error: {e}\033[0m')
    sys.exit(1)
except Exception as e:
    print(f'  \033[0;31mAPI Error: {e}\033[0m')
    sys.exit(1)
" || {
    echo -e "  ${RED}Kalshi verification failed!${NC}"
    echo "  Check credentials and try again."
    exit 1
}

# ------- Step 5: Restart scanner -------
echo -e "\n${YELLOW}[5/5] Restarting Kalshi scanner...${NC}"

# Check if scanner is running
if pgrep -f "kalshi_scanner.py" > /dev/null; then
    echo "  Stopping existing scanner..."
    pkill -f "kalshi_scanner.py" || true
    sleep 2
fi

# Start scanner in background
mkdir -p "$YOSHI_DIR/logs"
cd "$YOSHI_DIR"
nohup python3 scripts/kalshi_scanner.py \
    --symbol BTCUSDT --loop --interval 300 \
    --threshold 0.10 --live --exchange kraken \
    > logs/scanner.log 2>&1 &

SCANNER_PID=$!
sleep 3

if kill -0 "$SCANNER_PID" 2>/dev/null; then
    echo -e "  ${GREEN}Scanner started (PID: $SCANNER_PID)${NC}"
else
    echo -e "  ${YELLOW}Scanner may have exited. Check logs:${NC}"
    echo "  tail -20 $YOSHI_DIR/logs/scanner.log"
fi

echo ""
echo -e "${CYAN}=========================================="
echo "  Kalshi Setup Complete!"
echo -e "==========================================${NC}"
echo ""
echo "  Key ID:      $(grep '^KALSHI_KEY_ID=' "$YOSHI_ENV" | cut -d'=' -f2 | cut -c1-12)..."
echo "  Private Key:  $KALSHI_KEY_FILE ($(stat -c%a "$KALSHI_KEY_FILE") permissions)"
echo "  Scanner Log:  $YOSHI_DIR/logs/scanner.log"
echo ""
echo "Verify:"
echo "  tail -f $YOSHI_DIR/logs/scanner.log"
echo "  curl -s http://127.0.0.1:8000/status | python3 -m json.tool"
echo ""
echo "Then tell ClawdBot on Telegram:"
echo '  "Is Kalshi connected?"'
echo '  "Show me active BTC Kalshi markets"'
