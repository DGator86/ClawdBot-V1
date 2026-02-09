#!/bin/bash
set -e

# ================================================================
#  ClawdBot + Yoshi-Bot + Kalshi — ALL-IN-ONE Setup
#
#  ONE COMMAND (paste in SSH):
#    curl -sSL https://raw.githubusercontent.com/DGator86/ClawdBot-V1/genspark_ai_developer/scripts/setup-all.sh -o /tmp/setup.sh && bash /tmp/setup.sh
#
#  What this does:
#    1. Installs Node.js, moltbot, dependencies
#    2. Clones/updates ClawdBot with yoshi-trading skill
#    3. Finds Yoshi-Bot + Trading Core on this server
#    4. Auto-discovers Kalshi private key on disk
#    5. Pulls Telegram/OpenAI keys from Yoshi's .env
#    6. Configures moltbot + systemd services
#    7. Starts everything and verifies
#
#  Phone-friendly: auto-discovers everything, minimal typing
# ================================================================

# When piped through curl, stdin is the script itself.
# Reopen stdin from terminal for interactive prompts.
exec < /dev/tty 2>/dev/null || true

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

CLAWDBOT_REPO="https://github.com/DGator86/ClawdBot-V1.git"
# Ultimate-fix: default to ultimate-fix branch for enhanced ML + regime gating
CLAWDBOT_BRANCH="${CLAWDBOT_BRANCH:-ultimate-fix}"
KALSHI_KEY_ID="6858062e-6884-43b5-b002-0e13391be331"
KALSHI_KEY_DIR="$HOME/.kalshi"
KALSHI_KEY_FILE="$KALSHI_KEY_DIR/private_key.pem"

header() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════${NC}"
    echo ""
}

step() { echo -e "${YELLOW}[$1/$TOTAL] $2${NC}"; }
ok()   { echo -e "  ${GREEN}✓ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "  ${RED}✗ $1${NC}"; }

TOTAL=9

header "ClawdBot + Yoshi + Kalshi"

# ================================================================
# 1. System
# ================================================================
step 1 "System packages + ML dependencies"
apt-get update -y -qq >/dev/null 2>&1
apt-get install -y -qq curl git jq python3 python3-pip python3-venv build-essential >/dev/null 2>&1
pip3 install cryptography numpy -q 2>/dev/null || pip3 install cryptography numpy --break-system-packages -q 2>/dev/null || true
# Ultimate-fix: install ML stack
pip3 install lightgbm scikit-learn scipy pandas pyarrow requests --break-system-packages -q 2>/dev/null \
    || pip3 install lightgbm scikit-learn scipy pandas pyarrow requests -q 2>/dev/null || true
ok "Installed (system + ML)"

# ================================================================
# 2. Node.js + moltbot
# ================================================================
step 2 "Node.js + moltbot"
if ! command -v node &>/dev/null || [ "$(node -v | cut -d'v' -f2 | cut -d'.' -f1)" -lt 22 ]; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1
    apt-get install -y -qq nodejs >/dev/null 2>&1
fi
ok "Node $(node -v)"

if ! command -v moltbot &>/dev/null; then
    npm install -g moltbot@latest --silent >/dev/null 2>&1
fi
ok "moltbot $(moltbot --version 2>/dev/null || echo 'ok')"

# ================================================================
# 3. Find Yoshi-Bot
# ================================================================
step 3 "Finding Yoshi-Bot"

YOSHI_DIR=""
YOSHI_FOUND=false
for D in /root/Yoshi-Bot /home/root/Yoshi-Bot /home/*/Yoshi-Bot "$HOME/Yoshi-Bot"; do
    if [ -d "$D" ] && [ -f "$D/pyproject.toml" ]; then
        YOSHI_DIR="$D"
        YOSHI_FOUND=true
        break
    fi
done

# Also check for gnosis_particle_bot (alternate name)
if ! $YOSHI_FOUND; then
    for D in /root/gnosis_particle_bot /home/root/gnosis_particle_bot "$HOME/gnosis_particle_bot"; do
        if [ -d "$D" ] && [ -f "$D/pyproject.toml" ]; then
            YOSHI_DIR="$D"
            YOSHI_FOUND=true
            break
        fi
    done
fi

if $YOSHI_FOUND; then
    ok "Found at $YOSHI_DIR"
else
    YOSHI_DIR="/root/Yoshi-Bot"
    warn "Not found. Will use $YOSHI_DIR"
fi

# Trading Core
TRADING_CORE_UP=false
if curl -sf http://127.0.0.1:8000/health 2>/dev/null | grep -q "healthy"; then
    TRADING_CORE_UP=true
    ok "Trading Core: RUNNING on :8000"
else
    warn "Trading Core not on :8000 (start Yoshi first)"
fi

# ================================================================
# 4. ClawdBot repo
# ================================================================
step 4 "ClawdBot repo"

# Find or set ClawdBot dir (same parent as Yoshi)
PARENT_DIR=$(dirname "$YOSHI_DIR")
CLAWDBOT_DIR="$PARENT_DIR/ClawdBot-V1"

if [ -d "$CLAWDBOT_DIR" ]; then
    cd "$CLAWDBOT_DIR"
    git fetch origin --quiet 2>/dev/null
    git checkout "$CLAWDBOT_BRANCH" --quiet 2>/dev/null || git checkout -b "$CLAWDBOT_BRANCH" "origin/$CLAWDBOT_BRANCH" --quiet 2>/dev/null || true
    git reset --hard "origin/$CLAWDBOT_BRANCH" --quiet 2>/dev/null || true
    ok "Updated $CLAWDBOT_DIR (branch: $CLAWDBOT_BRANCH)"
else
    git clone --quiet "$CLAWDBOT_REPO" "$CLAWDBOT_DIR" 2>/dev/null
    cd "$CLAWDBOT_DIR"
    git checkout "$CLAWDBOT_BRANCH" --quiet 2>/dev/null
    ok "Cloned $CLAWDBOT_DIR"
fi

# Verify we're on the right branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ "$CURRENT_BRANCH" != "$CLAWDBOT_BRANCH" ]; then
    warn "On branch $CURRENT_BRANCH, switching..."
    git checkout "$CLAWDBOT_BRANCH" --force --quiet 2>/dev/null || true
fi

# Install node deps
if [ -f "$CLAWDBOT_DIR/package.json" ] && [ ! -d "$CLAWDBOT_DIR/node_modules" ]; then
    cd "$CLAWDBOT_DIR"
    npm install --silent >/dev/null 2>&1
    ok "npm install done"
fi

# Install ML deps from requirements-ml.txt if available
if [ -f "$CLAWDBOT_DIR/requirements-ml.txt" ]; then
    pip3 install -r "$CLAWDBOT_DIR/requirements-ml.txt" --quiet --break-system-packages 2>/dev/null \
        || pip3 install -r "$CLAWDBOT_DIR/requirements-ml.txt" --quiet 2>/dev/null || true
    ok "ML dependencies installed"
fi

# Validate forecaster imports (ultimate-fix)
cd "$CLAWDBOT_DIR"
python3 -c "
import sys; sys.path.insert(0, '.')
from scripts.forecaster.engine import Forecaster
fc = Forecaster()
print(f'  Forecaster: {len(fc._modules)} modules ready')
" 2>/dev/null && ok "Forecaster modules validated" || warn "Forecaster import check skipped"

# ================================================================
# 5. Kalshi credentials
# ================================================================
step 5 "Kalshi API"

KALSHI_CONFIGURED=false
YOSHI_ENV="$YOSHI_DIR/.env"

# Ensure .env exists
if [ ! -f "$YOSHI_ENV" ] && [ -f "$YOSHI_DIR/.env.example" ]; then
    cp "$YOSHI_DIR/.env.example" "$YOSHI_ENV"
fi
touch "$YOSHI_ENV" 2>/dev/null

# Write Key ID if missing
EXISTING_KID=$(grep "^KALSHI_KEY_ID=" "$YOSHI_ENV" 2>/dev/null | head -1 | cut -d'=' -f2 | tr -d '"' | tr -d "'")
if [ -z "$EXISTING_KID" ] || [ "$EXISTING_KID" = "your_kalshi_key_id_here" ]; then
    sed -i '/^KALSHI_KEY_ID=/d' "$YOSHI_ENV"
    echo "" >> "$YOSHI_ENV"
    echo "# Kalshi API (added by setup-all.sh)" >> "$YOSHI_ENV"
    echo "KALSHI_KEY_ID=$KALSHI_KEY_ID" >> "$YOSHI_ENV"
    ok "Key ID: ${KALSHI_KEY_ID:0:12}..."
else
    ok "Key ID: ${EXISTING_KID:0:12}..."
fi

# Check private key — auto-search first
EXISTING_PK=$(grep "^KALSHI_PRIVATE_KEY=" "$YOSHI_ENV" 2>/dev/null | head -1)

if [ -n "$EXISTING_PK" ] && ! echo "$EXISTING_PK" | grep -q "your_kalshi"; then
    ok "Private key in .env"
    KALSHI_CONFIGURED=true
elif [ -f "$KALSHI_KEY_FILE" ]; then
    ok "Private key at $KALSHI_KEY_FILE"
    sed -i '/^KALSHI_PRIVATE_KEY=/d' "$YOSHI_ENV"
    echo "KALSHI_PRIVATE_KEY=\"$(cat "$KALSHI_KEY_FILE")\"" >> "$YOSHI_ENV"
    KALSHI_CONFIGURED=true
else
    # Auto-search for PEM files on disk
    echo "  Searching for Kalshi private key..."
    FOUND_PEM=""
    for F in \
        "$HOME/.kalshi/private_key.pem" \
        "$YOSHI_DIR/kalshi_key.pem" \
        "$YOSHI_DIR/private_key.pem" \
        "$HOME/kalshi_key.pem" \
        "$HOME/private_key.pem" \
        "$HOME/.ssh/kalshi_key.pem" \
        /root/.kalshi/private_key.pem \
        /root/kalshi_key.pem \
        /home/root/.kalshi/private_key.pem; do
        if [ -f "$F" ]; then
            FOUND_PEM="$F"
            break
        fi
    done

    # Also search broadly
    if [ -z "$FOUND_PEM" ]; then
        FOUND_PEM=$(find /root /home/root "$HOME" -maxdepth 3 -name "*.pem" -exec grep -l "RSA PRIVATE KEY" {} \; 2>/dev/null | head -1)
    fi

    if [ -n "$FOUND_PEM" ]; then
        ok "Found key: $FOUND_PEM"
        mkdir -p "$KALSHI_KEY_DIR" && chmod 700 "$KALSHI_KEY_DIR"
        cp "$FOUND_PEM" "$KALSHI_KEY_FILE"
        chmod 600 "$KALSHI_KEY_FILE"
        sed -i '/^KALSHI_PRIVATE_KEY=/d' "$YOSHI_ENV"
        echo "KALSHI_PRIVATE_KEY=\"$(cat "$KALSHI_KEY_FILE")\"" >> "$YOSHI_ENV"
        ok "Private key configured from $FOUND_PEM"
        KALSHI_CONFIGURED=true
    else
        echo ""
        echo -e "  ${BOLD}No Kalshi private key found on disk.${NC}"
        echo ""
        echo "  Options:"
        echo "    1) Type path to PEM file"
        echo "    2) Type 'paste' to paste key"
        echo "    3) Press Enter to skip"
        echo ""
        read -p "  > " KEY_CHOICE </dev/tty 2>/dev/null || KEY_CHOICE=""

        if [ -n "$KEY_CHOICE" ] && [ "$KEY_CHOICE" != "paste" ] && [ -f "$KEY_CHOICE" ]; then
            mkdir -p "$KALSHI_KEY_DIR" && chmod 700 "$KALSHI_KEY_DIR"
            cp "$KEY_CHOICE" "$KALSHI_KEY_FILE"
            chmod 600 "$KALSHI_KEY_FILE"
            sed -i '/^KALSHI_PRIVATE_KEY=/d' "$YOSHI_ENV"
            echo "KALSHI_PRIVATE_KEY=\"$(cat "$KALSHI_KEY_FILE")\"" >> "$YOSHI_ENV"
            ok "Key loaded from $KEY_CHOICE"
            KALSHI_CONFIGURED=true
        elif [ "$KEY_CHOICE" = "paste" ]; then
            echo "  Paste key (end with Ctrl+D):"
            mkdir -p "$KALSHI_KEY_DIR" && chmod 700 "$KALSHI_KEY_DIR"
            cat > "$KALSHI_KEY_FILE" </dev/tty
            chmod 600 "$KALSHI_KEY_FILE"
            if grep -q "RSA PRIVATE KEY" "$KALSHI_KEY_FILE" 2>/dev/null; then
                sed -i '/^KALSHI_PRIVATE_KEY=/d' "$YOSHI_ENV"
                echo "KALSHI_PRIVATE_KEY=\"$(cat "$KALSHI_KEY_FILE")\"" >> "$YOSHI_ENV"
                ok "Key saved"
                KALSHI_CONFIGURED=true
            else
                warn "Not a valid RSA key"
            fi
        else
            warn "Skipped — run later:"
            echo "    bash $CLAWDBOT_DIR/scripts/setup-kalshi.sh"
        fi
    fi
fi

chmod 600 "$YOSHI_ENV" 2>/dev/null

# Verify Kalshi using the standalone KalshiClient module
if $KALSHI_CONFIGURED; then
    # Source env so KALSHI_KEY_ID and KALSHI_PRIVATE_KEY are available
    set -a; source "$YOSHI_ENV" 2>/dev/null; set +a
    # Use the shared kalshi_client module for verification
    python3 -c "
import sys, os
sys.path.insert(0, '$CLAWDBOT_DIR/scripts')

try:
    from kalshi_client import KalshiClient
    
    # Instantiate client and test connection
    client = KalshiClient()
    status = client.get_exchange_status()
    
    if status:
        print(f'  \\033[0;32m\\u2713 Kalshi CONNECTED (exchange={status.get(\"exchange_active\")}, trading={status.get(\"trading_active\")})\\033[0m')
    else:
        print('  \\033[1;33m\\u26a0 Kalshi API call failed\\033[0m')
except Exception as e:
    print(f'  \\033[0;31m\\u2717 Kalshi verify error: {e}\\033[0m')
" 2>/dev/null || warn "Could not verify Kalshi (missing cryptography? Run: pip3 install cryptography)"
fi

# ================================================================
# 6. Collect API keys
# ================================================================
step 6 "API keys"

# Source everything available
TELEGRAM_BOT_TOKEN=""
OPENAI_API_KEY=""

# Pull from Yoshi .env
[ -f "$YOSHI_ENV" ] && source "$YOSHI_ENV" 2>/dev/null

# Pull from ClawdBot .env
[ -f "$CLAWDBOT_DIR/.env" ] && source "$CLAWDBOT_DIR/.env" 2>/dev/null

# Pull from any .env in common locations
for F in /root/.env /home/root/.env "$HOME/.env"; do
    [ -f "$F" ] && source "$F" 2>/dev/null
done

# Check Telegram
if [ -z "$TELEGRAM_BOT_TOKEN" ] || echo "$TELEGRAM_BOT_TOKEN" | grep -q "your_"; then
    echo ""
    echo -e "  ${BOLD}Need Telegram Bot Token${NC}"
    read -p "  Token: " TELEGRAM_BOT_TOKEN </dev/tty 2>/dev/null || true
    if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
        warn "No Telegram token — bot won't connect"
    fi
else
    ok "Telegram: ...${TELEGRAM_BOT_TOKEN: -8}"
fi

# Check OpenAI
if [ -z "$OPENAI_API_KEY" ] || echo "$OPENAI_API_KEY" | grep -q "your_"; then
    echo ""
    echo -e "  ${BOLD}Need OpenAI API Key${NC}"
    read -p "  Key: " OPENAI_API_KEY </dev/tty 2>/dev/null || true
    if [ -z "$OPENAI_API_KEY" ]; then
        warn "No OpenAI key — ClawdBot can't reason"
    fi
else
    ok "OpenAI: ...${OPENAI_API_KEY: -6}"
fi

# Write ClawdBot .env
cat > "$CLAWDBOT_DIR/.env" << EOF
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
OPENAI_API_KEY=$OPENAI_API_KEY
KALSHI_KEY_ID=$KALSHI_KEY_ID
EOF
chmod 600 "$CLAWDBOT_DIR/.env"
ok "Wrote $CLAWDBOT_DIR/.env"

# ================================================================
# 7. Moltbot config
# ================================================================
step 7 "Moltbot config"

mkdir -p ~/.clawdbot

# Write ~/.clawdbot/.env for moltbot env var fallback
cat > ~/.clawdbot/.env << ENVEOF
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
OPENAI_API_KEY=$OPENAI_API_KEY
ENVEOF
chmod 600 ~/.clawdbot/.env

# Remove ALL leftover configs and state dirs (moltbot uses both paths)
rm -f ~/.clawdbot/moltbot.json ~/.clawdbot/moltbot.json.bak 2>/dev/null
rm -f ~/.moltbot/moltbot.json ~/.moltbot/moltbot.json.bak 2>/dev/null
rm -rf /root/.moltbot 2>/dev/null  # remove old state dir that triggers doctor warnings

# Generate gateway auth token
GATEWAY_TOKEN=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))")

# Build allowFrom list safely with Python
python3 << 'PYEOF' > ~/.clawdbot/moltbot.json
import json, os

gateway_token = os.environ.get("GATEWAY_TOKEN", "")
telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
telegram_allowlist = os.environ.get("TELEGRAM_ALLOWLIST", "").strip()
clawdbot_dir = os.environ.get("CLAWDBOT_DIR", "")

# Parse allowlist safely
if telegram_allowlist:
    allow_from = [x.strip() for x in telegram_allowlist.split(",") if x.strip()]
else:
    allow_from = []

config = {
    "agents": {
        "defaults": {
            "workspace": "~/clawd",
            "model": {
                "primary": "openai/gpt-4o"
            },
            "thinkingDefault": "low"
        },
        "list": [
            {
                "id": "main",
                "default": True,
                "identity": {
                    "name": "ClawdBot",
                    "theme": "crypto trading assistant",
                    "emoji": "🤖"
                }
            }
        ]
    },
    "gateway": {
        "mode": "local",
        "port": 18789,
        "bind": "loopback",
        "auth": {
            "mode": "token",
            "token": gateway_token
        }
    },
    "channels": {
        "telegram": {
            "enabled": True,
            "botToken": telegram_token,
            "dmPolicy": "open",
            "allowFrom": allow_from
        }
    },
    "skills": {
        "load": {
            "extraDirs": [clawdbot_dir + "/skills"]
        }
    }
}

json.dump(config, open(os.path.expanduser("~/.clawdbot/moltbot.json"), "w"), indent=2)
PYEOF

export GATEWAY_TOKEN TELEGRAM_BOT_TOKEN TELEGRAM_ALLOWLIST CLAWDBOT_DIR

chmod 700 ~/.clawdbot
ok "~/.clawdbot/moltbot.json (token: ${GATEWAY_TOKEN:0:8}...)"

# ================================================================
# 8. Systemd services
# ================================================================
step 8 "Services"

SCANNER_LOG="$YOSHI_DIR/logs/scanner.log"
mkdir -p "$YOSHI_DIR/logs" 2>/dev/null || true

# Find moltbot binary path
MOLTBOT_BIN=$(which moltbot 2>/dev/null || echo "/usr/bin/moltbot")

# Validate JSON with python (NOT moltbot doctor — it overwrites the config!)
echo "  Validating JSON..."
if python3 -c "import json; json.load(open('$HOME/.clawdbot/moltbot.json'))" 2>/dev/null; then
    ok "moltbot.json: valid JSON"
else
    fail "moltbot.json: invalid JSON!"
    cat ~/.clawdbot/moltbot.json
fi

cat > /etc/systemd/system/clawdbot.service << SVCEOF
[Unit]
Description=ClawdBot Telegram AI Trading Assistant (Ultimate-Fix)
After=network.target
Wants=kalshi-edge-scanner.service yoshi-bridge.service

[Service]
Type=simple
User=root
WorkingDirectory=$CLAWDBOT_DIR
EnvironmentFile=$CLAWDBOT_DIR/.env
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=NODE_ENV=production
Environment=CLAWDBOT_GATEWAY_TOKEN=$GATEWAY_TOKEN
ExecStart=$MOLTBOT_BIN gateway --port 18789
Environment=PYTHONPATH=$CLAWDBOT_DIR

# Kill anything holding the port before start
ExecStartPre=-/usr/bin/fuser -k -9 18789/tcp
ExecStartPre=/bin/sleep 1

# Rebuild config from env vars
ExecStartPre=-/usr/bin/python3 $CLAWDBOT_DIR/scripts/rebuild-config.py --quiet

# Ultimate-fix: Run diagnostics + auto-fix on start (quick boot mode)
ExecStartPre=-/usr/bin/python3 -c "import sys; sys.path.insert(0,'$CLAWDBOT_DIR'); from scripts.forecaster.diagnose import full_diagnostics_and_fix; full_diagnostics_and_fix(bars=500, forecasts=20, output_path='$CLAWDBOT_DIR/data/diagnostics_report.json')"

ExecStart=$MOLTBOT_BIN gateway --port 18789 --token $GATEWAY_TOKEN
Restart=always
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=clawdbot

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/root /home/root /tmp
ProtectHome=no

[Install]
WantedBy=multi-user.target
SVCEOF
ok "clawdbot.service (with auto-fix on boot)"

cat > /etc/systemd/system/yoshi-bridge.service << SVCEOF
[Unit]
Description=Yoshi-Bridge Scanner to Trading Core
After=network.target

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
SVCEOF
ok "yoshi-bridge.service"

# Kalshi Edge Scanner service (continuous best-pick finder)
mkdir -p "$CLAWDBOT_DIR/data" "$CLAWDBOT_DIR/logs" 2>/dev/null || true

cat > /etc/systemd/system/kalshi-edge-scanner.service << SVCEOF
[Unit]
Description=Kalshi Edge Scanner — Continuous Best-Pick Finder (Ultimate-Fix)
After=network.target
Wants=yoshi-bridge.service

[Service]
Type=simple
User=root
WorkingDirectory=$CLAWDBOT_DIR
EnvironmentFile=$YOSHI_DIR/.env
EnvironmentFile=-$CLAWDBOT_DIR/.env
Environment=TRADING_CORE_URL=http://127.0.0.1:8000
Environment=PYTHONPATH=$CLAWDBOT_DIR

# Fix PEM keys before start
ExecStartPre=-/usr/bin/python3 -c "import sys; sys.path.insert(0,'$CLAWDBOT_DIR'); from scripts.lib.pem_utils import fix_all_pem_files; fix_all_pem_files()"

ExecStart=/usr/bin/python3 $CLAWDBOT_DIR/scripts/kalshi-edge-scanner.py --loop --interval 120 --top 2 --min-edge 3.0 --propose
Restart=always
RestartSec=30
StartLimitIntervalSec=600
StartLimitBurst=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=kalshi-edge-scanner

[Install]
WantedBy=multi-user.target
SVCEOF
ok "kalshi-edge-scanner.service"

systemctl daemon-reload
systemctl enable clawdbot yoshi-bridge kalshi-edge-scanner >/dev/null 2>&1
ok "Enabled"

# ================================================================
# 9. Start
# ================================================================
step 9 "Starting"

# Kill stale gateway processes (leftover from previous runs)
systemctl stop clawdbot 2>/dev/null || true
pkill -9 -f "moltbot gateway" 2>/dev/null || true
pkill -9 -f "moltbot.*18789" 2>/dev/null || true
fuser -k 18789/tcp 2>/dev/null || true
sleep 1

systemctl restart yoshi-bridge 2>/dev/null || true
sleep 2
systemctl restart clawdbot 2>/dev/null || true
sleep 2
if $KALSHI_CONFIGURED; then
    systemctl restart kalshi-edge-scanner 2>/dev/null || true
fi
sleep 3

echo ""
CS=$(systemctl is-active clawdbot 2>/dev/null || echo "dead")
BS=$(systemctl is-active yoshi-bridge 2>/dev/null || echo "dead")

if [ "$CS" = "active" ]; then
    ok "ClawdBot: RUNNING"
else
    warn "ClawdBot: $CS"
    echo ""
    echo -e "  ${YELLOW}--- ClawdBot startup log ---${NC}"
    journalctl -u clawdbot -n 15 --no-pager 2>/dev/null || true
    echo -e "  ${YELLOW}--- end log ---${NC}"
    echo ""
    echo "  Try: moltbot doctor"
    echo "  Fix: moltbot doctor --fix"
fi
[ "$BS" = "active" ] && ok "Yoshi-Bridge: RUNNING" || warn "Yoshi-Bridge: $BS — check: journalctl -u yoshi-bridge -n 20"
$TRADING_CORE_UP && ok "Trading Core: RUNNING"

if $KALSHI_CONFIGURED; then
    ES=$(systemctl is-active kalshi-edge-scanner 2>/dev/null || echo "dead")
    [ "$ES" = "active" ] && ok "Edge Scanner: RUNNING (every 2min)" || warn "Edge Scanner: $ES"
    ok "Kalshi: CONFIGURED"
else
    warn "Kalshi: needs private key — edge scanner disabled"
fi

# ================================================================
header "DONE"

echo "Logs:"
echo "  journalctl -u clawdbot -f"
echo "  journalctl -u yoshi-bridge -f"
echo "  journalctl -u kalshi-edge-scanner -f"
echo ""
echo "Status:"
echo "  curl -s localhost:8000/status | python3 -m json.tool"
echo "  systemctl status clawdbot"
echo ""

if ! $KALSHI_CONFIGURED; then
    echo -e "${YELLOW}━━━ Kalshi private key still needed ━━━${NC}"
    echo "  bash $CLAWDBOT_DIR/scripts/setup-kalshi.sh"
    echo ""
fi

echo -e "${BOLD}Message your bot on Telegram:${NC}"
echo '  "Yoshi status"'
echo '  "Check Kalshi"'
echo '  "Show BTC markets"'
echo ""
echo -e "${BOLD}Ultimate-Fix Commands:${NC}"
echo "  python3 -m scripts.forecaster.diagnose --bars 2000 --auto-fix  # Full diagnostic"
echo "  python3 -m scripts.forecaster.engine --symbol BTCUSDT          # Forecast"
echo "  python3 scripts/monte-carlo/simulation.py --live               # Monte Carlo"
echo "  python3 scripts/fetch_coingecko_data.py --days 90 --onchain    # Multi-asset data"
echo "  python3 -m scripts.forecaster.evaluation --bars 1000           # Yoshi battery"
echo ""
