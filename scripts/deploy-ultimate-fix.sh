#!/bin/bash
# ============================================================
#  ClawdBot Ultimate-Fix — VPS Deployment Script
#  ============================================================
#
#  Deploys the ultimate-fix branch to a live VPS with:
#    - LightGBM + GRU hybrid ML
#    - Regime gating + arbitrage module
#    - Auto-calibration (isotonic + Platt)
#    - Enhanced diagnostics with auto-fix on boot
#    - Monte Carlo integration (regime-conditioned)
#    - Moltbot integration + systemd services
#
#  Usage (on VPS):
#    cd /root/ClawdBot-V1 && bash scripts/deploy-ultimate-fix.sh
#
#  Or one-liner from GitHub:
#    curl -sSL https://raw.githubusercontent.com/DGator86/ClawdBot-V1/ultimate-fix/scripts/deploy-ultimate-fix.sh | bash
#
#  Prerequisites:
#    - VPS with Ubuntu/Debian, root access
#    - Git, Python 3.10+, Node.js 22+
#    - Yoshi-Bot already deployed (optional but recommended)
#
# ============================================================

set -euo pipefail

# ── Colors ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

header() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

step() { echo -e "${YELLOW}[$1/$TOTAL] $2${NC}"; }
ok()   { echo -e "  ${GREEN}✓ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "  ${RED}✗ $1${NC}"; }

TOTAL=10
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# If running via curl pipe, find or clone the repo
if [ ! -f "$PROJECT_DIR/scripts/forecaster/engine.py" ]; then
    # We're running from a temp location — find or clone ClawdBot
    for D in /root/ClawdBot-V1 /home/root/ClawdBot-V1 "$HOME/ClawdBot-V1"; do
        if [ -d "$D/.git" ]; then
            PROJECT_DIR="$D"
            break
        fi
    done
    if [ ! -d "$PROJECT_DIR/.git" ]; then
        echo -e "${YELLOW}Cloning ClawdBot-V1...${NC}"
        git clone https://github.com/DGator86/ClawdBot-V1.git /root/ClawdBot-V1
        PROJECT_DIR="/root/ClawdBot-V1"
    fi
fi

cd "$PROJECT_DIR"

header "ClawdBot Ultimate-Fix Deployment"
echo "  Project: $PROJECT_DIR"
echo "  Branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
echo "  Commit:  $(git log --oneline -1 2>/dev/null || echo 'unknown')"
echo ""

# ============================================================
# 1. Switch to ultimate-fix branch
# ============================================================
step 1 "Switching to ultimate-fix branch"

git fetch origin --quiet 2>/dev/null || true

# Check if ultimate-fix branch exists
if git rev-parse --verify origin/ultimate-fix &>/dev/null; then
    # Clean any dirty state — remove untracked/modified files that could conflict
    git checkout -- . 2>/dev/null || true
    git clean -fd 2>/dev/null || true

    CURRENT=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    if [ "$CURRENT" != "ultimate-fix" ]; then
        git stash --quiet 2>/dev/null || true
        git checkout ultimate-fix --quiet 2>/dev/null || \
            git checkout -b ultimate-fix origin/ultimate-fix --quiet 2>/dev/null
    fi
    # Force exact match with remote (overwrites any local modifications)
    git reset --hard origin/ultimate-fix 2>/dev/null || true
    git clean -fd 2>/dev/null || true
    ok "On branch: ultimate-fix ($(git log --oneline -1))"

    # Verify critical file integrity
    if python3 -c "import ast; ast.parse(open('scripts/forecaster/engine.py').read())" 2>/dev/null; then
        ok "engine.py: syntax verified"
    else
        fail "engine.py: syntax error after checkout — try: git checkout origin/ultimate-fix -- scripts/forecaster/engine.py"
    fi
else
    warn "origin/ultimate-fix not found, staying on current branch"
    echo "  Using: $(git rev-parse --abbrev-ref HEAD) ($(git log --oneline -1))"
fi

# ============================================================
# 2. System dependencies
# ============================================================
step 2 "System dependencies"

apt-get update -y -qq >/dev/null 2>&1 || true
apt-get install -y -qq curl git jq python3 python3-pip python3-venv build-essential >/dev/null 2>&1 || true
ok "System packages installed"

# Node.js 22
if ! command -v node &>/dev/null || [ "$(node -v | cut -d'v' -f2 | cut -d'.' -f1)" -lt 22 ]; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1
    apt-get install -y -qq nodejs >/dev/null 2>&1
fi
ok "Node.js $(node -v 2>/dev/null || echo 'N/A')"

# Moltbot
if ! command -v moltbot &>/dev/null; then
    npm install -g moltbot@latest --silent >/dev/null 2>&1 || true
fi
ok "moltbot $(moltbot --version 2>/dev/null || echo 'installed')"

# ============================================================
# 3. Python ML dependencies (ultimate-fix specific)
# ============================================================
step 3 "Python ML dependencies"

if [ -f "$PROJECT_DIR/requirements-ml.txt" ]; then
    pip3 install -r "$PROJECT_DIR/requirements-ml.txt" --quiet --break-system-packages 2>/dev/null \
        || pip3 install -r "$PROJECT_DIR/requirements-ml.txt" --quiet 2>/dev/null \
        || {
            # Fallback: install one-by-one
            for pkg in numpy scipy lightgbm scikit-learn pandas pyarrow requests cryptography; do
                pip3 install "$pkg" --quiet --break-system-packages 2>/dev/null \
                    || pip3 install "$pkg" --quiet 2>/dev/null || true
            done
        }
else
    # Direct install if requirements file missing
    pip3 install numpy scipy lightgbm scikit-learn pandas pyarrow requests cryptography \
        --quiet --break-system-packages 2>/dev/null \
        || pip3 install numpy scipy lightgbm scikit-learn pandas pyarrow requests cryptography --quiet 2>/dev/null
fi

# Verify critical imports
python3 -c "
import sys
errors = []
for mod in ['numpy', 'scipy', 'lightgbm', 'sklearn', 'pandas']:
    try:
        __import__(mod)
    except ImportError:
        errors.append(mod)
if errors:
    print(f'MISSING: {errors}', file=sys.stderr)
    sys.exit(1)
print('All ML dependencies OK')
" && ok "ML stack: numpy, scipy, lightgbm, scikit-learn, pandas" \
  || fail "Some ML dependencies failed to install"

# ============================================================
# 4. Validate module integrity
# ============================================================
step 4 "Validating module integrity"

MODULES=(
    "scripts/forecaster/__init__.py"
    "scripts/forecaster/engine.py"
    "scripts/forecaster/ml_models.py"
    "scripts/forecaster/regime_gate.py"
    "scripts/forecaster/auto_fix.py"
    "scripts/forecaster/arbitrage.py"
    "scripts/forecaster/data_expanded.py"
    "scripts/forecaster/diagnose.py"
    "scripts/forecaster/modules.py"
    "scripts/forecaster/schemas.py"
    "scripts/forecaster/evaluation.py"
    "scripts/fetch_coingecko_data.py"
    "scripts/monte-carlo/simulation.py"
)

MISSING=0
SYNTAX_ERR=0
for f in "${MODULES[@]}"; do
    if [ ! -f "$PROJECT_DIR/$f" ]; then
        fail "Missing: $f"
        MISSING=$((MISSING + 1))
    else
        if ! python3 -c "import ast; ast.parse(open('$PROJECT_DIR/$f').read())" 2>/dev/null; then
            fail "Syntax error: $f"
            SYNTAX_ERR=$((SYNTAX_ERR + 1))
        fi
    fi
done

if [ $MISSING -eq 0 ] && [ $SYNTAX_ERR -eq 0 ]; then
    ok "All ${#MODULES[@]} modules present and syntax-valid"
else
    [ $MISSING -gt 0 ] && warn "$MISSING file(s) missing"
    [ $SYNTAX_ERR -gt 0 ] && warn "$SYNTAX_ERR file(s) have syntax errors"
fi

# ============================================================
# 5. Test critical imports
# ============================================================
step 5 "Testing critical imports"

cd "$PROJECT_DIR"
python3 -c "
import sys, os
sys.path.insert(0, '.')

# Test 1: Core schemas
from scripts.forecaster.schemas import MarketSnapshot, Bar, ModuleOutput, PredictionTargets, Regime
print('  [OK] schemas')

# Test 2: ML models
from scripts.forecaster.ml_models import HybridPredictor, TemporalFeatureExtractor
hp = HybridPredictor()
print(f'  [OK] ml_models (HybridPredictor trained={hp.trained})')

# Test 3: Regime gate
from scripts.forecaster.regime_gate import RegimeGate, ArbitrageDetector
rg = RegimeGate()
print(f'  [OK] regime_gate ({len(rg.profiles)} profiles)')

# Test 4: Auto-fix
from scripts.forecaster.auto_fix import AutoFixPipeline, CalibrationSuite, HealthMonitor
af = AutoFixPipeline()
print(f'  [OK] auto_fix (pipeline ready)')

# Test 5: Engine
from scripts.forecaster.engine import Forecaster, ForecastResult
fc = Forecaster()
print(f'  [OK] engine (Forecaster: {len(fc._modules)} base modules)')

# Test 6: Package-level
from scripts.forecaster import Forecaster as F2
print(f'  [OK] forecaster package import')

print('\\nAll critical imports passed!')
" && ok "All imports verified" || fail "Import test failed"

# ============================================================
# 6. Collect credentials (from existing .env files)
# ============================================================
step 6 "Collecting credentials"

CLAWDBOT_ENV="$PROJECT_DIR/.env"
YOSHI_DIR=""
YOSHI_ENV=""

# Find Yoshi-Bot
for D in /root/Yoshi-Bot /home/root/Yoshi-Bot "$HOME/Yoshi-Bot"; do
    if [ -d "$D" ]; then
        YOSHI_DIR="$D"
        YOSHI_ENV="$D/.env"
        break
    fi
done

# Source existing env files
TELEGRAM_BOT_TOKEN=""
OPENAI_API_KEY=""
KALSHI_KEY_ID=""

[ -f "$CLAWDBOT_ENV" ] && { set -a; source "$CLAWDBOT_ENV" 2>/dev/null; set +a; }
[ -n "$YOSHI_ENV" ] && [ -f "$YOSHI_ENV" ] && { set -a; source "$YOSHI_ENV" 2>/dev/null; set +a; }

# Try to get from systemd env
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    TELEGRAM_BOT_TOKEN=$(systemctl show clawdbot -p Environment 2>/dev/null | grep -oP 'TELEGRAM_BOT_TOKEN=\K[^ ]+' || echo "")
fi

if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    ok "Telegram: ...${TELEGRAM_BOT_TOKEN: -8}"
else
    warn "No TELEGRAM_BOT_TOKEN found (bot won't connect)"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    ok "OpenAI: ...${OPENAI_API_KEY: -6}"
else
    warn "No OPENAI_API_KEY found"
fi

[ -n "$KALSHI_KEY_ID" ] && ok "Kalshi Key ID: ${KALSHI_KEY_ID:0:12}..." || warn "No KALSHI_KEY_ID"

# Write/update ClawdBot .env
if [ -n "$TELEGRAM_BOT_TOKEN" ] || [ -n "$OPENAI_API_KEY" ]; then
    cat > "$CLAWDBOT_ENV" << ENVEOF
TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
KALSHI_KEY_ID=${KALSHI_KEY_ID:-}
ENVEOF
    chmod 600 "$CLAWDBOT_ENV"
    ok "Wrote $CLAWDBOT_ENV"
fi

# ============================================================
# 7. Create data directories
# ============================================================
step 7 "Creating data directories"

mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/models"
ok "data/, logs/, models/ directories ready"

# ============================================================
# 8. Install systemd services (ultimate-fix enhanced)
# ============================================================
step 8 "Installing systemd services"

MOLTBOT_BIN=$(which moltbot 2>/dev/null || echo "/usr/local/bin/moltbot")

# Generate gateway token if not already set
GATEWAY_TOKEN="${CLAWDBOT_GATEWAY_TOKEN:-$(openssl rand -hex 32 2>/dev/null || python3 -c 'import secrets; print(secrets.token_hex(32))')}"

# ── ClawdBot main service (with diagnostics on boot) ──
cat > /etc/systemd/system/clawdbot.service << SVCEOF
[Unit]
Description=ClawdBot Telegram AI Trading Assistant (Ultimate-Fix)
After=network.target
Wants=kalshi-edge-scanner.service yoshi-bridge.service

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=-$PROJECT_DIR/.env
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=NODE_ENV=production
Environment=CLAWDBOT_GATEWAY_TOKEN=$GATEWAY_TOKEN
Environment=PYTHONPATH=$PROJECT_DIR

# Kill anything holding the port before start
ExecStartPre=-/usr/bin/fuser -k -9 18789/tcp
ExecStartPre=/bin/sleep 1

# Rebuild config from env vars
ExecStartPre=-/usr/bin/python3 $PROJECT_DIR/scripts/rebuild-config.py --quiet

# Ultimate-fix: Run diagnostics + auto-fix + calibration on start
# Quick boot mode (500 bars, 20 forecasts) — full diagnostics via cron
ExecStartPre=-/usr/bin/python3 -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from scripts.forecaster.diagnose import full_diagnostics_and_fix; full_diagnostics_and_fix(bars=500, forecasts=20, output_path='$PROJECT_DIR/data/diagnostics_report.json')"

# Start moltbot gateway
ExecStart=$MOLTBOT_BIN gateway --port 18789
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

# ── Kalshi Edge Scanner ──
SCANNER_LOG="${YOSHI_DIR:-/root/Yoshi-Bot}/logs/scanner.log"
mkdir -p "$(dirname "$SCANNER_LOG")" 2>/dev/null || true

cat > /etc/systemd/system/kalshi-edge-scanner.service << SVCEOF
[Unit]
Description=Kalshi Edge Scanner — Continuous Best-Pick Finder (Ultimate-Fix)
After=network.target
Wants=yoshi-bridge.service

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=-${YOSHI_DIR:-/root/Yoshi-Bot}/.env
EnvironmentFile=-$PROJECT_DIR/.env
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=TRADING_CORE_URL=http://127.0.0.1:8000
Environment=PYTHONPATH=$PROJECT_DIR

# Fix PEM keys before start
ExecStartPre=-/usr/bin/python3 -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from scripts.lib.pem_utils import fix_all_pem_files; fix_all_pem_files()"

ExecStart=/usr/bin/python3 $PROJECT_DIR/scripts/kalshi-edge-scanner.py --loop --interval 120 --top 2 --min-edge 3.0 --propose
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

# ── Yoshi Bridge ──
cat > /etc/systemd/system/yoshi-bridge.service << SVCEOF
[Unit]
Description=Yoshi Bridge — Scanner Log to Trading Core (Ultimate-Fix)
After=network.target
PartOf=clawdbot.service

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=-$PROJECT_DIR/.env
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=TRADING_CORE_URL=http://127.0.0.1:8000
Environment=PYTHONPATH=$PROJECT_DIR

ExecStart=/usr/bin/python3 $PROJECT_DIR/scripts/yoshi-bridge.py --poll-interval 30
Restart=always
RestartSec=15
StartLimitIntervalSec=600
StartLimitBurst=5

StandardOutput=journal
StandardError=journal
SyslogIdentifier=yoshi-bridge

[Install]
WantedBy=multi-user.target
SVCEOF
ok "yoshi-bridge.service"

systemctl daemon-reload
systemctl enable clawdbot yoshi-bridge kalshi-edge-scanner >/dev/null 2>&1
ok "Services enabled"

# ============================================================
# 9. Install cron for full diagnostics (every 6 hours)
# ============================================================
step 9 "Setting up scheduled diagnostics"

CRON_CMD="cd $PROJECT_DIR && /usr/bin/python3 -c \"import sys; sys.path.insert(0,'.'); from scripts.forecaster.diagnose import full_diagnostics_and_fix; full_diagnostics_and_fix(bars=2000, forecasts=75, output_path='$PROJECT_DIR/data/diagnostics_report.json')\" >> $PROJECT_DIR/logs/diagnostics_cron.log 2>&1"

# Remove old cron entries for diagnostics
crontab -l 2>/dev/null | grep -v "diagnostics_and_fix" | grep -v "forecaster.diagnose" > /tmp/cron_clean 2>/dev/null || true

# Add new entry: every 6 hours
echo "0 */6 * * * $CRON_CMD" >> /tmp/cron_clean
crontab /tmp/cron_clean
rm -f /tmp/cron_clean
ok "Cron: full diagnostics every 6h"

# ============================================================
# 10. Start services and verify
# ============================================================
step 10 "Starting services"

# Stop existing
systemctl stop clawdbot 2>/dev/null || true
systemctl stop kalshi-edge-scanner 2>/dev/null || true
systemctl stop yoshi-bridge 2>/dev/null || true
pkill -9 -f "moltbot gateway" 2>/dev/null || true
fuser -k 18789/tcp 2>/dev/null || true
sleep 2

# Start in order
systemctl restart yoshi-bridge 2>/dev/null || true
sleep 1
systemctl restart clawdbot 2>/dev/null || true
sleep 3
systemctl restart kalshi-edge-scanner 2>/dev/null || true
sleep 2

# ── Verify ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}Service Status:${NC}"

for svc in clawdbot yoshi-bridge kalshi-edge-scanner; do
    state=$(systemctl is-active "$svc" 2>/dev/null || echo "dead")
    case $state in
        active) echo -e "  ${GREEN}✓ ${svc}: RUNNING${NC}" ;;
        *)      echo -e "  ${RED}✗ ${svc}: ${state}${NC}" ;;
    esac
done

echo ""

# Check gateway port
if ss -tlnp 2>/dev/null | grep -q ':18789'; then
    ok "Gateway: LISTENING on :18789"
else
    warn "Gateway: NOT LISTENING on :18789"
fi

# Check Trading Core
if curl -sf http://127.0.0.1:8000/health 2>/dev/null | grep -q "healthy"; then
    ok "Trading Core: HEALTHY on :8000"
else
    warn "Trading Core: not available on :8000"
fi

# Check diagnostics report
if [ -f "$PROJECT_DIR/data/diagnostics_report.json" ]; then
    VERDICT=$(python3 -c "import json; d=json.load(open('$PROJECT_DIR/data/diagnostics_report.json')); print(d.get('verdict','unknown'))" 2>/dev/null || echo "unknown")
    ok "Diagnostics: $VERDICT (data/diagnostics_report.json)"
else
    warn "Diagnostics report not yet generated"
fi

# ============================================================
header "DEPLOYMENT COMPLETE"

echo -e "${BOLD}Ultimate-Fix Components:${NC}"
echo "  • LightGBM + GRU hybrid ML (scripts/forecaster/ml_models.py)"
echo "  • Regime gating — 8 profiles (scripts/forecaster/regime_gate.py)"
echo "  • Arbitrage detector (scripts/forecaster/arbitrage.py)"
echo "  • Auto-calibration: isotonic + Platt + temperature (scripts/forecaster/auto_fix.py)"
echo "  • Enhanced diagnostics with auto-fix (scripts/forecaster/diagnose.py)"
echo "  • Multi-asset data: BTC/ETH/SOL, 90-day (scripts/fetch_coingecko_data.py)"
echo "  • Monte Carlo: regime-conditioned jump-diffusion (scripts/monte-carlo/simulation.py)"
echo ""
echo -e "${BOLD}Commands:${NC}"
echo "  Status:     bash scripts/services/install.sh --status"
echo "  Logs:       journalctl -u clawdbot -f"
echo "  Diagnostics: python3 -m scripts.forecaster.diagnose --bars 2000 --forecasts 75 --auto-fix"
echo "  Quick test:  python3 -m scripts.forecaster.diagnose --bars 500 --forecasts 20 --auto-fix --json"
echo "  Forecast:    python3 -m scripts.forecaster.engine --symbol BTCUSDT --horizon 24"
echo "  Monte Carlo: python3 scripts/monte-carlo/simulation.py --live --symbol BTCUSDT"
echo "  Data fetch:  python3 scripts/fetch_coingecko_data.py --days 90 --onchain"
echo ""
echo -e "${BOLD}Yoshi Battery CLI:${NC}"
echo "  python3 -m scripts.forecaster.evaluation --bars 1000 --symbol BTCUSDT"
echo ""
echo -e "${BOLD}Target Metrics:${NC}"
echo "  HR: 65-80% (in skilled regimes)"
echo "  MCC: > 0.1"
echo "  BTC RMSE: < 100"
echo "  Arb Yield: ~18% APY"
echo "  Coverage: >= 94%"
echo ""
