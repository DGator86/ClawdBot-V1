#!/bin/bash
# ============================================================
#  ClawdBot Service Installer
#  Installs and enables all systemd services for ClawdBot
#
#  Usage:
#    cd /root/ClawdBot-V1 && bash scripts/services/install.sh
#    bash scripts/services/install.sh --start     # install + start
#    bash scripts/services/install.sh --status     # check status
#    bash scripts/services/install.sh --uninstall  # remove services
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
SERVICES=(clawdbot kalshi-edge-scanner yoshi-bridge)

# ── Parse args ────────────────────────────────────────────
ACTION="install"
START_AFTER=false
for arg in "$@"; do
    case $arg in
        --start)   START_AFTER=true ;;
        --status)  ACTION="status" ;;
        --uninstall) ACTION="uninstall" ;;
        --restart) ACTION="restart" ;;
        --logs)    ACTION="logs" ;;
        --help|-h)
            echo "Usage: $0 [--start] [--status] [--uninstall] [--restart] [--logs]"
            exit 0 ;;
    esac
done

# ── Status ────────────────────────────────────────────────
if [ "$ACTION" = "status" ]; then
    echo -e "${CYAN}=== ClawdBot Service Status ===${NC}\n"
    for svc in "${SERVICES[@]}"; do
        state=$(systemctl is-active "$svc" 2>/dev/null || echo "not-found")
        enabled=$(systemctl is-enabled "$svc" 2>/dev/null || echo "not-found")
        case $state in
            active)    color=$GREEN ;;
            failed)    color=$RED ;;
            *)         color=$YELLOW ;;
        esac
        echo -e "  ${color}${svc}${NC}: ${state} (enabled: ${enabled})"
    done

    # Check gateway port
    echo ""
    if ss -tlnp 2>/dev/null | grep -q ':18789'; then
        pid=$(ss -tlnp 2>/dev/null | grep ':18789' | grep -oP 'pid=\K[0-9]+' | head -1)
        echo -e "  Gateway: ${GREEN}LISTENING${NC} on :18789 (PID ${pid:-?})"
    else
        echo -e "  Gateway: ${RED}NOT LISTENING${NC} on :18789"
    fi

    # Check Trading Core
    if curl -s --max-time 3 http://127.0.0.1:8000/health 2>/dev/null | grep -q "healthy"; then
        echo -e "  Trading Core: ${GREEN}HEALTHY${NC} on :8000"
    else
        echo -e "  Trading Core: ${YELLOW}NOT RUNNING${NC} on :8000"
    fi

    echo ""
    echo "Logs: journalctl -u clawdbot -f"
    exit 0
fi

# ── Uninstall ─────────────────────────────────────────────
if [ "$ACTION" = "uninstall" ]; then
    echo -e "${YELLOW}=== Uninstalling ClawdBot Services ===${NC}\n"
    for svc in "${SERVICES[@]}"; do
        systemctl stop "$svc" 2>/dev/null || true
        systemctl disable "$svc" 2>/dev/null || true
        rm -f "/etc/systemd/system/${svc}.service"
        echo -e "  ${GREEN}Removed: ${svc}${NC}"
    done
    systemctl daemon-reload
    echo -e "\n  ${GREEN}All services removed.${NC}"
    exit 0
fi

# ── Restart ───────────────────────────────────────────────
if [ "$ACTION" = "restart" ]; then
    echo -e "${CYAN}=== Restarting ClawdBot Services ===${NC}\n"
    for svc in "${SERVICES[@]}"; do
        systemctl restart "$svc" 2>/dev/null || true
        state=$(systemctl is-active "$svc" 2>/dev/null || echo "failed")
        echo -e "  ${svc}: ${state}"
    done
    exit 0
fi

# ── Logs ──────────────────────────────────────────────────
if [ "$ACTION" = "logs" ]; then
    echo -e "${CYAN}=== Recent ClawdBot Logs ===${NC}\n"
    for svc in "${SERVICES[@]}"; do
        echo -e "\n${YELLOW}--- ${svc} ---${NC}"
        journalctl -u "$svc" -n 10 --no-pager 2>/dev/null || echo "  (no logs)"
    done
    exit 0
fi

# ── Install ───────────────────────────────────────────────
echo -e "${CYAN}=== Installing ClawdBot Services (Ultimate-Fix) ===${NC}\n"

# 1. Fix PEM keys
echo -e "${YELLOW}[1/7] Fixing PEM keys...${NC}"
echo -e "${CYAN}=== Installing ClawdBot Services ===${NC}\n"

# 1. Fix PEM keys
echo -e "${YELLOW}[1/7] Fixing PEM keys...${NC}"
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from scripts.lib.pem_utils import fix_all_pem_files
n = fix_all_pem_files()
print(f'  Fixed {n} PEM file(s)')
" 2>/dev/null || echo "  Skipped (scripts.lib not available)"

# 2. Sync environment
echo -e "${YELLOW}[2/7] Syncing environment...${NC}"
echo -e "${YELLOW}[2/5] Syncing environment...${NC}"
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from scripts.lib.env_sync import sync_env_files
sync_env_files(apply=True, verbose=True)
" 2>/dev/null || echo "  Skipped (scripts.lib not available)"

# 3. Rebuild config
echo -e "${YELLOW}[3/7] Rebuilding moltbot.json...${NC}"
python3 "$PROJECT_DIR/scripts/rebuild-config.py" --quiet 2>/dev/null || \
    echo "  Skipped (rebuild-config.py not found)"

# 3b. Install ML dependencies (ultimate-fix)
echo -e "${YELLOW}[4/7] Installing ML dependencies...${NC}"
if [ -f "$PROJECT_DIR/requirements-ml.txt" ]; then
    pip3 install -r "$PROJECT_DIR/requirements-ml.txt" --quiet --break-system-packages 2>/dev/null \
        || pip3 install -r "$PROJECT_DIR/requirements-ml.txt" --quiet 2>/dev/null \
        || echo "  Skipped (pip install failed)"
else
    pip3 install numpy scipy lightgbm scikit-learn pandas pyarrow --quiet --break-system-packages 2>/dev/null \
        || pip3 install numpy scipy lightgbm scikit-learn pandas pyarrow --quiet 2>/dev/null \
        || echo "  Skipped"
fi
python3 -c "import lightgbm; import sklearn; print(f'  LightGBM {lightgbm.__version__}, scikit-learn {sklearn.__version__}')" 2>/dev/null || echo "  ML deps: some missing"

# 3c. Validate forecaster modules (ultimate-fix)
echo -e "${YELLOW}[5/7] Validating forecaster modules...${NC}"
python3 -c "
import sys, os
sys.path.insert(0, '$PROJECT_DIR')
try:
    from scripts.forecaster.engine import Forecaster
    fc = Forecaster()
    print(f'  Forecaster: {len(fc._modules)} modules, hybrid_ml={fc.enable_hybrid_ml}, regime_gate={fc.enable_regime_gate}')
except Exception as e:
    print(f'  Warning: {e}')
" 2>/dev/null || echo "  Skipped (import failed)"

# 4. Install service files
echo -e "${YELLOW}[6/7] Installing service files...${NC}"
echo -e "${YELLOW}[3/5] Rebuilding moltbot.json...${NC}"
python3 "$PROJECT_DIR/scripts/rebuild-config.py" --quiet 2>/dev/null || \
    echo "  Skipped (rebuild-config.py not found)"

# 3b. Install ML dependencies (ultimate-fix)
echo -e "${YELLOW}[4/7] Installing ML dependencies...${NC}"
if [ -f "$PROJECT_DIR/requirements-ml.txt" ]; then
    pip3 install -r "$PROJECT_DIR/requirements-ml.txt" --quiet --break-system-packages 2>/dev/null \
        || pip3 install -r "$PROJECT_DIR/requirements-ml.txt" --quiet 2>/dev/null \
        || echo "  Skipped (pip install failed)"
else
    pip3 install numpy scipy lightgbm scikit-learn pandas pyarrow --quiet --break-system-packages 2>/dev/null \
        || pip3 install numpy scipy lightgbm scikit-learn pandas pyarrow --quiet 2>/dev/null \
        || echo "  Skipped"
fi
python3 -c "import lightgbm; import sklearn; print(f'  LightGBM {lightgbm.__version__}, scikit-learn {sklearn.__version__}')" 2>/dev/null || echo "  ML deps: some missing"

# 3c. Validate forecaster modules (ultimate-fix)
echo -e "${YELLOW}[5/7] Validating forecaster modules...${NC}"
python3 -c "
import sys, os
sys.path.insert(0, '$PROJECT_DIR')
try:
    from scripts.forecaster.engine import Forecaster
    fc = Forecaster()
    print(f'  Forecaster: {len(fc._modules)} modules, hybrid_ml={fc.enable_hybrid_ml}, regime_gate={fc.enable_regime_gate}')
except Exception as e:
    print(f'  Warning: {e}')
" 2>/dev/null || echo "  Skipped (import failed)"

# 4. Install service files
echo -e "${YELLOW}[6/7] Installing service files...${NC}"

# Find moltbot binary
MOLTBOT_BIN=$(which moltbot 2>/dev/null || echo "/usr/local/bin/moltbot")

for svc in "${SERVICES[@]}"; do
    src="${SCRIPT_DIR}/${svc}.service"
    dst="/etc/systemd/system/${svc}.service"

    if [ ! -f "$src" ]; then
        echo -e "  ${RED}Missing: ${src}${NC}"
        continue
    fi

    # Copy and fix paths
    cp "$src" "$dst"

    # Update moltbot path if different from default
    if [ "$MOLTBOT_BIN" != "/usr/local/bin/moltbot" ]; then
        sed -i "s|/usr/local/bin/moltbot|$MOLTBOT_BIN|g" "$dst"
    fi

    # Update project path
    sed -i "s|/root/ClawdBot-V1|$PROJECT_DIR|g" "$dst"

    chmod 644 "$dst"
    echo -e "  ${GREEN}Installed: ${dst}${NC}"
done

systemctl daemon-reload
echo "  systemctl daemon-reload done"

# 5. Enable services
echo -e "${YELLOW}[7/7] Enabling services...${NC}"
echo -e "${YELLOW}[5/5] Enabling services...${NC}"
for svc in "${SERVICES[@]}"; do
    systemctl enable "$svc" 2>/dev/null || true
    echo -e "  ${GREEN}Enabled: ${svc}${NC}"
done

echo -e "\n${GREEN}=== Installation Complete (Ultimate-Fix) ===${NC}"
echo -e "\n${GREEN}=== Installation Complete ===${NC}"
echo ""
echo "Commands:"
echo "  systemctl start clawdbot                    # Start gateway"
echo "  systemctl start kalshi-edge-scanner          # Start scanner"
echo "  systemctl start yoshi-bridge                 # Start bridge"
echo "  bash scripts/services/install.sh --status    # Check status"
echo "  bash scripts/services/install.sh --restart   # Restart all"
echo "  journalctl -u clawdbot -f                    # View logs"
echo ""
echo "Ultimate-Fix commands:"
echo "  python3 -m scripts.forecaster.diagnose --auto-fix        # Diagnostics"
echo "  python3 -m scripts.forecaster.engine --symbol BTCUSDT   # Forecast"
echo "  python3 scripts/monte-carlo/simulation.py --live         # Monte Carlo"
echo "  bash scripts/deploy-ultimate-fix.sh                     # Full deploy"

# Start if requested
if [ "$START_AFTER" = true ]; then
    echo -e "\n${YELLOW}Starting services...${NC}"
    for svc in "${SERVICES[@]}"; do
        systemctl restart "$svc" 2>/dev/null || true
    done
    sleep 5

    # Verify
    echo ""
    for svc in "${SERVICES[@]}"; do
        state=$(systemctl is-active "$svc" 2>/dev/null || echo "failed")
        case $state in
            active)    color=$GREEN ;;
            *)         color=$RED ;;
        esac
        echo -e "  ${color}${svc}: ${state}${NC}"
    done

    # Check gateway
    echo ""
    if ss -tlnp 2>/dev/null | grep -q ':18789'; then
        echo -e "  Gateway: ${GREEN}LISTENING${NC} on :18789"
    else
        echo -e "  Gateway: ${RED}NOT LISTENING${NC} on :18789"
    fi
fi
