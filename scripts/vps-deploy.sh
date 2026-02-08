#!/bin/bash
# ════════════════════════════════════════════════════════════
# VPS Deploy Script — ClawdBot + Kalshi + Telegram
# ════════════════════════════════════════════════════════════
# Single script to deploy and configure everything on the VPS.
# Run: bash scripts/vps-deploy.sh
#
# What it does:
#   1. Pulls latest code from genspark_ai_developer
#   2. Configures .env with OpenRouter + Kalshi keys
#   3. Configures Telegram bot (@KalshiYoshiBot)
#   4. Validates API connectivity
#   5. Runs integration tests
#   6. Shows startup commands
# ════════════════════════════════════════════════════════════

set -euo pipefail

REPO_DIR="/root/ClawdBot-V1"
ENV_FILE="${REPO_DIR}/.env"

echo "════════════════════════════════════════════"
echo "  VPS DEPLOY — $(date '+%Y-%m-%d %H:%M %Z')"
echo "════════════════════════════════════════════"

# ── Step 1: Pull latest code ──────────────────────────────
echo -e "\n[1/5] Pulling latest code..."
cd "${REPO_DIR}"
git fetch origin
git reset --hard origin/genspark_ai_developer
git clean -fd
echo "  ✓ Code updated to $(git log --oneline -1)"

# ── Step 2: Configure .env ────────────────────────────────
echo -e "\n[2/5] Configuring .env..."

# Preserve existing Kalshi keys if present
EXISTING_KALSHI_KEY_ID=""
EXISTING_KALSHI_PK=""
if [ -f "${ENV_FILE}" ]; then
    EXISTING_KALSHI_KEY_ID=$(grep -oP '^KALSHI_KEY_ID=\K.*' "${ENV_FILE}" 2>/dev/null || true)
    # Don't try to extract multi-line PEM here
fi

# Set OpenRouter key (remove any stale OpenAI config)
sed -i '/^OPENAI_API_KEY=/d' "${ENV_FILE}" 2>/dev/null || true
sed -i '/^OPENAI_BASE_URL=/d' "${ENV_FILE}" 2>/dev/null || true
sed -i '/^OPENAI_MODEL=/d' "${ENV_FILE}" 2>/dev/null || true

# Add OpenRouter config
cat >> "${ENV_FILE}" << 'ENVBLOCK'

# ── LLM (OpenRouter free tier) ──
OPENAI_API_KEY=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
ENVBLOCK

# Also unset from current shell to prevent conflicts
unset OPENAI_BASE_URL 2>/dev/null || true

echo "  ✓ OpenRouter key configured"

# Set Telegram bot token (always update to latest)
YOSHI_TOKEN="8501633363:AAFDTd4U3S_qoCKVN2Y6m0D7v_qePVuUnnI"
if grep -q '^TELEGRAM_BOT_TOKEN=' "${ENV_FILE}" 2>/dev/null; then
    sed -i "s|^TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=${YOSHI_TOKEN}|" "${ENV_FILE}"
    echo "  ✓ Telegram bot token updated"
else
    cat >> "${ENV_FILE}" << TGBLOCK

# ── Telegram Bot (@KalshiYoshiBot) ──
TELEGRAM_BOT_TOKEN=${YOSHI_TOKEN}
TGBLOCK
    echo "  ✓ Telegram bot token added"
fi

# Set Telegram chat ID
YOSHI_CHAT_ID="8236163940"
if grep -q '^TELEGRAM_CHAT_ID=' "${ENV_FILE}" 2>/dev/null; then
    sed -i "s|^TELEGRAM_CHAT_ID=.*|TELEGRAM_CHAT_ID=${YOSHI_CHAT_ID}|" "${ENV_FILE}"
else
    echo "TELEGRAM_CHAT_ID=${YOSHI_CHAT_ID}" >> "${ENV_FILE}"
fi
echo "  ✓ Telegram chat ID configured"

# Check Kalshi credentials
if [ -n "${EXISTING_KALSHI_KEY_ID}" ] && [ "${EXISTING_KALSHI_KEY_ID}" != "your_kalshi_key_id_here" ]; then
    echo "  ✓ Kalshi key ID found: ${EXISTING_KALSHI_KEY_ID:0:12}..."
elif [ -n "${KALSHI_KEY_ID:-}" ]; then
    echo "  ✓ Kalshi key ID from env: ${KALSHI_KEY_ID:0:12}..."
else
    echo "  ⚠ Kalshi key ID not found — scanner will fail"
    echo "    Add to ${ENV_FILE}: KALSHI_KEY_ID=your-key-id"
fi

# Check for PEM key
if [ -f ~/.kalshi/private_key.pem ]; then
    echo "  ✓ Kalshi PEM found at ~/.kalshi/private_key.pem"
elif grep -q "BEGIN.*PRIVATE KEY" "${ENV_FILE}" 2>/dev/null; then
    echo "  ✓ Kalshi PEM found in .env"
else
    echo "  ⚠ Kalshi private key not found"
    echo "    Save to ~/.kalshi/private_key.pem (chmod 600)"
fi

# ── Step 3: Validate API connectivity ─────────────────────
echo -e "\n[3/6] Testing API connectivity..."

# Test OpenRouter
python3 -c "
import json, os
from urllib import request
os.environ.pop('OPENAI_BASE_URL', None)
key = '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'
url = 'https://openrouter.ai/api/v1/chat/completions'
payload = {'model': 'meta-llama/llama-3.3-70b-instruct:free', 'messages': [{'role': 'user', 'content': 'Say OK'}], 'max_tokens': 5}
body = json.dumps(payload).encode()
req = request.Request(url, data=body, headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}, method='POST')
with request.urlopen(req, timeout=15) as resp:
    data = json.loads(resp.read().decode())
    print(f'  ✓ OpenRouter: {data.get(\"model\", \"?\")} — {data[\"choices\"][0][\"message\"][\"content\"][:20]}')
" 2>&1 || echo "  ✗ OpenRouter: FAILED"

# Test Kalshi exchange
python3 -c "
import json
from urllib import request
url = 'https://api.elections.kalshi.com/trade-api/v2/exchange/status'
req = request.Request(url, headers={'Accept': 'application/json'})
with request.urlopen(req, timeout=10) as resp:
    data = json.loads(resp.read().decode())
    active = data.get('exchange_active', False)
    trading = data.get('trading_active', False)
    print(f'  ✓ Kalshi: exchange={\"OPEN\" if active else \"CLOSED\"}, trading={\"ACTIVE\" if trading else \"INACTIVE\"}')
" 2>&1 || echo "  ✗ Kalshi: FAILED"

# Test LLM routing detection
python3 -c "
import os, sys
sys.path.insert(0, '.')
os.environ.pop('OPENAI_BASE_URL', None)
from gnosis.reasoning.client import LLMConfig
cfg = LLMConfig.from_yaml()
print(f'  ✓ LLM routing: env={cfg._environment}, model={cfg.model}')
" 2>&1 || echo "  ✗ LLM routing: FAILED"

# Test Telegram bot
python3 -c "
import json
from urllib import request as urlreq
url = 'https://api.telegram.org/bot8501633363:AAFDTd4U3S_qoCKVN2Y6m0D7v_qePVuUnnI/getMe'
with urlreq.urlopen(url, timeout=10) as resp:
    data = json.loads(resp.read().decode())
    if data.get('ok'):
        bot = data['result']
        print(f'  ✓ Telegram: @{bot.get(\"username\", \"?\")} (id={bot.get(\"id\", \"?\")})') 
    else:
        print('  ✗ Telegram: invalid token')
" 2>&1 || echo "  ✗ Telegram: FAILED"

# ── Step 4: Integration tests ─────────────────────────────
echo -e "\n[4/6] Running integration tests..."
cd "${REPO_DIR}"
python3 tests/test_integration.py 2>&1 | tail -5

# ── Step 5: Kill old processes ────────────────────────────
echo -e "\n[5/6] Stopping old processes..."
pkill -f 'telegram-bot.py' 2>/dev/null && echo "  ✓ Old Telegram bot stopped" || echo "  - No old Telegram bot running"
pkill -f 'kalshi-system.py' 2>/dev/null && echo "  ✓ Old Kalshi system stopped" || echo "  - No old Kalshi system running"

# ── Step 6: Ready ─────────────────────────────────────────
echo -e "\n[6/6] Deploy complete!"
echo ""
echo "════════════════════════════════════════════"
echo "  READY — Run these commands:"
echo "════════════════════════════════════════════"
echo ""
echo "  # ★ TELEGRAM BOT (recommended — alerts + commands):"
echo "  nohup python3 scripts/telegram-bot.py --interval 60 > /tmp/yoshi-bot.log 2>&1 &"
echo ""
echo "  # View bot logs:"
echo "  tail -f /tmp/yoshi-bot.log"
echo ""
echo "  # Telegram bot (no Kalshi, forecast only):"
echo "  python3 scripts/telegram-bot.py --no-kalshi --interval 120"
echo ""
echo "  # Quick scan (single run, no Telegram):"
echo "  python3 scripts/kalshi-pipeline.py"
echo ""
echo "  # Full system (no Telegram):"
echo "  python3 scripts/kalshi-system.py"
echo ""
echo "════════════════════════════════════════════"
