#!/bin/bash
# ClawdBot Gateway Fix v2 — handles ALL known issues:
#   1) Stale gateway PID blocking port 18789 (nuclear kill)
#   2) Missing/corrupt ~/.clawdbot/moltbot.json
#   3) PEM key deserialization (newlines stripped in .env)
#   4) Edge scanner using wrong python path
#   5) Telegram token conflict across multiple .env files
#   6) Post-restart Telegram health verification
#
# Run:  cd /root/ClawdBot-V1 && bash scripts/fix-gateway.sh

set -euo pipefail

echo "=== ClawdBot Gateway Fix v2 ==="

# ── 1. Nuclear kill: stop services + kill ALL moltbot processes + free port ──
echo "1/9 Stopping services + killing ALL moltbot processes..."
systemctl stop clawdbot 2>/dev/null || true
systemctl stop kalshi-edge-scanner 2>/dev/null || true
systemctl stop yoshi-bridge 2>/dev/null || true

# Kill every moltbot process — gateway, child, orphan, zombie
pkill -9 -f "moltbot" 2>/dev/null || true
sleep 1

# Kill anything still holding port 18789
fuser -k -9 18789/tcp 2>/dev/null || true
sleep 1

# Final verification loop — retry up to 3 times
for i in 1 2 3; do
    if ! fuser 18789/tcp 2>/dev/null; then
        echo "  Port 18789: FREE"
        break
    fi
    echo "  Port 18789 still in use (attempt $i/3), force killing..."
    fuser -k -9 18789/tcp 2>/dev/null || true
    # Also kill by lsof as backup
    lsof -ti:18789 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 2
done

# Final check — fail loudly if port is still occupied
if fuser 18789/tcp 2>/dev/null; then
    echo "  FATAL: Cannot free port 18789 after 3 attempts"
    echo "  Run: lsof -i:18789"
    exit 1
fi
echo "  Done"

# ── 2. Nuke ALL stale moltbot state (with backup) ──
echo "2/8 Cleaning state..."
# Backup and remove directories with confirmation
FORCE_NUKE="${FORCE_NUKE:-}"
for DIR in /root/.moltbot /root/.clawdbot; do
    if [ -d "$DIR" ]; then
        if [ -z "$FORCE_NUKE" ]; then
            # Create timestamped backup
            BACKUP_DIR="${DIR}.bak.$(date +%s)"
            mv "$DIR" "$BACKUP_DIR" 2>/dev/null || true
            echo "  Backed up $DIR to $BACKUP_DIR"
        else
            # Force removal without backup
            rm -rf "$DIR" 2>/dev/null || true
            echo "  Removed $DIR (FORCE_NUKE set)"
        fi
    fi
done
# ── 2. Nuke ALL stale moltbot state ──
echo "2/9 Cleaning state..."
rm -rf /root/.moltbot
rm -rf /root/.clawdbot
# Also clean stale lock files
rm -f /tmp/moltbot/*.lock 2>/dev/null || true
rm -f /tmp/moltbot-*.lock 2>/dev/null || true
mkdir -p /root/.clawdbot
chmod 700 /root/.clawdbot
echo "  Done"

# ── 3. Pull latest code ──
echo "3/9 Updating code..."
cd /root/ClawdBot-V1 || { echo "ERROR: /root/ClawdBot-V1 not found"; exit 1; }
# Use branch from argument or env var, defaulting to main
BRANCH="${1:-${TARGET_BRANCH:-main}}"
git fetch origin 2>/dev/null || true
git reset --hard "origin/$BRANCH" 2>/dev/null || true
echo "  HEAD: $(git log --oneline -1)"

# ── 4. Install Python deps ──
echo "4/9 Installing Python deps..."
pip3 install cryptography numpy -q 2>/dev/null || \
    pip3 install cryptography numpy --break-system-packages -q 2>/dev/null || true
echo "  Done"

# ── 5. Collect + VALIDATE Telegram token ──
# Source all .env files, then TEST each unique token to find the working one
echo "5/9 Resolving Telegram token..."

GATEWAY_TOKEN=$(openssl rand -hex 32)

# Collect all candidate tokens from .env files (order: ClawdBot, Yoshi, root)
declare -a CANDIDATE_TOKENS=()
declare -A TOKEN_SOURCE=()
for F in /root/ClawdBot-V1/.env /root/Yoshi-Bot/.env /root/.env; do
    [ -f "$F" ] || continue
    T=$(grep -m1 '^TELEGRAM_BOT_TOKEN=' "$F" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'" || true)
    if [ -n "$T" ]; then
        # Deduplicate
        SEEN=0
        for CT in "${CANDIDATE_TOKENS[@]+"${CANDIDATE_TOKENS[@]}"}"; do
            [ "$CT" = "$T" ] && SEEN=1 && break
        done
        if [ "$SEEN" = "0" ]; then
            CANDIDATE_TOKENS+=("$T")
            TOKEN_SOURCE["$T"]="$F"
        fi
    fi
done

echo "  Found ${#CANDIDATE_TOKENS[@]} unique token(s)"

# Test each token against Telegram getMe API
WORKING_TOKEN=""
for T in "${CANDIDATE_TOKENS[@]+"${CANDIDATE_TOKENS[@]}"}"; do
    LAST8="${T: -8}"
    RESULT=$(curl -s --max-time 10 "https://api.telegram.org/bot${T}/getMe" 2>/dev/null || echo '{"ok":false}')
    if echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('ok') else 1)" 2>/dev/null; then
        BOT_NAME=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result'].get('username','?'))" 2>/dev/null || echo "?")
        echo "  VALID: ...${LAST8} → @${BOT_NAME} (from ${TOKEN_SOURCE[$T]})"
        WORKING_TOKEN="$T"
        break
    else
        echo "  DEAD:  ...${LAST8} (from ${TOKEN_SOURCE[$T]}) — 401 Unauthorized"
    fi
done

if [ -z "$WORKING_TOKEN" ]; then
    echo "  FATAL: No working Telegram token found across .env files"
    echo "  Go to https://t.me/BotFather → /token to get a new one"
    echo "  Then set TELEGRAM_BOT_TOKEN=<new_token> in /root/ClawdBot-V1/.env"
    exit 1
fi

export TELEGRAM_BOT_TOKEN="$WORKING_TOKEN"

# ── 6. Synchronize the working token into ALL .env files ──
echo "6/9 Synchronizing token + sourcing env..."
for F in /root/ClawdBot-V1/.env /root/Yoshi-Bot/.env /root/.env; do
    [ -f "$F" ] || continue
    # Replace any existing TELEGRAM_BOT_TOKEN line with the working one
    if grep -q '^TELEGRAM_BOT_TOKEN=' "$F" 2>/dev/null; then
        sed -i "s|^TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=${WORKING_TOKEN}|" "$F"
        echo "  Updated: $F"
    fi
done

# Now source all envs to pick up GOOGLE_API_KEY, OPENAI_API_KEY, KALSHI keys, etc.
# Token is already locked to the working one — sourcing won't override it
export GOOGLE_API_KEY="" OPENAI_API_KEY=""
for F in /root/ClawdBot-V1/.env /root/Yoshi-Bot/.env /root/.env; do
    if [ -f "$F" ]; then
        # Source but protect TELEGRAM_BOT_TOKEN
        SAVED_TOKEN="$TELEGRAM_BOT_TOKEN"
        set -a && source "$F" 2>/dev/null && set +a
        export TELEGRAM_BOT_TOKEN="$SAVED_TOKEN"
    fi
done

# Ensure GOOGLE_API_KEY persists in ClawdBot .env
if [ -n "$GOOGLE_API_KEY" ]; then
    if grep -q '^GOOGLE_API_KEY=' /root/ClawdBot-V1/.env 2>/dev/null; then
        sed -i "s|^GOOGLE_API_KEY=.*|GOOGLE_API_KEY=${GOOGLE_API_KEY}|" /root/ClawdBot-V1/.env
    else
        echo "GOOGLE_API_KEY=$GOOGLE_API_KEY" >> /root/ClawdBot-V1/.env
    fi
fi

echo "  Telegram: ...${TELEGRAM_BOT_TOKEN: -8}"
echo "  Google API Key: ${GOOGLE_API_KEY:0:8}..."

# ── 6. Fix Kalshi PEM key (convert \n literals to real newlines) ──
echo "6/8 Fixing Kalshi key + writing config..."
# Fix the PEM key in all .env files — many tools store PEM as single-line with literal \n
for ENVFILE in /root/Yoshi-Bot/.env /root/ClawdBot-V1/.env; do
    [ -f "$ENVFILE" ] || continue

    # Use python to fix PEM in .env file
    python3 -c "
import os
envfile = '$ENVFILE'
if not os.path.isfile(envfile):
    exit()
content = open(envfile).read()
# If KALSHI_PRIVATE_KEY has literal backslash-n, fix it
import re
def fix_pem(m):
    val = m.group(1)
    # Replace literal \\\\n (four-char sequence: backslash backslash n) with real newlines
    val = val.replace('\\\\n', '\n')
    # Also handle cases where it's just \\n (three-char sequence: backslash n)
    val = val.replace('\\n', '\n')
    return 'KALSHI_PRIVATE_KEY=\"' + val + '\"'
content = re.sub(r'KALSHI_PRIVATE_KEY=\"(.+?)\"', fix_pem, content, flags=re.DOTALL)
open(envfile, 'w').write(content)
" 2>/dev/null || true
done
echo "  Google API Key: ${GOOGLE_API_KEY:+${GOOGLE_API_KEY:0:8}...}${GOOGLE_API_KEY:-NONE}"

# ── 7. Fix Kalshi PEM key + write config ──
echo "7/9 Fixing Kalshi PEM + writing config..."

# Use shared PEM utility (scripts/lib/pem_utils.py)
cd /root/ClawdBot-V1
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from scripts.lib.pem_utils import fix_all_pem_files
    n = fix_all_pem_files(verbose=True)
    print(f'  Fixed {n} PEM file(s) via shared utility')
except ImportError:
    # Fallback: inline PEM fix (if shared utils not yet deployed)
    import os, re
    for envfile in ['/root/Yoshi-Bot/.env', '/root/ClawdBot-V1/.env']:
        if not os.path.isfile(envfile): continue
        content = open(envfile).read()
        def fix_pem(m):
            val = m.group(1)
            val = val.replace(r'\n', '\n')
            return 'KALSHI_PRIVATE_KEY=\"' + val + '\"'
        new = re.sub(r'KALSHI_PRIVATE_KEY=\"(.+?)\"', fix_pem, content, flags=re.DOTALL)
        if new != content:
            open(envfile, 'w').write(new)
            print(f'  Fixed PEM in: {envfile}')
" 2>/dev/null || true

# Write moltbot.json config using shared rebuild-config.py
export GATEWAY_TOKEN
python3 scripts/rebuild-config.py --gateway-token "$GATEWAY_TOKEN" 2>/dev/null || {
    # Fallback: inline config writer
    python3 << 'PYEOF'
import json, os

google_key = os.environ.get("GOOGLE_API_KEY", "")
telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")

if google_key:
    primary_model = "google/gemini-3-flash-preview"
    fallback_models = ["google/gemini-2.5-flash", "google/gemini-2.5-flash-lite"]
else:
    primary_model = "openai/gpt-4o-mini"
    fallback_models = ["openai/gpt-4o"]

# Load allowlist from environment or use a secure default
allowFrom = os.environ.get("TELEGRAM_ALLOWLIST", "").strip()
if allowFrom:
    # Parse comma-separated list of user IDs
    allowFrom = [x.strip() for x in allowFrom.split(",") if x.strip()]
else:
    # Default: require explicit configuration to avoid open access
    allowFrom = []

# Validate allowlist
if not allowFrom or allowFrom == ["*"]:
    print("  WARNING: No valid Telegram allowlist configured!")
    print("  Set TELEGRAM_ALLOWLIST env var with comma-separated user IDs")
    print("  Defaulting to empty allowlist (will reject all DMs)")
    allowFrom = []

config = {
    "agents": {
        "defaults": {
            "workspace": "~/clawd",
            "model": {
                "primary": primary_model,
                "fallbacks": fallback_models
            },
            "thinkingDefault": "low"
        },
        "list": [{
            "id": "main",
            "default": True,
            "identity": {"name": "ClawdBot", "theme": "crypto trading assistant", "emoji": "\U0001f916"}
        }]
    },
    "gateway": {
        "mode": "local",
        "port": 18789,
        "bind": "loopback",
        "auth": {"mode": "token", "token": os.environ["GATEWAY_TOKEN"]}
    },
    "channels": {
        "telegram": {
            "enabled": True,
            "botToken": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            "dmPolicy": "open",
            "allowFrom": allowFrom
        }
    },
    "skills": {
        "load": {"extraDirs": ["/root/ClawdBot-V1/skills"]}
    }
    "agents": {"defaults": {"workspace": "~/clawd", "model": {"primary": primary_model, "fallbacks": fallback_models}, "thinkingDefault": "low"},
        "list": [{"id": "main", "default": True, "identity": {"name": "ClawdBot", "theme": "crypto trading assistant", "emoji": "\U0001f916"}}]},
    "gateway": {"mode": "local", "port": 18789, "bind": "loopback", "auth": {"mode": "token", "token": os.environ["GATEWAY_TOKEN"]}},
    "channels": {"telegram": {"enabled": True, "botToken": telegram_token, "dmPolicy": "open", "allowFrom": ["*"]}},
    "skills": {"load": {"extraDirs": ["/root/ClawdBot-V1/skills"]}}
}
if google_key:
    config["env"] = {"GOOGLE_API_KEY": google_key, "GEMINI_API_KEY": google_key}

for cfg_path in [os.path.expanduser("~/.clawdbot/moltbot.json"), os.path.expanduser("~/.moltbot/moltbot.json")]:
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config written: {cfg_path}")

# Verify
with open(os.path.expanduser("~/.clawdbot/moltbot.json")) as f:
    d = json.load(f)
assert d["gateway"]["mode"] == "local"
assert d["gateway"]["auth"]["token"] == os.environ["GATEWAY_TOKEN"]
assert d["agents"]["defaults"]["model"]["primary"] == primary_model
print(f"  Verified: model={primary_model}, allowlist={len(allowFrom)} users")
print(f"  Model: {primary_model}, token=...{telegram_token[-8:]}")
PYEOF
}

# ── 8. Install systemd services + restart ──
echo "8/9 Installing systemd services + restarting..."

# Use the service installer if available, otherwise install inline
if [ -x /root/ClawdBot-V1/scripts/services/install.sh ]; then
    bash /root/ClawdBot-V1/scripts/services/install.sh --start
else
    MOLTBOT_BIN=$(which moltbot 2>/dev/null || echo "/usr/local/bin/moltbot")

    # Install service files from scripts/services/ if they exist
    for SVC in clawdbot kalshi-edge-scanner yoshi-bridge; do
        SRC="/root/ClawdBot-V1/scripts/services/${SVC}.service"
        if [ -f "$SRC" ]; then
            cp "$SRC" "/etc/systemd/system/${SVC}.service"
            # Fix moltbot path
            sed -i "s|/usr/local/bin/moltbot|$MOLTBOT_BIN|g" "/etc/systemd/system/${SVC}.service"
            echo "  Installed: $SVC"
        fi
    done

    # Fix edge scanner python path
    sed -i "s|/root/Yoshi-Bot/venv/bin/python3|/usr/bin/python3|" \
        /etc/systemd/system/kalshi-edge-scanner.service 2>/dev/null || true

# ── 7. Write systemd services ──
echo "7/8 Writing services..."
MOLTBOT_BIN=$(which moltbot 2>/dev/null || echo "/usr/bin/moltbot")

cat > /etc/systemd/system/clawdbot.service << SVCEOF
[Unit]
Description=ClawdBot Telegram AI Trading Assistant
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/ClawdBot-V1
EnvironmentFile=/root/ClawdBot-V1/.env
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=NODE_ENV=production
Environment=CLAWDBOT_GATEWAY_TOKEN=$GATEWAY_TOKEN
Environment=GOOGLE_API_KEY=${GOOGLE_API_KEY:-}
Environment=GEMINI_API_KEY=${GOOGLE_API_KEY:-}
ExecStart=$MOLTBOT_BIN gateway --port 18789
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=clawdbot

[Install]
WantedBy=multi-user.target
SVCEOF

# Fix edge scanner to use system python (no venv)
sed -i "s|/root/Yoshi-Bot/venv/bin/python3|/usr/bin/python3|" /etc/systemd/system/kalshi-edge-scanner.service 2>/dev/null || true
    systemctl daemon-reload
    systemctl restart clawdbot
    systemctl restart kalshi-edge-scanner 2>/dev/null || true
    systemctl restart yoshi-bridge 2>/dev/null || true
fi
echo "  Done"

# ── 9. Post-restart health check ──
echo "9/9 Verifying services + Telegram connectivity..."
sleep 8

echo ""
for SVC in clawdbot kalshi-edge-scanner yoshi-bridge; do
    ST=$(systemctl is-active "$SVC" 2>/dev/null || echo "dead")
    [ "$ST" = "active" ] && echo "  $SVC: RUNNING" || echo "  $SVC: $ST"
done

# Check gateway is listening
echo ""
if ss -tlnp 2>/dev/null | grep -q ':18789'; then
    GW_PID=$(ss -tlnp 2>/dev/null | grep ':18789' | grep -oP 'pid=\K[0-9]+' | head -1)
    echo "  Gateway: LISTENING on :18789 (PID ${GW_PID:-?})"
else
    echo "  Gateway: NOT LISTENING on :18789"
    echo "  Check: journalctl -u clawdbot -n 20 --no-pager"
fi

# Check Telegram connectivity in logs
echo ""
TG_LOG=$(journalctl -u clawdbot --since "30 sec ago" --no-pager 2>/dev/null | grep -iE "telegram|polling|401|unauthorized" | tail -5)
if echo "$TG_LOG" | grep -qi "polling\|connected\|started"; then
    echo "  Telegram: CONNECTED"
    echo "$TG_LOG" | head -3 | sed 's/^/    /'
elif echo "$TG_LOG" | grep -qi "401\|unauthorized"; then
    echo "  Telegram: FAILED (401 Unauthorized)"
    echo "  The token passed validation at startup but the gateway is rejecting it."
    echo "  Check: journalctl -u clawdbot -n 20 --no-pager | grep -i telegram"
    echo "$TG_LOG" | head -3 | sed 's/^/    /'
elif [ -z "$TG_LOG" ]; then
    echo "  Telegram: PENDING (no log entries yet — gateway may still be initializing)"
    echo "  Check in 10s: journalctl -u clawdbot -n 20 --no-pager | grep -i telegram"
else
    echo "  Telegram: UNKNOWN"
    echo "$TG_LOG" | head -3 | sed 's/^/    /'
fi

echo ""
CS=$(systemctl is-active clawdbot 2>/dev/null || echo "dead")
if [ "$CS" = "active" ]; then
    echo "============================="
    echo "  ClawdBot: RUNNING"
    echo "============================="
else
    echo "============================="
    echo "  ClawdBot: $CS"
    echo "============================="
    echo ""
    echo "Last 15 log lines:"
    journalctl -u clawdbot -n 15 --no-pager 2>/dev/null || true
fi

ES=$(systemctl is-active kalshi-edge-scanner 2>/dev/null || echo "dead")
if [ "$ES" != "active" ]; then
    echo ""
    echo "Edge Scanner logs:"
    journalctl -u kalshi-edge-scanner -n 10 --no-pager 2>/dev/null || true
fi

echo ""
echo "Done."
