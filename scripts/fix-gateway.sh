#!/bin/bash
# ClawdBot Gateway Fix — handles ALL known issues:
#   1) Stale gateway PID blocking port 18789
#   2) Missing/corrupt ~/.clawdbot/moltbot.json
#   3) PEM key deserialization (newlines stripped in .env)
#   4) Edge scanner using wrong python path
#
# Run:  curl -sSL URL -o /tmp/fix.sh && bash /tmp/fix.sh

echo "=== ClawdBot Gateway Fix ==="

# ── 1. Kill EVERYTHING on port 18789 + stop services ──
echo "1/8 Stopping services + killing stale processes..."
systemctl stop clawdbot 2>/dev/null || true
systemctl stop kalshi-edge-scanner 2>/dev/null || true
# Kill any stale moltbot/gateway processes
pkill -9 -f "moltbot gateway" 2>/dev/null || true
pkill -9 -f "moltbot.*18789" 2>/dev/null || true
# Kill anything on port 18789
fuser -k 18789/tcp 2>/dev/null || true
sleep 2
# Verify port is free
if fuser 18789/tcp 2>/dev/null; then
    echo "  WARNING: port 18789 still in use, force killing..."
    fuser -k -9 18789/tcp 2>/dev/null || true
    sleep 1
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
mkdir -p /root/.clawdbot
chmod 700 /root/.clawdbot
echo "  Done"

# ── 3. Pull latest code ──
echo "3/8 Updating code..."
cd /root/ClawdBot-V1 || { echo "ERROR: /root/ClawdBot-V1 not found"; exit 1; }
# Use branch from argument or env var, defaulting to main
BRANCH="${1:-${TARGET_BRANCH:-main}}"
git fetch origin 2>/dev/null || true
git reset --hard "origin/$BRANCH" 2>/dev/null || true
echo "  HEAD: $(git log --oneline -1)"

# ── 4. Install cryptography for Kalshi API ──
echo "4/8 Installing Python deps..."
pip3 install cryptography numpy -q 2>/dev/null || pip3 install cryptography numpy --break-system-packages -q 2>/dev/null || true
echo "  Done"

# ── 5. Generate token + source env ──
GATEWAY_TOKEN=$(openssl rand -hex 32)
echo "5/8 Token: ${GATEWAY_TOKEN:0:8}..."

export TELEGRAM_BOT_TOKEN="" OPENAI_API_KEY="" GOOGLE_API_KEY=""
for F in /root/ClawdBot-V1/.env /root/Yoshi-Bot/.env /root/.env; do
    [ -f "$F" ] && set -a && source "$F" 2>/dev/null && set +a
done

# Ensure GOOGLE_API_KEY is in .env for persistence
if [ -n "$GOOGLE_API_KEY" ]; then
    grep -q 'GOOGLE_API_KEY=' /root/ClawdBot-V1/.env 2>/dev/null || \
        echo "GOOGLE_API_KEY=$GOOGLE_API_KEY" >> /root/ClawdBot-V1/.env
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
    # Replace literal \\\\n with real newlines (double backslash)
    val = val.replace('\\\\n', '\n')
    # Also handle cases where it's just \\n (single backslash)
    val = val.replace('\\n', '\n')
    return 'KALSHI_PRIVATE_KEY=\"' + val + '\"'
content = re.sub(r'KALSHI_PRIVATE_KEY=\"(.+?)\"', fix_pem, content, flags=re.DOTALL)
open(envfile, 'w').write(content)
" 2>/dev/null || true
done

# Also ensure ~/.kalshi/private_key.pem has proper newlines
if [ -f /root/.kalshi/private_key.pem ]; then
    python3 -c "
p = '/root/.kalshi/private_key.pem'
d = open(p).read()
if '\\\\n' in d:
    open(p, 'w').write(d.replace('\\\\n', '\n'))
" 2>/dev/null || true
fi

# Write config with python3 (no heredoc interpolation issues)
export GATEWAY_TOKEN
python3 << 'PYEOF'
import json, os

# Determine model: prefer Gemini (free/cheap), fall back to OpenAI
google_key = os.environ.get("GOOGLE_API_KEY", "")
openai_key = os.environ.get("OPENAI_API_KEY", "")

if google_key:
    primary_model = "google/gemini-2.0-flash"
    fallback_models = ["google/gemini-2.0-flash-lite"]
    print(f"  Model: {primary_model} (Google Gemini — free tier)")
else:
    primary_model = "openai/gpt-4o-mini"
    fallback_models = ["openai/gpt-4o"]
    print(f"  Model: {primary_model} (OpenAI fallback)")

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
}

# Set Google API key in env block if available
if google_key:
    config["env"] = {"GOOGLE_API_KEY": google_key, "GEMINI_API_KEY": google_key}

# Write to BOTH config paths (moltbot reads from either)
for cfg_path in [
    os.path.expanduser("~/.clawdbot/moltbot.json"),
    os.path.expanduser("~/.moltbot/moltbot.json"),
]:
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
PYEOF

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
echo "  Done"

# ── 8. Restart all ──
echo "8/8 Restarting..."
systemctl daemon-reload
systemctl restart clawdbot
systemctl restart kalshi-edge-scanner 2>/dev/null || true
systemctl restart yoshi-bridge 2>/dev/null || true
sleep 6

# ── Result ──
echo ""
for SVC in clawdbot kalshi-edge-scanner yoshi-bridge; do
    ST=$(systemctl is-active $SVC 2>/dev/null || echo "dead")
    [ "$ST" = "active" ] && echo "  $SVC: RUNNING" || echo "  $SVC: $ST"
done
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
