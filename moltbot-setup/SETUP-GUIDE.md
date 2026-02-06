# MoltBot Setup Guide for DigitalOcean Droplet

A step-by-step guide to set up MoltBot on your DigitalOcean VPS.

## Prerequisites

- **DigitalOcean account** with a droplet running **Ubuntu 24.04 LTS**
- **2GB RAM minimum** (recommended)
- **SSH access** to your droplet
- **LLM API key** (Anthropic Claude, OpenAI, etc.)

---

## Quick Start (Copy & Paste Commands)

### Step 1: SSH into Your Droplet

```bash
ssh root@YOUR_DROPLET_IP
```

Replace `YOUR_DROPLET_IP` with your actual droplet IP address.

---

### Step 2: Create a Dedicated User (Security Best Practice)

```bash
# Create the clawd user and give sudo access
adduser clawd && usermod -aG sudo clawd

# Switch to the new user
su - clawd
```

Set a password when prompted.

---

### Step 3: Install MoltBot

```bash
# Install MoltBot using the official installer
curl -fsSL https://clawd.bot/install.sh | bash

# Reload your shell to get the new commands
exec bash
```

---

### Step 4: Run the Onboarding Wizard

```bash
moltbot onboard --install-daemon
```

The wizard will guide you through:
1. **LLM Provider Configuration** - Enter your API key (Anthropic recommended)
2. **Workspace Setup** - Creates `~/clawd/` directory
3. **Channel Setup** - Connect WhatsApp, Telegram, Slack, etc.
4. **Daemon Installation** - Auto-start service via systemd

For **WhatsApp**: You'll see a QR code - scan it with WhatsApp > Settings > Linked Devices > Link a Device

---

## Managing MoltBot

### Gateway Commands

```bash
# Check status
moltbot gateway status

# Start/Stop/Restart
moltbot gateway start
moltbot gateway stop
moltbot gateway restart

# View logs
moltbot logs --follow
```

### Chat Interfaces

```bash
# Terminal UI (interactive chat)
moltbot tui

# Send a message directly
moltbot agent --message "Hello, what can you do?"
```

---

## Accessing the Web Dashboard

The dashboard runs on `localhost:18789`. To access it from your local machine:

### Option 1: SSH Tunnel (Recommended)

First, set up SSH access for the clawd user:

```bash
# On the droplet (as clawd user)
mkdir -p ~/.ssh && chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
# Paste your public SSH key from your local machine
# Save and exit (Ctrl+X, Y, Enter)
chmod 600 ~/.ssh/authorized_keys
```

Then from your **local machine**:

```bash
ssh -L 18789:127.0.0.1:18789 clawd@YOUR_DROPLET_IP
```

Now open in browser: **http://127.0.0.1:18789**

### Option 2: Tailscale (For Secure Remote Access)

```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Configure MoltBot to use Tailscale Serve
moltbot config set gateway.tailscale.mode serve
```

---

## Installing Skills

MoltBot has 50+ built-in skills and you can add more from ClawdHub:

```bash
# Search for skills
clawdhub search "calendar"

# Install a skill
clawdhub install google-calendar

# List installed skills
clawdhub list
```

Popular skills:
- `github` - GitHub integration
- `google-calendar` - Calendar management
- `slack` - Slack messaging
- `brave-search` - Web search
- `browser` - Browser automation

---

## Workspace Files

MoltBot stores its memory and configuration in `~/clawd/`:

```
~/clawd/
├── AGENTS.md      # Available agents
├── IDENTITY.md    # Bot's identity
├── SOUL.md        # Personality traits
├── TOOLS.md       # Available tools
├── USER.md        # What it knows about you
├── memory/        # Long-term memory
├── skills/        # Installed skills
└── canvas/        # Working directory
```

You can edit these files to customize MoltBot's behavior!

---

## Backing Up Your Data

```bash
# From your local machine - download the entire workspace
scp -r clawd@YOUR_DROPLET_IP:~/clawd ~/moltbot-backup-$(date +%Y%m%d)
```

---

## Configuration

Edit `~/.clawdbot/moltbot.json` for advanced settings:

```json
{
  "agent": {
    "model": "anthropic/claude-opus-4-5"
  },
  "channels": {
    "whatsapp": {
      "enabled": true
    },
    "telegram": {
      "botToken": "YOUR_TELEGRAM_BOT_TOKEN"
    }
  }
}
```

---

## Troubleshooting

### Check if MoltBot is running
```bash
moltbot gateway status
```

### View recent logs
```bash
moltbot logs --follow
```

### Restart the gateway
```bash
moltbot gateway restart
```

### Re-run onboarding
```bash
moltbot onboard
```

### Check system resources
```bash
htop  # or: top
```

---

## Useful Commands Reference

| Command | Description |
|---------|-------------|
| `moltbot onboard` | Run setup wizard |
| `moltbot tui` | Terminal chat interface |
| `moltbot gateway status` | Check gateway status |
| `moltbot gateway restart` | Restart the gateway |
| `moltbot logs --follow` | Stream logs |
| `moltbot doctor` | Diagnose issues |
| `clawdhub search <query>` | Search for skills |
| `clawdhub install <skill>` | Install a skill |

---

## Security Notes

⚠️ **Important Security Considerations:**

1. **Don't run as root** - Always use the dedicated `clawd` user
2. **DM Policy** - By default, unknown senders receive a pairing code
3. **Sensitive Data** - Don't store sensitive data in the MoltBot workspace
4. **Firewall** - Consider setting up UFW to restrict access
5. **Updates** - Keep MoltBot updated: `npm update -g moltbot`

---

## Need Help?

- **Official Docs**: https://docs.molt.bot/
- **Discord**: https://discord.gg/clawd
- **GitHub**: https://github.com/moltbot/moltbot

Happy automating! 🦞
