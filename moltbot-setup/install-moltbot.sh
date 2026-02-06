#!/bin/bash
#
# MoltBot Installation Script for DigitalOcean Droplet (Ubuntu 24.04 LTS)
# Run this script on your VPS after SSHing in
#
# Usage: bash install-moltbot.sh
#

set -e  # Exit on any error

echo "================================================"
echo "   MoltBot Installation Script for DigitalOcean"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Running as root. Will create a dedicated 'clawd' user for security."
    IS_ROOT=true
else
    IS_ROOT=false
fi

# Step 1: System Update
echo ""
echo "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_status "System updated"

# Step 2: Install required dependencies
echo ""
echo "Step 2: Installing dependencies..."
sudo apt install -y curl git build-essential
print_status "Dependencies installed"

# Step 3: Install Node.js 22 (required for MoltBot)
echo ""
echo "Step 3: Installing Node.js 22..."
if ! command -v node &> /dev/null || [[ $(node -v | cut -d'v' -f2 | cut -d'.' -f1) -lt 22 ]]; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt install -y nodejs
    print_status "Node.js $(node -v) installed"
else
    print_status "Node.js $(node -v) already installed"
fi

# Verify Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 22 ]; then
    print_error "Node.js 22+ is required. Current version: $(node -v)"
    exit 1
fi

# Step 4: Create clawd user (if running as root)
if [ "$IS_ROOT" = true ]; then
    echo ""
    echo "Step 4: Creating dedicated 'clawd' user..."
    if id "clawd" &>/dev/null; then
        print_warning "User 'clawd' already exists"
    else
        adduser --disabled-password --gecos "" clawd
        usermod -aG sudo clawd
        # Allow clawd to run sudo without password (optional, for convenience)
        echo "clawd ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/clawd
        print_status "User 'clawd' created with sudo privileges"
    fi
    
    # Copy SSH keys to clawd user
    if [ -d "/root/.ssh" ]; then
        mkdir -p /home/clawd/.ssh
        cp /root/.ssh/authorized_keys /home/clawd/.ssh/ 2>/dev/null || true
        chown -R clawd:clawd /home/clawd/.ssh
        chmod 700 /home/clawd/.ssh
        chmod 600 /home/clawd/.ssh/authorized_keys 2>/dev/null || true
        print_status "SSH keys copied to clawd user"
    fi
    
    echo ""
    print_warning "Please run the following to switch to clawd user and continue:"
    echo "    su - clawd"
    echo ""
    echo "Then run: bash ~/install-moltbot-user.sh"
    
    # Copy the user installation script
    cat > /home/clawd/install-moltbot-user.sh << 'USERSCRIPT'
#!/bin/bash
# MoltBot User Installation Script (run as clawd user)

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

echo ""
echo "Installing MoltBot for user: $(whoami)"
echo ""

# Install MoltBot globally
echo "Installing MoltBot..."
npm install -g moltbot@latest
print_status "MoltBot installed: $(moltbot --version 2>/dev/null || echo 'installed')"

# Reload shell to ensure moltbot is in PATH
export PATH="$PATH:$(npm config get prefix)/bin"

echo ""
echo "================================================"
echo "   MoltBot Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Run the onboarding wizard with daemon installation:"
echo "   ${GREEN}moltbot onboard --install-daemon${NC}"
echo ""
echo "2. The wizard will guide you through:"
echo "   - Configuring your LLM provider (Anthropic, OpenAI, etc.)"
echo "   - Setting up your workspace"
echo "   - Connecting chat channels (WhatsApp, Telegram, etc.)"
echo "   - Installing the systemd service for auto-start"
echo ""
echo "3. After setup, you can:"
echo "   - Chat via terminal: ${GREEN}moltbot tui${NC}"
echo "   - Check status: ${GREEN}moltbot gateway status${NC}"
echo "   - View logs: ${GREEN}moltbot logs --follow${NC}"
echo ""
echo "4. To access the web dashboard:"
echo "   From your local machine, create an SSH tunnel:"
echo "   ${YELLOW}ssh -L 18789:127.0.0.1:18789 clawd@YOUR_DROPLET_IP${NC}"
echo "   Then open: http://127.0.0.1:18789"
echo ""

USERSCRIPT
    
    chown clawd:clawd /home/clawd/install-moltbot-user.sh
    chmod +x /home/clawd/install-moltbot-user.sh
    
    exit 0
fi

# If not root, continue with installation
echo ""
echo "Step 4: Installing MoltBot..."
npm install -g moltbot@latest
print_status "MoltBot installed"

echo ""
echo "================================================"
echo "   MoltBot Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Run the onboarding wizard with daemon installation:"
echo "   moltbot onboard --install-daemon"
echo ""
echo "2. The wizard will guide you through:"
echo "   - Configuring your LLM provider (Anthropic, OpenAI, etc.)"
echo "   - Setting up your workspace"
echo "   - Connecting chat channels (WhatsApp, Telegram, etc.)"
echo "   - Installing the systemd service for auto-start"
echo ""
