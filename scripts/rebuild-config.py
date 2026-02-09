#!/usr/bin/env python3
"""
Rebuild moltbot.json — Model selection + gateway config
=========================================================
Writes the moltbot.json config to both ~/.clawdbot and ~/.moltbot,
with intelligent model selection based on available API keys.

Model priority:
  1. Google Gemini 3 Flash (if GOOGLE_API_KEY set)
  2. OpenAI GPT-4o-mini → GPT-4o fallback (if OPENAI_API_KEY set)
  3. Anthropic Claude (if ANTHROPIC_API_KEY set)

Gateway configuration:
  - Port 18789, loopback binding
  - Token-based auth (auto-generated if not provided)
  - Telegram channel with validated bot token
  - Yoshi-trading skill loaded from /root/ClawdBot-V1/skills

Usage:
    python3 scripts/rebuild-config.py                    # auto-detect everything
    python3 scripts/rebuild-config.py --model openai     # force OpenAI
    python3 scripts/rebuild-config.py --port 18789       # custom port
    python3 scripts/rebuild-config.py --dry-run          # show config without writing
"""
import argparse
import json
import os
import secrets
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.lib.env_sync import parse_env_file, ENV_FILES
except ImportError:
    # Fallback if run standalone
    ENV_FILES = [
        "/root/ClawdBot-V1/.env",
        "/root/Yoshi-Bot/.env",
        "/root/.env",
    ]
    def parse_env_file(path):
        result = {}
        if not os.path.isfile(path):
            return result
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip().strip('"').strip("'")
        return result


# Config output paths
CONFIG_PATHS = [
    os.path.expanduser("~/.clawdbot/moltbot.json"),
    os.path.expanduser("~/.moltbot/moltbot.json"),
]

# Skills directory (try both VPS and sandbox paths)
SKILLS_DIRS = [
    "/root/ClawdBot-V1/skills",
    "/home/root/ClawdBot-V1/skills",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills"),
]

# Model configurations
MODELS = {
    "google": {
        "primary": "google/gemini-3-flash-preview",
        "fallbacks": ["google/gemini-2.5-flash", "google/gemini-2.5-flash-lite"],
        "requires_key": "GOOGLE_API_KEY",
        "label": "Gemini 3 Flash Preview",
    },
    "openai": {
        "primary": "openai/gpt-4o-mini",
        "fallbacks": ["openai/gpt-4o"],
        "requires_key": "OPENAI_API_KEY",
        "label": "GPT-4o-mini",
    },
    "anthropic": {
        "primary": "anthropic/claude-3-5-sonnet",
        "fallbacks": ["anthropic/claude-3-haiku"],
        "requires_key": "ANTHROPIC_API_KEY",
        "label": "Claude 3.5 Sonnet",
    },
}

# Model selection priority
MODEL_PRIORITY = ["google", "openai", "anthropic"]


def collect_env_vars() -> dict:
    """Collect all env vars from .env files and os.environ."""
    env = {}

    # Load from .env files
    for path in ENV_FILES:
        if os.path.isfile(path):
            parsed = parse_env_file(path)
            for k, v in parsed.items():
                if v and not v.startswith("your_"):
                    env.setdefault(k, v)

    # Override with actual environment
    for key in ["GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN",
                 "CLAWDBOT_GATEWAY_TOKEN"]:
        val = os.environ.get(key, "").strip()
        if val:
            env[key] = val

    # GEMINI_API_KEY is an alias for GOOGLE_API_KEY
    if "GEMINI_API_KEY" in env and "GOOGLE_API_KEY" not in env:
        env["GOOGLE_API_KEY"] = env["GEMINI_API_KEY"]

    return env


def select_model(env: dict, force: str = None) -> dict:
    """
    Select the best available model based on API keys.

    Args:
        env: Collected environment variables
        force: Force a specific provider ("google", "openai", "anthropic")

    Returns:
        dict with primary, fallbacks, label
    """
    if force and force in MODELS:
        model = MODELS[force]
        key = env.get(model["requires_key"], "")
        if not key:
            print(f"  WARNING: Forced {force} but {model['requires_key']} not set")
        return model

    # Auto-detect based on available keys
    for provider in MODEL_PRIORITY:
        model = MODELS[provider]
        key = env.get(model["requires_key"], "")
        if key:
            return model

    # No keys found — default to OpenAI (user will need to set key)
    print("  WARNING: No API keys found. Defaulting to OpenAI (key required).")
    return MODELS["openai"]


def find_skills_dir() -> str:
    """Find the skills directory."""
    for d in SKILLS_DIRS:
        if os.path.isdir(d):
            return d
    # Fallback
    return SKILLS_DIRS[0]


def build_config(env: dict,
                 model: dict,
                 port: int = 18789,
                 gateway_token: str = None) -> dict:
    """Build the moltbot.json configuration."""
    telegram_token = env.get("TELEGRAM_BOT_TOKEN", "")
    if not gateway_token:
        gateway_token = env.get("CLAWDBOT_GATEWAY_TOKEN", "")
    if not gateway_token:
        gateway_token = secrets.token_hex(32)

    skills_dir = find_skills_dir()

    config = {
        "agents": {
            "defaults": {
                "workspace": "~/clawd",
                "model": {
                    "primary": model["primary"],
                    "fallbacks": model["fallbacks"],
                },
                "thinkingDefault": "low",
            },
            "list": [{
                "id": "main",
                "default": True,
                "identity": {
                    "name": "ClawdBot",
                    "theme": "crypto trading assistant",
                    "emoji": "\U0001f916",
                },
            }],
        },
        "gateway": {
            "mode": "local",
            "port": port,
            "bind": "loopback",
            "auth": {
                "mode": "token",
                "token": gateway_token,
            },
        },
        "channels": {
            "telegram": {
                "enabled": bool(telegram_token),
                "botToken": telegram_token,
                "dmPolicy": "open",
                "allowFrom": ["*"],
            },
        },
        "skills": {
            "load": {
                "extraDirs": [skills_dir],
            },
        },
    }

    # Add provider-specific env passthrough
    google_key = env.get("GOOGLE_API_KEY", "")
    if google_key:
        config["env"] = {
            "GOOGLE_API_KEY": google_key,
            "GEMINI_API_KEY": google_key,
        }

    return config, gateway_token


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base.
    Override values win for non-dict fields.
    For dicts, recurse. For lists, override wins.
    """
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def write_config(config: dict, paths: list[str] = None,
                 verbose: bool = True, merge: bool = False) -> list[str]:
    """
    Write moltbot.json to all config paths.

    If merge=True, reads the existing config first and deep-merges our
    fields on top.  This preserves moltbot-managed keys (auth bindings,
    plugin state, agent routing) that ``moltbot agents add`` writes.

    Returns list of paths written.
    """
    if paths is None:
        paths = CONFIG_PATHS

    written = []
    for cfg_path in paths:
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)

        if merge and os.path.isfile(cfg_path):
            try:
                with open(cfg_path) as f:
                    existing = json.load(f)
                final = _deep_merge(existing, config)
                if verbose:
                    print(f"  Merged: {cfg_path}")
            except Exception:
                final = config
                if verbose:
                    print(f"  Written (merge failed, fresh): {cfg_path}")
        else:
            final = config
            if verbose:
                print(f"  Written: {cfg_path}")

        with open(cfg_path, "w") as f:
            json.dump(final, f, indent=2)
        os.chmod(cfg_path, 0o600)  # restrict permissions (contains tokens)
        written.append(cfg_path)

    return written


def verify_config(verbose: bool = True) -> bool:
    """
    Verify that the config was written correctly.
    Returns True if all configs are valid.
    """
    ok = True
    for cfg_path in CONFIG_PATHS:
        if not os.path.isfile(cfg_path):
            if verbose:
                print(f"  MISSING: {cfg_path}")
            ok = False
            continue

        try:
            with open(cfg_path) as f:
                config = json.load(f)

            # Check required fields
            checks = [
                ("agents.defaults.model.primary", config.get("agents", {})
                    .get("defaults", {}).get("model", {}).get("primary")),
                ("gateway.port", config.get("gateway", {}).get("port")),
                ("gateway.auth.token", config.get("gateway", {})
                    .get("auth", {}).get("token")),
                ("channels.telegram.botToken", config.get("channels", {})
                    .get("telegram", {}).get("botToken")),
            ]

            for name, val in checks:
                if not val:
                    if verbose:
                        print(f"  WARN: {name} is empty in {cfg_path}")
                    ok = False

            if verbose:
                model = config["agents"]["defaults"]["model"]["primary"]
                token_last8 = config["channels"]["telegram"]["botToken"][-8:] \
                    if config["channels"]["telegram"]["botToken"] else "NONE"
                print(f"  OK: {cfg_path} (model={model}, token=...{token_last8})")

        except Exception as e:
            if verbose:
                print(f"  ERROR: {cfg_path}: {e}")
            ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild moltbot.json with intelligent model selection"
    )
    parser.add_argument("--model", choices=["google", "openai", "anthropic"],
                        help="Force a specific model provider")
    parser.add_argument("--port", type=int, default=18789,
                        help="Gateway port (default: 18789)")
    parser.add_argument("--gateway-token", type=str, default=None,
                        help="Gateway auth token (auto-generated if not set)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show config without writing")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing config files")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing config (preserves moltbot agent auth/bindings)")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.verify:
        if verbose:
            print("Verifying moltbot.json configs...\n")
        ok = verify_config(verbose=verbose)
        sys.exit(0 if ok else 1)

    if verbose:
        print("=== Rebuild moltbot.json ===\n")

    # Collect env vars
    env = collect_env_vars()
    if verbose:
        for key in ["GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            val = env.get(key, "")
            status = f"...{val[-8:]}" if val else "NOT SET"
            print(f"  {key}: {status}")
        tg = env.get("TELEGRAM_BOT_TOKEN", "")
        print(f"  TELEGRAM_BOT_TOKEN: {'...' + tg[-8:] if tg else 'NOT SET'}")

    # Select model
    model = select_model(env, force=args.model)
    if verbose:
        print(f"\n  Selected model: {model['label']} ({model['primary']})")
        print(f"  Fallbacks: {', '.join(model['fallbacks'])}")

    # Build config
    config, gw_token = build_config(
        env, model, port=args.port, gateway_token=args.gateway_token
    )

    if args.dry_run:
        print(f"\n  Config (dry-run):\n")
        print(json.dumps(config, indent=2))
        return

    # Write config
    if verbose:
        print(f"\n  Writing configs...")
    written = write_config(config, verbose=verbose, merge=args.merge)

    # Verify
    if verbose:
        print(f"\n  Verifying...")
    verify_config(verbose=verbose)

    if verbose:
        print(f"\n  Gateway token: ...{gw_token[-8:]}")
        print(f"  Config written to {len(written)} location(s)")

    # Export gateway token for systemd
    print(f"\nGATEWAY_TOKEN={gw_token}")


if __name__ == "__main__":
    main()
