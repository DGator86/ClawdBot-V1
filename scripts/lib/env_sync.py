#!/usr/bin/env python3
"""
Environment Synchronization — Unify tokens across all .env files
=================================================================
Finds and validates the working Telegram bot token, then synchronizes
it across all known .env files. Removes dead/revoked tokens.

Handles:
  1. Multiple .env files with different tokens
  2. Dead/revoked tokens (validates against Telegram getMe API)
  3. Conflicting OPENAI_API_KEY / GOOGLE_API_KEY across files
  4. Missing required keys

Usage:
    python3 scripts/lib/env_sync.py                  # dry-run
    python3 scripts/lib/env_sync.py --apply           # write changes
    python3 scripts/lib/env_sync.py --apply --verbose  # with details
"""
import json
import os
import re
import sys
from typing import Optional
from urllib import request, error

# All known .env file locations
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_FILES = [
    os.path.join(_PROJECT_ROOT, ".env"),
    "/root/ClawdBot-V1/.env",
    "/root/Yoshi-Bot/.env",
    "/root/.env",
    "/home/root/ClawdBot-V1/.env",
    "/home/root/Yoshi-Bot/.env",
]

# Keys that should be synchronized (same value across all files)
SYNC_KEYS = {
    "TELEGRAM_BOT_TOKEN",
    "KALSHI_KEY_ID",
}

# Keys to collect (take first non-empty value found)
COLLECT_KEYS = {
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "KALSHI_PRIVATE_KEY",
    "DIGITALOCEAN_TOKEN",
    "TELEGRAM_CHAT_ID",
    "ELEVENLABS_API_KEY",
}


def validate_telegram_token(token: str, timeout: int = 10) -> Optional[str]:
    """
    Validate a Telegram bot token against the getMe API.

    Returns:
        Bot username if valid, None if invalid/dead.
    """
    if not token or len(token) < 20:
        return None

    try:
        req = request.Request(
            f"https://api.telegram.org/bot{token}/getMe",
            headers={"User-Agent": "ClawdBot/1.0"},
        )
        with request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            if data.get("ok"):
                return data["result"].get("username", "unknown")
    except Exception:
        pass
    return None


def parse_env_file(path: str) -> dict:
    """Parse a .env file into a key-value dict."""
    result = {}
    if not os.path.isfile(path):
        return result

    try:
        with open(path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, _, val = raw.partition("=")
                key = key.strip()
                val = val.strip()

                # Handle multi-line PEM values
                if val.startswith('"') and not val.endswith('"'):
                    lines = [val[1:]]
                    for extra in f:
                        extra = extra.rstrip("\n")
                        if extra.endswith('"'):
                            lines.append(extra[:-1])
                            break
                        lines.append(extra)
                    val = "\n".join(lines)
                else:
                    val = val.strip('"').strip("'")

                if key and val:
                    result[key] = val
    except Exception:
        pass
    return result


def update_env_key(path: str, key: str, value: str) -> bool:
    """
    Update or add a key in a .env file.
    Preserves comments and ordering. Returns True if changed.
    """
    if not os.path.isfile(path):
        return False

    with open(path) as f:
        lines = f.readlines()

    found = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            # For multi-line values like PEM, skip continuation lines
            if '-----BEGIN' in value or '\n' in value:
                new_lines.append(f'{key}="{value}"\n')
            else:
                new_lines.append(f"{key}={value}\n")
            found = True
            # Skip old multi-line continuation
            continue
        new_lines.append(line)

    if not found:
        new_lines.append(f"\n{key}={value}\n")

    with open(path, "w") as f:
        f.writelines(new_lines)
    return True


def remove_env_key(path: str, key: str) -> bool:
    """Remove a key from a .env file. Returns True if removed."""
    if not os.path.isfile(path):
        return False

    with open(path) as f:
        lines = f.readlines()

    new_lines = []
    removed = False
    skip_multiline = False
    for line in lines:
        stripped = line.strip()
        if skip_multiline:
            if stripped.endswith('"'):
                skip_multiline = False
                removed = True
            continue
        if stripped.startswith(f"{key}="):
            val = stripped.partition("=")[2].strip()
            if val.startswith('"') and not val.endswith('"'):
                skip_multiline = True
            removed = True
            continue
        new_lines.append(line)

    if removed:
        with open(path, "w") as f:
            f.writelines(new_lines)
    return removed


def discover_env_state(verbose: bool = False) -> dict:
    """
    Scan all .env files and collect the state of all keys.

    Returns dict with:
        - files: {path: {key: value, ...}}
        - conflicts: {key: [{path, value}, ...]}
        - working_telegram_token: str or None
        - dead_tokens: [(path, token_last8), ...]
        - unified: {key: value} (resolved canonical values)
    """
    state = {
        "files": {},
        "conflicts": {},
        "working_telegram_token": None,
        "working_telegram_bot": None,
        "dead_tokens": [],
        "unified": {},
    }

    # Parse all files
    for path in ENV_FILES:
        if os.path.isfile(path):
            state["files"][path] = parse_env_file(path)
            if verbose:
                n = len(state["files"][path])
                print(f"  Found: {path} ({n} keys)")

    # Find and validate Telegram tokens
    seen_tokens = set()
    for path, env in state["files"].items():
        token = env.get("TELEGRAM_BOT_TOKEN", "")
        if token and token not in seen_tokens and not token.startswith("your_"):
            seen_tokens.add(token)
            last8 = token[-8:]

            if verbose:
                print(f"  Testing token ...{last8} from {path}...", end=" ")

            bot_name = validate_telegram_token(token)
            if bot_name:
                if verbose:
                    print(f"VALID (@{bot_name})")
                if not state["working_telegram_token"]:
                    state["working_telegram_token"] = token
                    state["working_telegram_bot"] = bot_name
            else:
                if verbose:
                    print("DEAD")
                state["dead_tokens"].append((path, last8))

    # Collect all keys (take first non-empty across files)
    for key in COLLECT_KEYS:
        for path, env in state["files"].items():
            val = env.get(key, "")
            if val and not val.startswith("your_"):
                state["unified"][key] = val
                break

    # Override with validated token
    if state["working_telegram_token"]:
        state["unified"]["TELEGRAM_BOT_TOKEN"] = state["working_telegram_token"]

    # Detect conflicts
    for key in SYNC_KEYS:
        values = {}
        for path, env in state["files"].items():
            val = env.get(key, "")
            if val and not val.startswith("your_"):
                values.setdefault(val, []).append(path)
        if len(values) > 1:
            conflicts = []
            for val, paths in values.items():
                for p in paths:
                    conflicts.append({"path": p, "value": f"...{val[-8:]}"})
            state["conflicts"][key] = conflicts

    return state


def sync_env_files(apply: bool = False, verbose: bool = True) -> dict:
    """
    Discover, validate, and synchronize all .env files.

    Args:
        apply: If True, write changes. If False, dry-run only.
        verbose: Print details.

    Returns:
        Summary dict with actions taken.
    """
    if verbose:
        mode = "APPLY" if apply else "DRY-RUN"
        print(f"\n=== Environment Sync ({mode}) ===\n")

    state = discover_env_state(verbose=verbose)
    actions = {"updated": [], "dead_removed": [], "errors": []}

    # Report findings
    if verbose:
        if state["working_telegram_token"]:
            t = state["working_telegram_token"]
            print(f"\n  Working token: ...{t[-8:]} (@{state['working_telegram_bot']})")
        else:
            print("\n  WARNING: No working Telegram token found!")

        if state["dead_tokens"]:
            print(f"  Dead tokens: {len(state['dead_tokens'])}")
            for path, last8 in state["dead_tokens"]:
                print(f"    ...{last8} in {path}")

        if state["conflicts"]:
            print(f"\n  Conflicts detected:")
            for key, entries in state["conflicts"].items():
                print(f"    {key}:")
                for e in entries:
                    print(f"      {e['value']} in {e['path']}")

    # Synchronize
    if apply:
        working_token = state["working_telegram_token"]
        if working_token:
            for path in state["files"]:
                env = state["files"][path]
                current = env.get("TELEGRAM_BOT_TOKEN", "")
                if current != working_token:
                    update_env_key(path, "TELEGRAM_BOT_TOKEN", working_token)
                    actions["updated"].append(
                        f"TELEGRAM_BOT_TOKEN in {path}"
                    )
                    if verbose:
                        print(f"  Updated: TELEGRAM_BOT_TOKEN in {path}")

        # Sync other collected keys to the primary .env
        primary = None
        for p in ENV_FILES:
            if os.path.isfile(p):
                primary = p
                break

        if primary:
            for key, val in state["unified"].items():
                if key == "TELEGRAM_BOT_TOKEN":
                    continue  # handled above
                current = state["files"].get(primary, {}).get(key, "")
                if not current or current.startswith("your_"):
                    update_env_key(primary, key, val)
                    actions["updated"].append(f"{key} in {primary}")
                    if verbose:
                        print(f"  Added: {key} to {primary}")

    summary = {
        "working_token": bool(state["working_telegram_token"]),
        "bot_name": state["working_telegram_bot"],
        "dead_tokens": len(state["dead_tokens"]),
        "conflicts": len(state["conflicts"]),
        "files_scanned": len(state["files"]),
        "actions": actions,
        "unified_keys": list(state["unified"].keys()),
    }

    if verbose:
        print(f"\n  Summary: {summary['files_scanned']} files scanned, "
              f"{len(actions['updated'])} updated, "
              f"{summary['dead_tokens']} dead tokens found")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unify Telegram tokens and env vars across .env files"
    )
    parser.add_argument("--apply", action="store_true",
                        help="Write changes (default: dry-run)")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    result = sync_env_files(apply=args.apply, verbose=args.verbose)

    if not args.apply and result["conflicts"] > 0:
        print("\n  Run with --apply to synchronize files.")
    sys.exit(0 if result["working_token"] else 1)
