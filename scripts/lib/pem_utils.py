#!/usr/bin/env python3
"""
PEM Key Utilities — Shared across all Kalshi-related scripts
==============================================================
Handles all PEM deserialization edge cases:
  1. Literal \\n instead of real newlines (systemd EnvironmentFile)
  2. Single-line PEM with headers but no line breaks
  3. Raw base64 with NO headers at all
  4. Double-quoted PEM in .env files
  5. Mixed \\r\\n line endings
  6. Truncated PEM from dotenv parsers

Usage:
    from scripts.lib.pem_utils import fix_pem, load_pem_key, load_env_files
"""
import os
import re
from typing import Optional

# Keys that must be force-overwritten from .env (systemd mangles multi-line)
FORCE_OVERWRITE_KEYS = {"KALSHI_PRIVATE_KEY"}

# Standard .env search paths (in priority order)
ENV_SEARCH_PATHS = [
    "/root/ClawdBot-V1/.env",
    "/root/Yoshi-Bot/.env",
    "/home/root/ClawdBot-V1/.env",
    "/home/root/Yoshi-Bot/.env",
    os.path.expanduser("~/.env"),
]

# Standard PEM file search paths
PEM_SEARCH_PATHS = [
    os.path.expanduser("~/.kalshi/private_key.pem"),
    "/root/.kalshi/private_key.pem",
    "/home/root/.kalshi/private_key.pem",
]


def fix_pem(raw: str) -> str:
    """
    Normalize a PEM key that may have been mangled by env var storage.

    Handles all known mangling cases:
      - Literal \\n instead of real newlines
      - Single-line PEM with headers but no line breaks
      - Raw base64 with spaces instead of newlines
      - Raw base64 with no whitespace at all
      - Double-quoted wrappers
      - Mixed \\r\\n line endings

    Returns a properly formatted PEM string.
    """
    if not raw:
        return raw

    # Strip outer quotes
    raw = raw.strip()
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]

    # Replace literal \n with real newlines
    if "\\n" in raw:
        raw = raw.replace("\\n", "\n")

    # Normalize line endings
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # If it has PEM headers but is mangled onto one/two lines
    if "-----BEGIN" in raw and raw.count("\n") <= 2:
        m = re.search(
            r"-----BEGIN ([A-Z ]+)-----\s*(.*?)\s*-----END ([A-Z ]+)-----",
            raw, re.DOTALL
        )
        if m:
            key_type = m.group(1)
            body = m.group(2).replace(" ", "").replace("\n", "").replace("\t", "")
            lines = [body[i:i+64] for i in range(0, len(body), 64)]
            raw = (
                f"-----BEGIN {key_type}-----\n"
                + "\n".join(lines)
                + f"\n-----END {key_type}-----"
            )
        return raw.strip()

    # If it has PEM headers and looks properly formatted, just clean up
    if "-----BEGIN" in raw:
        return raw.strip()

    # NO PEM headers -- raw base64 (possibly with spaces instead of newlines)
    body = re.sub(r"\s+", "", raw)
    if len(body) > 100 and re.match(r"^[A-Za-z0-9+/=]+$", body):
        lines = [body[i:i+64] for i in range(0, len(body), 64)]
        raw = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            + "\n".join(lines)
            + "\n-----END RSA PRIVATE KEY-----"
        )

    return raw.strip()


def fix_pem_in_env_file(env_path: str) -> bool:
    """
    Fix PEM formatting in a .env file in-place.
    Returns True if changes were made.
    """
    if not os.path.isfile(env_path):
        return False

    content = open(env_path).read()

    def _fix_match(m):
        val = m.group(1)
        val = val.replace("\\n", "\n")
        return f'KALSHI_PRIVATE_KEY="{val}"'

    new_content = re.sub(
        r'KALSHI_PRIVATE_KEY="(.+?)"', _fix_match, content, flags=re.DOTALL
    )

    if new_content != content:
        with open(env_path, "w") as f:
            f.write(new_content)
        return True
    return False


def fix_pem_file(pem_path: str) -> bool:
    """
    Fix PEM formatting in a standalone .pem file.
    Returns True if changes were made.
    """
    if not os.path.isfile(pem_path):
        return False

    data = open(pem_path).read()
    fixed = fix_pem(data)
    if fixed != data:
        with open(pem_path, "w") as f:
            f.write(fixed)
        return True
    return False


def load_env_file(path: str, force_keys: set = None):
    """
    Parse a .env file and set vars in os.environ.

    Args:
        path: Path to the .env file
        force_keys: Set of keys that should overwrite existing env vars
                   (default: KALSHI_PRIVATE_KEY)
    """
    if force_keys is None:
        force_keys = FORCE_OVERWRITE_KEYS

    if not os.path.isfile(path):
        return

    try:
        with open(path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, _, val = raw.partition("=")
                key = key.strip()
                val = val.strip()

                # Handle multi-line PEM values wrapped in quotes
                if val.startswith('"') and not val.endswith('"'):
                    lines = [val[1:]]  # strip opening quote
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
                    if key in force_keys:
                        os.environ[key] = val
                    else:
                        os.environ.setdefault(key, val)
    except Exception:
        pass


def load_env_files(paths: list[str] = None, force_keys: set = None):
    """Load multiple .env files in order."""
    if paths is None:
        paths = ENV_SEARCH_PATHS
    for path in paths:
        if os.path.isfile(path):
            load_env_file(path, force_keys)


def load_pem_key(key_id_var: str = "KALSHI_KEY_ID",
                 key_var: str = "KALSHI_PRIVATE_KEY"):
    """
    Load and return a validated RSA private key for Kalshi API auth.

    Searches environment vars, .env files, and PEM files on disk.
    Applies PEM normalization before loading.

    Returns:
        tuple: (key_id, private_key_object)

    Raises:
        ValueError: If credentials are not found or key fails to parse.
    """
    # Load env files if key not already set
    if not os.environ.get(key_id_var):
        load_env_files()

    key_id = os.environ.get(key_id_var, "").strip()
    if not key_id:
        raise ValueError(f"{key_id_var} not set in environment or .env files")

    pk_raw = os.environ.get(key_var, "").strip()

    # Try loading from PEM file if env var is empty
    if not pk_raw:
        for pk_path in PEM_SEARCH_PATHS:
            if os.path.isfile(pk_path):
                with open(pk_path) as f:
                    pk_raw = f.read().strip()
                break

    if not pk_raw:
        raise ValueError(
            f"{key_var} not set and no PEM file found at: "
            + ", ".join(PEM_SEARCH_PATHS)
        )

    # Apply PEM normalization
    pk_raw = fix_pem(pk_raw)

    # Load the RSA private key
    try:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        private_key = load_pem_private_key(pk_raw.encode(), password=None)
    except ImportError:
        raise ImportError(
            "cryptography package not installed. Run: pip3 install cryptography"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to parse PEM key ({len(pk_raw)} bytes): {e}\n"
            f"First 50 chars: {pk_raw[:50]}..."
        )

    return key_id, private_key


def fix_all_pem_files(verbose: bool = True) -> int:
    """
    Fix PEM formatting in all known locations.
    Returns the number of files fixed.
    """
    fixed = 0

    # Fix .env files
    for env_path in ENV_SEARCH_PATHS:
        if fix_pem_in_env_file(env_path):
            if verbose:
                print(f"  Fixed PEM in: {env_path}")
            fixed += 1

    # Fix standalone PEM files
    for pem_path in PEM_SEARCH_PATHS:
        if fix_pem_file(pem_path):
            if verbose:
                print(f"  Fixed PEM: {pem_path}")
            fixed += 1

    return fixed


if __name__ == "__main__":
    print("PEM Key Utilities — fixing all known PEM files...")
    n = fix_all_pem_files(verbose=True)
    print(f"\nFixed {n} file(s).")

    # Validate key loading
    try:
        key_id, pk = load_pem_key()
        print(f"Key loaded successfully: {key_id[:12]}...")
    except Exception as e:
        print(f"Key loading failed: {e}")
