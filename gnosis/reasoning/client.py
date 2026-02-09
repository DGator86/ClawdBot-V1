"""
LLM Client — Unified interface for OpenAI-compatible API calls.
================================================================
Environment-aware routing:
  1. GenSpark sandbox — uses ~/.genspark_llm.yaml (gpt-5 via proxy)
  2. OpenRouter — OPENAI_API_KEY starts with sk-or-* (free tier available)
  3. VPS / direct OpenAI — uses OPENAI_API_KEY env var (gpt-4o-mini)
  4. Custom endpoint — uses OPENAI_API_KEY + OPENAI_BASE_URL env vars
  5. Offline / no key — falls back to deterministic StubLLM

Detection order:
  a) ~/.genspark_llm.yaml with a RESOLVED api_key → GenSpark proxy
  b) OPENAI_API_KEY with sk-or-* prefix → OpenRouter (free models)
  c) OPENAI_API_KEY + OPENAI_BASE_URL → custom endpoint
  d) OPENAI_API_KEY (sk-*) without BASE_URL → direct OpenAI
  e) No key anywhere → stub mode
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request, error

# ── Constants ──────────────────────────────────────────────────
GENSPARK_PROXY_URL = "https://www.genspark.ai/api/llm_proxy/v1"
OPENAI_DIRECT_URL = "https://api.openai.com/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# Models per environment
GENSPARK_MODEL = "gpt-5"        # GenSpark proxy supports gpt-5 family
OPENAI_MODEL = "gpt-4o-mini"    # Cost-effective default for direct OpenAI
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"  # Best free model


_PLACEHOLDER_PREFIXES = ("your_", "replace", "REPLACE", "xxx", "changeme", "TODO")
_PLACEHOLDER_WORDS = ("your", "replace", "test", "fake", "dummy", "example", "changeme", "todo")

# Real API keys: sk-* (OpenAI ≥51 chars), gsk-* (GenSpark), etc.
_MIN_REAL_KEY_LENGTH = 20


def _is_placeholder(value: str) -> bool:
    """Check if a value looks like a placeholder rather than a real secret."""
    if not value:
        return True
    for prefix in _PLACEHOLDER_PREFIXES:
        if value.startswith(prefix):
            return True
    if value.startswith("${") and value.endswith("}"):
        return True  # Unresolved template
    # Catch "sk-your-new-key", "sk-test-key", etc.
    lower = value.lower()
    if any(word in lower for word in _PLACEHOLDER_WORDS):
        return True
    # Real API keys are typically 20+ characters
    if len(value) < _MIN_REAL_KEY_LENGTH:
        return True
    return False


def _load_dotenv(path: str = None) -> dict:
    """Load key=value pairs from a .env file (no shell expansion).

    Skips placeholder values like 'your_openai_api_key_here'.

    Looks in these locations (first found wins):
      1. Explicit path argument
      2. .env in current working directory
      3. .env next to this source file's project root
      4. /root/ClawdBot-V1/.env  (VPS standard location)
    """
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend([
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), ".env"),
        "/root/ClawdBot-V1/.env",
    ])

    for p in candidates:
        if os.path.isfile(p):
            env = {}
            try:
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if not _is_placeholder(v):
                                env[k] = v
            except Exception:
                pass
            if env:  # Only return if we found real values
                return env
    return {}


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    model: str = ""             # Empty → auto-detect
    api_key: str = ""
    base_url: str = ""          # Empty → auto-detect
    timeout_seconds: int = 60
    max_tokens: int = 4096
    temperature: float = 0.2
    # Retry settings
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    # Metadata: which environment was detected
    _environment: str = "unknown"

    @classmethod
    def from_yaml(cls, path: str = None) -> "LLMConfig":
        """Load config with full environment detection.

        Priority:
          1. ~/.genspark_llm.yaml with resolved API key → GenSpark proxy
          2. OPENAI_API_KEY + OPENAI_BASE_URL env vars → custom endpoint
          3. OPENAI_API_KEY (sk-*) without BASE_URL → direct OpenAI (api.openai.com)
          4. .env file as fallback for env vars
          5. No key → stub mode
        """
        config = cls()

        # ── Try GenSpark YAML ──────────────────────────────────
        yaml_path = path or os.path.join(os.path.expanduser("~"), ".genspark_llm.yaml")
        genspark_key = ""
        genspark_url = ""

        if os.path.exists(yaml_path):
            try:
                import yaml
                with open(yaml_path) as f:
                    raw = yaml.safe_load(f) or {}
                openai_cfg = raw.get("openai", {}) or {}
                api_key_raw = openai_cfg.get("api_key", "")

                # Handle ${GENSPARK_TOKEN} template
                if api_key_raw.startswith("${") and api_key_raw.endswith("}"):
                    env_var = api_key_raw[2:-1]
                    genspark_key = os.environ.get(env_var, "")
                elif api_key_raw and not api_key_raw.startswith("$"):
                    # Literal key in YAML (actually resolved)
                    genspark_key = api_key_raw

                genspark_url = openai_cfg.get("base_url", GENSPARK_PROXY_URL)
            except Exception:
                pass

        # If GenSpark YAML has a real key, use it
        if genspark_key and genspark_key.startswith("gsk-"):
            config.api_key = genspark_key
            config.base_url = genspark_url or GENSPARK_PROXY_URL
            config.model = GENSPARK_MODEL
            config._environment = "genspark"
            return config

        # ── Try environment variables ──────────────────────────
        env_key = os.environ.get("OPENAI_API_KEY", "")
        env_url = os.environ.get("OPENAI_BASE_URL", "")
        env_model = os.environ.get("OPENAI_MODEL", "")

        # ── Try .env file fallback ─────────────────────────────
        if not env_key:
            dotenv = _load_dotenv()
            env_key = dotenv.get("OPENAI_API_KEY", "")
            if not env_url:
                env_url = dotenv.get("OPENAI_BASE_URL", "")
            if not env_model:
                env_model = dotenv.get("OPENAI_MODEL", "")

        if env_key:
            config.api_key = env_key

            # ── OpenRouter auto-detection ──────────────────────
            # sk-or-* keys are OpenRouter tokens
            _is_openrouter_key = env_key.startswith("sk-or-")
            _is_openrouter_url = "openrouter.ai" in env_url if env_url else False

            if _is_openrouter_key or _is_openrouter_url:
                config.base_url = env_url if _is_openrouter_url else OPENROUTER_URL
                config.model = env_model or OPENROUTER_MODEL
                config._environment = "openrouter"
                return config

            # ── Detect key/URL mismatches ──────────────────────
            # - sk-* key + GenSpark URL = misconfiguration → fix to api.openai.com
            _is_sk_key = env_key.startswith("sk-")
            _is_genspark_url = "genspark.ai" in env_url if env_url else False

            if env_url and not (_is_sk_key and _is_genspark_url):
                # Explicit base URL that matches the key type → custom endpoint
                config.base_url = env_url
                config.model = env_model or config.model or OPENAI_MODEL
                config._environment = "custom"
            elif _is_sk_key:
                # OpenAI key (sk-*): always route to api.openai.com
                # This covers: no URL set, OR URL was GenSpark (mismatch)
                config.base_url = OPENAI_DIRECT_URL
                config.model = env_model or OPENAI_MODEL
                config._environment = "openai_direct"
                if _is_genspark_url:
                    import sys
                    print(f"[LLM] WARNING: OPENAI_API_KEY (sk-*) with GenSpark BASE_URL — "
                          f"overriding to {OPENAI_DIRECT_URL}. "
                          f"Unset OPENAI_BASE_URL or use a gsk-* key for GenSpark.",
                          file=sys.stderr)
            else:
                # Non-sk key without URL → could be GenSpark token set via env
                config.base_url = GENSPARK_PROXY_URL
                config.model = env_model or GENSPARK_MODEL
                config._environment = "genspark_env"

            return config

        # If genspark YAML existed but had an unresolved template → still try it
        # (the proxy might accept the request in certain sandbox setups)
        if genspark_url:
            config.api_key = genspark_key  # may be empty
            config.base_url = genspark_url
            config.model = GENSPARK_MODEL
            config._environment = "genspark_unresolved"
            return config

        # ── No key found anywhere ──────────────────────────────
        config._environment = "stub"
        return config

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url)


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: str = ""
    parsed: Optional[Dict[str, Any]] = None
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    error: Optional[str] = None
    is_stub: bool = False


class LLMClient:
    """Unified LLM client with environment-aware routing and stub fallback."""

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig.from_yaml()
        self._use_stub = not self.config.is_configured

        import sys
        if self._use_stub:
            print(f"[LLM] No API key found; using stub responses "
                  f"(env={self.config._environment})", file=sys.stderr)
        else:
            print(f"[LLM] Configured: env={self.config._environment}, "
                  f"model={self.config.model}, "
                  f"base_url={self.config.base_url[:50]}...", file=sys.stderr)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        response_format: str = None,  # "json" for JSON mode
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            system_prompt: System instructions
            user_prompt: User message (the data + question)
            temperature: Override default temperature
            max_tokens: Override default max tokens
            response_format: "json" to request JSON output

        Returns:
            LLMResponse with content and optional parsed JSON
        """
        if self._use_stub:
            return self._stub_response(system_prompt, user_prompt)

        return self._api_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            response_format=response_format,
        )

    def _api_call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: str = None,
    ) -> LLMResponse:
        """Make actual API call to OpenAI-compatible endpoint."""
        t0 = time.time()
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")

        for attempt in range(self.config.max_retries + 1):
            try:
                req = request.Request(
                    endpoint,
                    data=body,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with request.urlopen(
                    req, timeout=self.config.timeout_seconds
                ) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                elapsed = (time.time() - t0) * 1000

                # Try to parse as JSON
                parsed = None
                try:
                    parsed = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    # Try extracting JSON from markdown code block
                    if "```json" in content:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                        try:
                            parsed = json.loads(json_str)
                        except (json.JSONDecodeError, IndexError):
                            pass
                    elif "```" in content:
                        json_str = content.split("```")[1].split("```")[0].strip()
                        try:
                            parsed = json.loads(json_str)
                        except (json.JSONDecodeError, IndexError):
                            pass

                return LLMResponse(
                    content=content,
                    parsed=parsed,
                    model=data.get("model", self.config.model),
                    usage={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                    elapsed_ms=round(elapsed, 1),
                )

            except error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    pass
                # Retry on transient errors
                if attempt < self.config.max_retries and e.code in (429, 500, 502, 503):
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))
                    continue
                return LLMResponse(
                    error=f"HTTP {e.code}: {err_body[:500]}",
                    elapsed_ms=round((time.time() - t0) * 1000, 1),
                )
            except Exception as e:
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))
                    continue
                return LLMResponse(
                    error=str(e),
                    elapsed_ms=round((time.time() - t0) * 1000, 1),
                )

        return LLMResponse(
            error="max retries exceeded",
            elapsed_ms=round((time.time() - t0) * 1000, 1),
        )

    def _stub_response(
        self, system_prompt: str, user_prompt: str
    ) -> LLMResponse:
        """Generate a deterministic stub response for offline testing."""
        # Extract key signals from the user prompt to build meaningful stub
        analysis = {
            "summary": "Stub analysis (no LLM API configured)",
            "signal_quality": "POOR",
            "confidence_assessment": "Cannot assess without LLM — using rule-based fallback",
            "regime_interpretation": "Rule-based: see KPCOFGS and regime_gate outputs",
            "trade_suggestion": {
                "action": "HOLD",
                "reason": "No LLM available for extrapolation; defer to regime gate decision",
                "position_size_pct": 0.0,
            },
            "risk_notes": [
                "LLM reasoning unavailable — using conservative defaults",
                "Rely on regime_gate action field for trade/skip decision",
            ],
            "extrapolations": [],
            "next_steps": [
                "Configure LLM API key for full reasoning capabilities",
                "Option 1: Set OPENAI_API_KEY in .env or environment",
                "Option 2: Inject key via GenSpark UI → ~/.genspark_llm.yaml",
            ],
            "is_stub": True,
        }
        return LLMResponse(
            content=json.dumps(analysis, indent=2),
            parsed=analysis,
            model="stub",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            elapsed_ms=0.1,
            is_stub=True,
        )
