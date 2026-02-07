"""
LLM Client — Unified interface for OpenAI-compatible API calls.
================================================================
Supports:
  - GenSpark proxy (gpt-5 via ~/.genspark_llm.yaml)
  - Direct OpenAI API
  - Stub/local fallback for testing without API keys

The client loads credentials from:
  1. ~/.genspark_llm.yaml (preferred, auto-configured in sandbox)
  2. OPENAI_API_KEY + OPENAI_BASE_URL env vars
  3. Falls back to StubLLM for offline operation
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request, error


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    model: str = "gpt-5"
    api_key: str = ""
    base_url: str = "https://www.genspark.ai/api/llm_proxy/v1"
    timeout_seconds: int = 60
    max_tokens: int = 4096
    temperature: float = 0.2
    # Retry settings
    max_retries: int = 2
    retry_delay_seconds: float = 1.0

    @classmethod
    def from_yaml(cls, path: str = None) -> "LLMConfig":
        """Load config from ~/.genspark_llm.yaml or given path."""
        if path is None:
            path = os.path.join(os.path.expanduser("~"), ".genspark_llm.yaml")

        config = cls()

        if os.path.exists(path):
            try:
                import yaml
                with open(path) as f:
                    raw = yaml.safe_load(f) or {}
                openai_cfg = raw.get("openai", {}) or {}
                api_key_raw = openai_cfg.get("api_key", "")
                # Handle ${GENSPARK_TOKEN} template
                if api_key_raw.startswith("${") and api_key_raw.endswith("}"):
                    env_var = api_key_raw[2:-1]
                    config.api_key = os.environ.get(env_var, "")
                else:
                    config.api_key = api_key_raw
                config.base_url = openai_cfg.get("base_url", config.base_url)
            except Exception:
                pass

        # Fall back to env vars
        if not config.api_key:
            config.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not config.base_url or config.base_url == cls.base_url:
            env_url = os.environ.get("OPENAI_BASE_URL", "")
            if env_url:
                config.base_url = env_url

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
    """Unified LLM client with fallback to stub."""

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig.from_yaml()
        self._use_stub = not self.config.is_configured

        if self._use_stub:
            import sys
            print("[LLM] No API key found; using stub responses", file=sys.stderr)

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
                "Check ~/.genspark_llm.yaml or set OPENAI_API_KEY env var",
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
