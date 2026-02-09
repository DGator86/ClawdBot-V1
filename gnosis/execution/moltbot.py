"""Moltbot orchestration layer for AI-guided trade planning.

This module bridges Yoshi forecasts into external services and optional
LLM-based reasoning to produce a trade plan payload.

Environment-aware: uses gnosis.reasoning.client for LLM routing so
GenSpark sandbox, direct OpenAI, and custom endpoints all work.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from urllib import request

import yaml

# Re-use the unified LLM config from reasoning layer
from gnosis.reasoning.client import (
    LLMConfig,
    GENSPARK_PROXY_URL,
    OPENAI_DIRECT_URL,
    OPENROUTER_URL,
    GENSPARK_MODEL,
    OPENAI_MODEL,
    OPENROUTER_MODEL,
)


@dataclass
class AIProviderConfig:
    """Configuration for the AI provider."""

    provider: str = "openai"
    model: str = ""  # Auto-detected from environment
    api_key_env: str = "OPENAI_API_KEY"
    endpoint: str = ""  # Auto-detected from environment
    timeout_seconds: int = 60
    system_prompt: str = (
        "You are Moltbot, the reasoning core of a dual-bot crypto trading system. "
        "You receive forecast data from ClawdBot (14-paradigm ensemble) and "
        "validation data from Yoshi (walk-forward, KPCOFGS regimes, backtest). "
        "Use this data to propose trade plans. Return JSON only. "
        "Be honest about signal quality. Never hallucinate edges."
    )

    def resolve_endpoint(self) -> str:
        """Resolve the actual endpoint to use based on environment.

        If self.endpoint is explicitly set to a non-GenSpark URL, use it.
        Otherwise, auto-detect like LLMConfig.from_yaml().
        """
        if self.endpoint and self.endpoint not in (
            f"{GENSPARK_PROXY_URL}/chat/completions",
            f"{OPENAI_DIRECT_URL}/chat/completions",
        ):
            # User provided a custom endpoint — use it
            return self.endpoint

        # Auto-detect using the same logic as reasoning client
        llm_cfg = LLMConfig.from_yaml()
        base = llm_cfg.base_url.rstrip("/")
        return f"{base}/chat/completions"

    def resolve_model(self) -> str:
        """Resolve the model name based on environment."""
        if self.model and self.model not in (GENSPARK_MODEL, OPENAI_MODEL):
            return self.model  # User override

        llm_cfg = LLMConfig.from_yaml()
        return llm_cfg.model or OPENAI_MODEL


@dataclass
class ServiceConfig:
    """Configuration for outbound service integrations."""

    name: str
    service_type: str = "webhook"
    endpoint: str = ""
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    template: Optional[str] = None


@dataclass
class MoltbotConfig:
    """Top-level Moltbot configuration."""

    ai: AIProviderConfig = field(default_factory=AIProviderConfig)
    services: List[ServiceConfig] = field(default_factory=list)
    risk: Dict[str, Any] = field(default_factory=dict)


def load_moltbot_config(path: str) -> MoltbotConfig:
    """Load Moltbot configuration from YAML."""
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    ai_raw = raw.get("ai", {}) or {}
    services_raw = raw.get("services", []) or []
    ai = AIProviderConfig(**{k: v for k, v in ai_raw.items()
                             if k in AIProviderConfig.__dataclass_fields__})
    services = [
        ServiceConfig(
            name=service.get("name", ""),
            service_type=service.get("type", "webhook"),
            endpoint=service.get("endpoint", ""),
            method=service.get("method", "POST"),
            headers=service.get("headers", {}) or {},
            template=service.get("template"),
        )
        for service in services_raw
    ]
    return MoltbotConfig(ai=ai, services=services, risk=raw.get("risk", {}) or {})


class AIClient:
    """Abstract AI client interface."""

    def generate_plan(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIChatClient(AIClient):
    """Environment-aware OpenAI-compatible client using stdlib."""

    def __init__(self, config: AIProviderConfig):
        self.config = config

    def _get_api_key(self) -> str:
        """Get API key via the unified LLMConfig detection."""
        llm_cfg = LLMConfig.from_yaml()
        if llm_cfg.api_key:
            return llm_cfg.api_key

        # Direct env check as fallback
        api_key = os.getenv(self.config.api_key_env)
        if api_key:
            return api_key

        raise RuntimeError(
            f"Missing API key: set {self.config.api_key_env} env var, "
            f"add to .env file, or configure ~/.genspark_llm.yaml"
        )

    def generate_plan(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        api_key = self._get_api_key()
        endpoint = self.config.resolve_endpoint()
        model = self.config.resolve_model()

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "user", "content": json.dumps(context)},
            ],
            "temperature": 0.1,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


class StubAIClient(AIClient):
    """Fallback AI client for local testing without external calls."""

    def generate_plan(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        forecast = context.get("forecast", {})
        direction = forecast.get("direction", "flat")
        confidence = forecast.get("confidence", 0.5)
        return {
            "action": "hold" if direction == "flat" else "trade",
            "side": "buy" if direction == "up" else "sell",
            "confidence": confidence,
            "reason": "stubbed-response",
        }


class ServiceAdapter:
    """Abstract outbound service adapter."""

    def send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class WebhookService(ServiceAdapter):
    """Simple webhook adapter."""

    def __init__(self, config: ServiceConfig):
        self.config = config

    def send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.config.endpoint,
            data=body,
            headers=self.config.headers,
            method=self.config.method,
        )
        with request.urlopen(req) as resp:
            return {"status": resp.status, "service": self.config.name}


class MoltbotOrchestrator:
    """Orchestrates AI trade planning and service notifications."""

    def __init__(
        self,
        config: MoltbotConfig,
        ai_client: Optional[AIClient] = None,
        services: Optional[Iterable[ServiceAdapter]] = None,
    ):
        self.config = config
        self.ai_client = ai_client or self._default_ai_client()
        self.services = list(services) if services is not None else self._load_services()

    def _default_ai_client(self) -> AIClient:
        if self.config.ai.provider.lower() == "openai":
            return OpenAIChatClient(self.config.ai)
        return StubAIClient()

    def _load_services(self) -> List[ServiceAdapter]:
        adapters: List[ServiceAdapter] = []
        for service in self.config.services:
            if service.service_type == "webhook":
                adapters.append(WebhookService(service))
        return adapters

    def build_prompt(self, forecast: Dict[str, Any]) -> str:
        risk = self.config.risk or {}
        return (
            "Create a trade plan based on the forecast and risk rules. "
            f"Risk rules: {json.dumps(risk)}. "
            f"Forecast summary: {json.dumps(forecast)}."
        )

    def propose_trade(self, forecast: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.build_prompt(forecast)
        context = {"forecast": forecast, "risk": self.config.risk}
        return self.ai_client.generate_plan(prompt, context)

    def notify(self, trade_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        responses = []
        for service in self.services:
            responses.append(service.send(trade_plan))
        return responses
