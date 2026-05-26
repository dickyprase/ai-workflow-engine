"""Multi-model routing with provider abstraction and intelligent selection."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

import aiohttp

from engine.config import EngineConfig, ModelProviderConfig

logger = logging.getLogger(__name__)


class ModelCapability(str, Enum):
    """Capabilities that models can advertise."""

    CHAT = "chat"
    CODE = "code"
    REASONING = "reasoning"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    LONG_CONTEXT = "long_context"
    FAST_INFERENCE = "fast_inference"


@dataclass
class ModelResponse:
    """Standardized response from any model provider."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of the routing algorithm."""

    provider: str
    model: str
    reason: str
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    def __init__(self, name: str, config: ModelProviderConfig):
        self.name = name
        self.config = config
        self._request_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Send a completion request to the provider."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion response."""
        ...

    @property
    def capabilities(self) -> list[ModelCapability]:
        """Return capabilities supported by this provider."""
        return [ModelCapability.CHAT]

    @property
    def stats(self) -> dict[str, Any]:
        """Return usage statistics."""
        return {
            "requests": self._request_count,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
        }


class OpenAIProvider(ModelProvider):
    """OpenAI API provider (GPT-4, GPT-4o, etc.)."""

    COST_PER_1K = {"gpt-4o": (0.005, 0.015), "gpt-4o-mini": (0.00015, 0.0006)}

    def __init__(self, config: ModelProviderConfig):
        super().__init__("openai", config)
        self.base_url = config.base_url or "https://api.openai.com/v1"

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        model = model or self.config.default_model or "gpt-4o"
        start = time.perf_counter()

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as resp:
                data = await resp.json()

        latency = (time.perf_counter() - start) * 1000
        self._request_count += 1

        usage = data.get("usage", {})
        self._total_tokens += usage.get("total_tokens", 0)

        cost = self._calculate_cost(model, usage)
        self._total_cost += cost

        return ModelResponse(
            content=data["choices"][0]["message"]["content"],
            model=model,
            provider="openai",
            usage=usage,
            latency_ms=latency,
            cost=cost,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        model = model or self.config.default_model or "gpt-4o"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                async for line in resp.content:
                    decoded = line.decode().strip()
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        import json

                        chunk = json.loads(decoded[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]

    @property
    def capabilities(self) -> list[ModelCapability]:
        return [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
        ]

    def _calculate_cost(self, model: str, usage: dict) -> float:
        rates = self.COST_PER_1K.get(model, (0.005, 0.015))
        input_cost = (usage.get("prompt_tokens", 0) / 1000) * rates[0]
        output_cost = (usage.get("completion_tokens", 0) / 1000) * rates[1]
        return input_cost + output_cost


class AnthropicProvider(ModelProvider):
    """Anthropic API provider (Claude models)."""

    COST_PER_1K = {
        "claude-sonnet-4-20250514": (0.003, 0.015),
        "claude-haiku-35": (0.00025, 0.00125),
    }

    def __init__(self, config: ModelProviderConfig):
        super().__init__("anthropic", config)
        self.base_url = config.base_url or "https://api.anthropic.com/v1"

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        model = model or self.config.default_model or "claude-sonnet-4-20250514"
        start = time.perf_counter()

        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2024-01-01",
        }

        # Convert system message to top-level param (Anthropic format)
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        payload: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if system_msg:
            payload["system"] = system_msg

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as resp:
                data = await resp.json()

        latency = (time.perf_counter() - start) * 1000
        self._request_count += 1

        usage = data.get("usage", {})
        self._total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

        cost = self._calculate_cost(model, usage)
        self._total_cost += cost

        content = data["content"][0]["text"] if data.get("content") else ""

        return ModelResponse(
            content=content,
            model=model,
            provider="anthropic",
            usage=usage,
            latency_ms=latency,
            cost=cost,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        # Simplified streaming implementation
        response = await self.complete(messages, model, **kwargs)
        yield response.content

    @property
    def capabilities(self) -> list[ModelCapability]:
        return [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.LONG_CONTEXT,
        ]

    def _calculate_cost(self, model: str, usage: dict) -> float:
        rates = self.COST_PER_1K.get(model, (0.003, 0.015))
        input_cost = (usage.get("input_tokens", 0) / 1000) * rates[0]
        output_cost = (usage.get("output_tokens", 0) / 1000) * rates[1]
        return input_cost + output_cost


class ModelRouter:
    """Intelligent model routing based on task requirements and strategy."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._providers: dict[str, ModelProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize configured model providers."""
        provider_classes = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
        }

        for name, provider_config in self.config.models.items():
            if name in provider_classes:
                self._providers[name] = provider_classes[name](provider_config)
                logger.info(f"Initialized provider: {name}")

    def register_provider(self, name: str, provider: ModelProvider) -> None:
        """Register a custom model provider."""
        self._providers[name] = provider
        logger.info(f"Registered custom provider: {name}")

    async def route(
        self,
        messages: list[dict[str, str]],
        required_capabilities: list[ModelCapability] | None = None,
        preferred_provider: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Route a request to the optimal model based on strategy."""
        decision = self._make_routing_decision(required_capabilities, preferred_provider)
        logger.info(f"Routing to {decision.provider}/{decision.model}: {decision.reason}")

        provider = self._providers[decision.provider]

        # Try with fallback chain
        fallback_chain = self.config.routing.fallback_chain
        tried = set()

        for provider_name in [decision.provider] + fallback_chain:
            if provider_name in tried or provider_name not in self._providers:
                continue
            tried.add(provider_name)

            try:
                return await self._providers[provider_name].complete(messages, **kwargs)
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue

        raise RuntimeError("All providers in fallback chain failed")

    def _make_routing_decision(
        self,
        required_capabilities: list[ModelCapability] | None = None,
        preferred_provider: str | None = None,
    ) -> RoutingDecision:
        """Determine the best provider/model for the request."""
        if preferred_provider and preferred_provider in self._providers:
            provider = self._providers[preferred_provider]
            return RoutingDecision(
                provider=preferred_provider,
                model=provider.config.default_model,
                reason="User-specified provider preference",
            )

        strategy = self.config.routing.strategy

        if strategy == "cost_optimized":
            return self._route_by_cost(required_capabilities)
        elif strategy == "latency_optimized":
            return self._route_by_latency(required_capabilities)
        else:
            return self._route_by_capability(required_capabilities)

    def _route_by_cost(
        self, capabilities: list[ModelCapability] | None
    ) -> RoutingDecision:
        """Select the cheapest provider that meets capability requirements."""
        # Prefer cheaper providers first
        cost_order = ["anthropic", "openai"]
        for name in cost_order:
            if name in self._providers:
                provider = self._providers[name]
                if capabilities and not all(c in provider.capabilities for c in capabilities):
                    continue
                return RoutingDecision(
                    provider=name,
                    model=provider.config.default_model,
                    reason=f"Cost-optimized selection",
                )

        # Fallback to first available
        name = next(iter(self._providers))
        return RoutingDecision(
            provider=name,
            model=self._providers[name].config.default_model,
            reason="Fallback - no optimal match",
        )

    def _route_by_latency(
        self, capabilities: list[ModelCapability] | None
    ) -> RoutingDecision:
        """Select the fastest provider."""
        # OpenAI generally has lower latency for simple tasks
        latency_order = ["openai", "anthropic"]
        for name in latency_order:
            if name in self._providers:
                return RoutingDecision(
                    provider=name,
                    model=self._providers[name].config.default_model,
                    reason="Latency-optimized selection",
                )

        name = next(iter(self._providers))
        return RoutingDecision(
            provider=name,
            model=self._providers[name].config.default_model,
            reason="Fallback",
        )

    def _route_by_capability(
        self, capabilities: list[ModelCapability] | None
    ) -> RoutingDecision:
        """Select provider with best capability match."""
        if not capabilities:
            name = next(iter(self._providers))
            return RoutingDecision(
                provider=name,
                model=self._providers[name].config.default_model,
                reason="No specific capabilities required",
            )

        best_match = None
        best_score = -1

        for name, provider in self._providers.items():
            score = sum(1 for c in capabilities if c in provider.capabilities)
            if score > best_score:
                best_score = score
                best_match = name

        if best_match:
            return RoutingDecision(
                provider=best_match,
                model=self._providers[best_match].config.default_model,
                reason=f"Capability match score: {best_score}/{len(capabilities)}",
            )

        name = next(iter(self._providers))
        return RoutingDecision(
            provider=name,
            model=self._providers[name].config.default_model,
            reason="Fallback",
        )

    def get_cost_estimate(self, input_tokens: int, output_tokens: int, provider: str = "openai") -> float:
        """Estimate cost for a given token count."""
        if provider == "openai":
            return (input_tokens / 1000) * 0.005 + (output_tokens / 1000) * 0.015
        elif provider == "anthropic":
            return (input_tokens / 1000) * 0.003 + (output_tokens / 1000) * 0.015
        return 0.0

    @property
    def available_providers(self) -> list[str]:
        """List registered provider names."""
        return list(self._providers.keys())
