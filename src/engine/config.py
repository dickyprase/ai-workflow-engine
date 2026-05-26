"""Configuration management for the AI Workflow Engine."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RetryPolicy(BaseModel):
    """Retry configuration for failed tasks."""

    max_retries: int = 3
    backoff_factor: float = 2.0
    retry_on: list[str] = Field(default_factory=lambda: ["timeout", "rate_limit", "server_error"])


class ModelProviderConfig(BaseModel):
    """Configuration for a single model provider."""

    api_key: str = ""
    base_url: str | None = None
    default_model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 30.0
    rate_limit_rpm: int = 60


class RoutingConfig(BaseModel):
    """Model routing strategy configuration."""

    strategy: str = "cost_optimized"
    fallback_chain: list[str] = Field(default_factory=lambda: ["openai", "anthropic"])
    capability_map: dict[str, list[str]] = Field(default_factory=dict)


class EngineConfig(BaseModel):
    """Top-level engine configuration."""

    max_concurrency: int = 10
    default_timeout: float = 30.0
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    models: dict[str, ModelProviderConfig] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    plugins: list[str] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EngineConfig":
        """Load configuration from a YAML file with environment variable interpolation."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        content = path.read_text()
        content = cls._interpolate_env_vars(content)
        data = yaml.safe_load(content)

        # Flatten nested 'engine' key if present
        if "engine" in data and isinstance(data["engine"], dict):
            engine_data = data.pop("engine")
            data = {**engine_data, **data}

        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
        """Create configuration from a dictionary."""
        return cls(**data)

    @staticmethod
    def _interpolate_env_vars(content: str) -> str:
        """Replace ${VAR_NAME} patterns with environment variable values."""
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default = ""
            if ":-" in var_name:
                var_name, default = var_name.split(":-", 1)
            return os.environ.get(var_name, default)

        return pattern.sub(replacer, content)

    def get_provider_config(self, provider: str) -> ModelProviderConfig | None:
        """Get configuration for a specific model provider."""
        return self.models.get(provider)
