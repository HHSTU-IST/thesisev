"""LangChain model configuration for thesis evaluation commentary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model

DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"

PROVIDER_ENV_MAPPING = {
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "anthropic": ("ANTHROPIC_API_KEY", None),
    "google_genai": ("GOOGLE_API_KEY", None),
}


@dataclass(slots=True)
class ModelConfig:
    """Runtime model configuration used by the evaluation pipeline."""

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    max_tokens: int = 400
    timeout: int = 60

    @property
    def api_key_env(self) -> str | None:
        """Return the API key env var for the configured provider."""

        return PROVIDER_ENV_MAPPING.get(self.provider, (None, None))[0]

    @property
    def base_url_env(self) -> str | None:
        """Return the base URL env var for the configured provider."""

        return PROVIDER_ENV_MAPPING.get(self.provider, (None, None))[1]

    def is_available(self) -> bool:
        """Whether the configured model has the required credentials."""

        if not self.api_key_env:
            return False
        return bool(os.getenv(self.api_key_env))

    def to_metadata(self) -> dict[str, Any]:
        """Serialize model configuration for result metadata."""

        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "api_key_env": self.api_key_env,
            "base_url_env": self.base_url_env,
            "available": self.is_available(),
        }


def build_model_config(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 400,
    timeout: int = 60,
) -> ModelConfig:
    """Build a model config from CLI or service inputs."""

    resolved_provider = (provider or DEFAULT_PROVIDER).strip().lower()
    resolved_model = (model or default_model_for_provider(resolved_provider)).strip()
    return ModelConfig(
        provider=resolved_provider,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def default_model_for_provider(provider: str) -> str:
    """Return a sensible default model name for each provider."""

    return {
        "deepseek": "deepseek-chat",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-latest",
        "google_genai": "gemini-2.5-flash",
    }.get(provider, provider)


def create_chat_model(config: ModelConfig):
    """Create a LangChain chat model instance from the runtime config."""

    extra_kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
    }
    if config.base_url_env and os.getenv(config.base_url_env):
        extra_kwargs["base_url"] = os.getenv(config.base_url_env)
    return init_chat_model(
        model=config.model,
        model_provider=config.provider,
        **extra_kwargs,
    )
