"""LangChain model configuration for thesis evaluation commentary."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from thesisev.paths import config_dir

DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
PROVIDER_ENV_MAPPING = {}


@dataclass(slots=True)
class ModelConfig:
    """Runtime model configuration used by the evaluation pipeline."""

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    max_tokens: int = 400
    timeout: int = 60

    @property
    def api_key(self) -> str | None:
        """Return the API key env var for the configured provider."""

        return PROVIDER_ENV_MAPPING.get(self.provider, (None, None))[0]

    @property
    def base_url_env(self) -> str | None:
        """Return the base URL env var for the configured provider."""

        return PROVIDER_ENV_MAPPING.get(self.provider, (None, None))[1]

    def is_available(self) -> bool:
        """Whether the configured model has the required credentials."""

        if not self.api_key:
            return False
        return bool(os.getenv(self.api_key))

    def to_metadata(self) -> dict[str, Any]:
        """Serialize model configuration for result metadata."""

        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "api_key": self.api_key,
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
    }.get(provider, provider)


def create_chat_model(config: ModelConfig):
    """Create a LangChain chat model instance from the runtime config."""

    if config.provider == "deepseek":
        return create_deepseek_chat_model(config)

    extra_kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
    }
    if config.base_url_env and os.getenv(config.base_url_env):
        extra_kwargs["base_url"] = os.getenv(config.base_url_env)
    return init_chat_model(
        model=config.model, model_provider=config.provider, **extra_kwargs
    )


def create_deepseek_chat_model(config: ModelConfig) -> ChatOpenAI:
    """Create a DeepSeek chat model through the OpenAI-compatible client."""

    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "base_url": resolve_deepseek_base_url(config),
    }
    if config.api_key and os.getenv(config.api_key):
        kwargs["api_key"] = os.getenv(config.api_key)
    return ChatOpenAI(**kwargs)


def resolve_deepseek_base_url(config: ModelConfig) -> str:
    """Resolve DeepSeek's OpenAI-compatible base URL."""

    if config.base_url_env and os.getenv(config.base_url_env):
        return os.getenv(config.base_url_env) or DEFAULT_DEEPSEEK_BASE_URL
    return DEFAULT_DEEPSEEK_BASE_URL


def load_provider_env_mapping() -> dict[str, tuple[str | None, str | None]]:
    """Load provider env mapping from the bundled TOML config."""

    config_path = config_dir() / "provider_env.toml"
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    providers = config.get("providers", {})
    return {
        provider: (values.get("api_key"), values.get("base_url_env"))
        for provider, values in providers.items()
    }


PROVIDER_ENV_MAPPING = load_provider_env_mapping()
