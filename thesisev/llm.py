"""LangChain model configuration for thesis evaluation commentary.

Two **engines** are supported, and the engine is derived from the selected
provider:

``local`` (default, provider ``ollama``)
    An Ollama server on this host, reached through its OpenAI-compatible
    endpoint.  The thesis never leaves the machine, and the scoring model is
    pinned to one weight version, so scores stay comparable across runs.

``remote`` (provider ``deepseek``)
    DeepSeek's hosted API.  Higher scoring quality on subjective rubric items,
    at the cost of uploading the document to a third party and of a model
    alias that can be re-pointed without notice.

Because the engine is derived, ``--provider deepseek`` and ``--engine remote``
are equivalent; :func:`build_model_config` maps an engine name onto its
provider.  Both engines share the same LangChain call path, so switching only
changes the endpoint and the request body, and neither needs an extra
dependency.
"""

from __future__ import annotations

import json
import os
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from thesisev.paths import config_dir

ENGINE_LOCAL = "local"
ENGINE_REMOTE = "remote"

#: Providers served from this host.  They need no credential, so they count as
#: available as soon as their runtime is reachable.
LOCAL_PROVIDERS = frozenset({"ollama"})

#: Provider selected for each engine when the caller names only an engine.
ENGINE_PROVIDERS: dict[str, str] = {
    ENGINE_LOCAL: "ollama",
    ENGINE_REMOTE: "deepseek",
}

DEFAULT_ENGINE = ENGINE_LOCAL
DEFAULT_PROVIDER = ENGINE_PROVIDERS[DEFAULT_ENGINE]
DEFAULT_MODEL = "qwen3:8b"

DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama": "http://127.0.0.1:11434/v1",
    "deepseek": "https://api.deepseek.com",
}
DEFAULT_DEEPSEEK_BASE_URL = DEFAULT_BASE_URLS["deepseek"]
DEFAULT_OLLAMA_BASE_URL = DEFAULT_BASE_URLS["ollama"]

#: Ollama ignores the API key, but the OpenAI client refuses to build without
#: one, so a placeholder is supplied for the local engine.
OLLAMA_PLACEHOLDER_API_KEY = "ollama"

#: ``langchain-openai`` unconditionally rewrites the ``max_tokens`` keyword to
#: ``max_completion_tokens``, a name neither DeepSeek nor Ollama documents, so
#: the limit has to be repeated in ``extra_body`` to reach the wire.
#:
#: The local engine gets a wider default because a rubric item's JSON carries
#: prose ``evidence`` / ``deductions`` / ``suggestions``; the remote callers
#: tune theirs through the CLI and the API.
DEFAULT_MAX_TOKENS: dict[str, int] = {"ollama": 1024}
FALLBACK_MAX_TOKENS = 400

#: DeepSeek retired the ``deepseek-chat`` / ``deepseek-reasoner`` aliases on
#: 2026-07-24; requests to them now fail with an HTTP error.  The documented
#: model IDs are ``deepseek-flash`` and ``deepseek-v4-pro``.
RETIRED_DEEPSEEK_MODELS = frozenset({"deepseek-chat", "deepseek-reasoner"})

#: Ollama's OpenAI-compatible route drops an unknown ``num_ctx`` key while
#: decoding, so the context window can only be raised server-side.  A 12 GiB
#: card falls in the ``< 24 GiB`` tier, whose default is 4096 tokens -- close
#: enough to this pipeline's prompt size that overflow silently discards the
#: *front* of the conversation, system prompt included.
MIN_RECOMMENDED_LOCAL_CONTEXT = 16384
LOCAL_CONTEXT_ENV = "OLLAMA_CONTEXT_LENGTH"


def _read_env(name: str | None) -> str | None:
    """Read a non-empty environment variable, if a name was configured."""

    if not name:
        return None
    return os.getenv(name) or None


def _as_optional_str(value: object) -> str | None:
    """Coerce a TOML value to a stripped string, or ``None`` when empty."""

    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    """Credential and endpoint settings declared for one provider.

    ``api_key_env`` / ``base_url_env`` name environment variables and win when
    they are set; ``api_key`` / ``base_url`` hold literal values so a local run
    can be configured without exporting anything.
    """

    api_key_env: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None

    def resolve_api_key(self) -> str | None:
        """Return the credential value, preferring the environment variable."""

        from_env = _read_env(self.api_key_env)
        if from_env:
            return from_env
        return self.api_key

    def resolve_base_url(self, default: str | None = None) -> str | None:
        """Return the endpoint base URL, preferring the environment variable."""

        from_env = _read_env(self.base_url_env)
        if from_env:
            return from_env
        return self.base_url or default


@dataclass(frozen=True, slots=True)
class LocalRuntimeProbe:
    """Outcome of a reachability probe against a local inference server."""

    reachable: bool
    model_installed: bool | None
    models: tuple[str, ...]
    detail: str


@dataclass(slots=True)
class ModelConfig:
    """Runtime model configuration used by the evaluation pipeline."""

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    temperature: float = 0.2

    #: Explicit output-token override.  ``None`` -- the default -- resolves to
    #: the provider's own budget through :attr:`output_tokens`, so a config
    #: built here directly and one built by :func:`build_model_config` cannot
    #: drift apart on the budget.
    max_tokens: int | None = None
    timeout: int = 60

    #: Result of :func:`probe_local_runtime` for a local engine.  ``None``
    #: means the runtime was never probed.  ``False`` forces the deterministic
    #: degradation path so a stopped server costs no retry backoff.
    runtime_reachable: bool | None = None

    @property
    def output_tokens(self) -> int:
        """Return the output-token budget actually sent to the provider."""

        if self.max_tokens is not None:
            return self.max_tokens
        return default_max_tokens_for_provider(self.provider)

    @property
    def engine(self) -> str:
        """Return the engine this provider belongs to."""

        return ENGINE_LOCAL if self.provider in LOCAL_PROVIDERS else ENGINE_REMOTE

    @property
    def is_local(self) -> bool:
        """Whether this provider is served from this host."""

        return self.provider in LOCAL_PROVIDERS

    @property
    def requires_credential(self) -> bool:
        """Whether this provider cannot run without an API key."""

        return not self.is_local

    @property
    def settings(self) -> ProviderSettings:
        """Return the provider settings declared in ``provider_env.toml``."""

        return PROVIDER_SETTINGS.get(self.provider, ProviderSettings())

    @property
    def api_key(self) -> str | None:
        """Return the resolved credential value, or ``None`` when unneeded."""

        if self.is_local:
            return None
        return self.settings.resolve_api_key()

    @property
    def base_url(self) -> str | None:
        """Return the resolved endpoint base URL for the configured provider."""

        fallback = DEFAULT_BASE_URLS.get(self.provider)
        resolved = self.settings.resolve_base_url(fallback)
        if resolved:
            return resolved
        return fallback

    @property
    def credential_source(self) -> str | None:
        """Describe where the credential came from without exposing its value."""

        if self.is_local:
            return "not-required"
        settings = self.settings
        if _read_env(settings.api_key_env):
            return f"env:{settings.api_key_env}"
        if settings.api_key:
            return "config"
        return None

    def is_available(self) -> bool:
        """Whether the configured model can actually answer requests.

        A local engine needs no credential and therefore counts as available;
        the runtime probe is what can demote it, so callers must run
        :func:`resolve_runtime` before relying on this for a local engine.
        """

        if self.runtime_reachable is False:
            return False
        if not self.requires_credential:
            return True
        return bool(self.api_key)

    def availability(self) -> str:
        """Return a short, non-secret reason for the availability verdict."""

        if self.runtime_reachable is False:
            return "runtime_unreachable"
        if not self.requires_credential:
            return "credential_not_required"
        if self.api_key:
            return f"credential:{self.credential_source}"
        return "credential_missing"

    def to_metadata(self) -> dict[str, Any]:
        """Serialize model configuration for result metadata.

        The credential value is deliberately never included: this metadata is
        persisted into ``data/history.json`` alongside every review.
        """

        return {
            "engine": self.engine,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.output_tokens,
            "timeout": self.timeout,
            "base_url": self.base_url,
            "requires_credential": self.requires_credential,
            "credential_source": self.credential_source,
            "available": self.is_available(),
            "availability": self.availability(),
        }


def build_model_config(
    *,
    engine: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    timeout: int = 60,
) -> ModelConfig:
    """Build a model config from CLI, API or UI inputs.

    ``provider`` wins over ``engine`` because the engine is derived from the
    provider; ``engine`` exists so a caller can switch modes without knowing
    which provider backs each one.  ``max_tokens=None`` selects the provider's
    own default.
    """

    resolved_provider = (provider or _provider_for_engine(engine)).strip().lower()
    resolved_model = (model or default_model_for_provider(resolved_provider)).strip()
    return ModelConfig(
        provider=resolved_provider,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def _provider_for_engine(engine: str | None) -> str:
    """Map an engine name onto its provider, defaulting to the local engine."""

    normalized = (engine or DEFAULT_ENGINE).strip().lower()
    return ENGINE_PROVIDERS.get(normalized, ENGINE_PROVIDERS[DEFAULT_ENGINE])


def default_model_for_provider(provider: str) -> str:
    """Return a sensible default model name for each provider."""

    return {
        "ollama": DEFAULT_MODEL,
        "deepseek": "deepseek-flash",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-latest",
    }.get(provider, provider)


def default_max_tokens_for_provider(provider: str) -> int:
    """Return the output-token budget for each provider."""

    return DEFAULT_MAX_TOKENS.get(provider, FALLBACK_MAX_TOKENS)


def build_deepseek_extra_body(max_tokens: int) -> dict[str, Any]:
    """Build the DeepSeek-specific top-level request fields.

    ``extra_body`` is merged into the top-level JSON body by the OpenAI SDK,
    which is the only route to DeepSeek's own ``thinking`` switch.  Thinking is
    disabled because it is enabled by default, and the chain of thought would
    otherwise consume the token budget and pollute the JSON reply.
    """

    return {"thinking": {"type": "disabled"}, "max_tokens": max_tokens}


def build_ollama_extra_body(max_tokens: int) -> dict[str, Any]:
    """Build the Ollama-specific top-level request fields.

    ``reasoning_effort`` is the OpenAI-compatible route to Ollama's thinking
    switch; ``"none"`` requests no thinking output.  Qwen3 runs with thinking
    enabled by default in Ollama, and the trace would otherwise eat the token
    budget and leave ``content`` empty.  The native ``think`` field is not
    accepted on this route -- only the Responses API exposes it.
    """

    return {"reasoning_effort": "none", "max_tokens": max_tokens}


def extract_response_text(response: Any) -> str:
    """Extract text content from a LangChain response object."""

    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def create_chat_model(config: ModelConfig):
    """Create a LangChain chat model instance from the runtime config."""

    if config.provider == "deepseek":
        return create_deepseek_chat_model(config)
    if config.provider == "ollama":
        return create_ollama_chat_model(config)

    extra_kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.output_tokens,
        "timeout": config.timeout,
    }
    base_url = config.base_url
    if base_url:
        extra_kwargs["base_url"] = base_url
    return init_chat_model(
        model=config.model, model_provider=config.provider, **extra_kwargs
    )


def create_openai_compatible_chat_model(
    config: ModelConfig,
    *,
    default_base_url: str,
    extra_body: dict[str, Any],
    placeholder_api_key: str | None = None,
) -> ChatOpenAI:
    """Build a client for a provider that speaks the OpenAI protocol."""

    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.output_tokens,
        "timeout": config.timeout,
        "base_url": config.base_url or default_base_url,
        "extra_body": extra_body,
    }
    api_key = config.api_key or placeholder_api_key
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


def create_deepseek_chat_model(config: ModelConfig) -> ChatOpenAI:
    """Create a DeepSeek chat model through the OpenAI-compatible client."""

    return create_openai_compatible_chat_model(
        config,
        default_base_url=DEFAULT_DEEPSEEK_BASE_URL,
        extra_body=build_deepseek_extra_body(config.output_tokens),
    )


def create_ollama_chat_model(config: ModelConfig) -> ChatOpenAI:
    """Create a chat model backed by the local Ollama server."""

    return create_openai_compatible_chat_model(
        config,
        default_base_url=DEFAULT_OLLAMA_BASE_URL,
        extra_body=build_ollama_extra_body(config.output_tokens),
        placeholder_api_key=OLLAMA_PLACEHOLDER_API_KEY,
    )


def probe_local_runtime(
    config: ModelConfig, *, timeout: float = 2.0
) -> LocalRuntimeProbe | None:
    """Check that a local engine's server answers and holds the model.

    Returns ``None`` for a remote engine.  The probe is a single short HTTP
    request to the server's model list, so it can fail fast and describe *why*
    a local run would otherwise degrade silently.

    The request bypasses any configured proxy on purpose: the endpoint is on
    the loopback interface, and an HTTP proxy in the environment would turn a
    healthy local server into a spurious failure.
    """

    if not config.is_local:
        return None
    base_url = (config.base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    endpoint = f"{base_url}/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(endpoint, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return LocalRuntimeProbe(
            reachable=False,
            model_installed=None,
            models=(),
            detail=(
                f"local inference server is not reachable at {base_url}: {exc.reason}"
            ),
        )
    except (OSError, ValueError) as exc:
        return LocalRuntimeProbe(
            reachable=False,
            model_installed=None,
            models=(),
            detail=(
                f"local inference server at {base_url} returned an invalid "
                f"response: {exc}"
            ),
        )

    models = _extract_model_ids(payload)
    installed = _model_is_installed(config.model, models)
    if installed:
        detail = f"local inference server at {base_url} is ready with {config.model}"
    else:
        detail = (
            f"local inference server at {base_url} is running but {config.model} "
            f"is not pulled (installed: {', '.join(models) or 'none'})"
        )
    return LocalRuntimeProbe(
        reachable=True,
        model_installed=installed,
        models=models,
        detail=detail,
    )


def _extract_model_ids(payload: Any) -> tuple[str, ...]:
    """Read model identifiers out of an OpenAI-compatible ``/models`` reply."""

    if not isinstance(payload, dict):
        return ()
    entries = payload.get("data")
    if not isinstance(entries, list):
        return ()
    return tuple(
        str(entry["id"])
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    )


def _model_is_installed(model: str, installed: tuple[str, ...]) -> bool:
    """Match a configured model name against the server's installed list.

    A bare repository name matches any installed tag of that repository, so
    ``model="qwen3"`` is satisfied by ``qwen3:8b``.
    """

    wanted = model.strip()
    if wanted in installed:
        return True
    wanted_base = wanted.split(":", 1)[0]
    return any(name.split(":", 1)[0] == wanted_base for name in installed)


def resolve_runtime(
    config: ModelConfig, *, probe_timeout: float = 2.0
) -> tuple[ModelConfig, str | None]:
    """Probe a local engine and demote the config when its server is down.

    Returns the resolved config together with a human-readable warning, or
    ``None`` when the run can proceed without comment.  Demoting sets
    ``runtime_reachable`` to ``False``, which makes :meth:`is_available` false
    and sends the pipeline straight down its deterministic path instead of
    paying for connection-refused retries on every one of its calls.
    """

    probe = probe_local_runtime(config, timeout=probe_timeout)
    if probe is None:
        return config, None
    if not probe.reachable:
        config.runtime_reachable = False
        return config, probe.detail
    if probe.model_installed is False:
        config.runtime_reachable = False
        return config, probe.detail
    config.runtime_reachable = True
    return config, None


def local_context_hint(config: ModelConfig) -> str | None:
    """Describe the server-side context-window prerequisite for a local engine.

    The Ollama OpenAI-compatible route silently discards an unknown
    ``num_ctx`` key, so the window can only be raised by the server.  Returning
    a hint rather than a warning keeps it informational: a small window
    degrades answer quality without failing the run.
    """

    if not config.is_local:
        return None
    return (
        f"本地引擎的上下文窗口需在服务端设置：启动 Ollama 前设 "
        f"{LOCAL_CONTEXT_ENV}={MIN_RECOMMENDED_LOCAL_CONTEXT}。"
        "OpenAI 兼容接口会丢弃请求体内的 num_ctx，"
        "而本机显存档位对应的默认窗口偏小，超出时会静默丢弃 prompt 开头部分。"
    )


def invoke_chat_model_with_retry(
    model: Any,
    messages: Any,
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
) -> Any:
    """Invoke a chat model with exponential-backoff retries.

    Transient provider/network errors are retried with ``base_delay *
    2 ** attempt`` sleeps between attempts. When every attempt fails the last
    exception is re-raised so callers can degrade to deterministic scoring.
    """

    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return model.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - provider errors are heterogeneous
            last_error = exc
            if attempt + 1 < max(1, attempts):
                time.sleep(base_delay * (2**attempt))
    if last_error is not None:
        raise last_error
    msg = "invoke_chat_model_with_retry called without a model"
    raise RuntimeError(msg)


def load_provider_settings() -> dict[str, ProviderSettings]:
    """Load provider credentials and endpoints from the bundled TOML config."""

    config_path = config_dir() / "provider_env.toml"
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    providers = config.get("providers", {})
    settings: dict[str, ProviderSettings] = {}
    for provider, values in providers.items():
        settings[provider] = ProviderSettings(
            api_key_env=_as_optional_str(values.get("api_key_env")),
            api_key=_as_optional_str(values.get("api_key")),
            base_url=_as_optional_str(values.get("base_url")),
            base_url_env=_as_optional_str(values.get("base_url_env")),
        )
    return settings


PROVIDER_SETTINGS = load_provider_settings()
