"""B2/B3: 双档引擎 — 本地默认 / 远程显式切换。

覆盖三类历史缺陷与两类新增约束：`api_key` 被当成环境变量名导致
`is_available()` 恒为假、默认模型名指向已停用的 `deepseek-chat`、密钥被写进
落盘元数据；以及本地引擎无需凭据、远程引擎必须显式选择。
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, override

import pytest

from thesisev import llm as llm_module
from thesisev.llm import (
    DEFAULT_ENGINE,
    DEFAULT_MODEL,
    ENGINE_LOCAL,
    ENGINE_REMOTE,
    OLLAMA_PLACEHOLDER_API_KEY,
    RETIRED_DEEPSEEK_MODELS,
    ModelConfig,
    ProviderSettings,
    build_model_config,
    create_chat_model,
    create_deepseek_chat_model,
    create_ollama_chat_model,
    default_max_tokens_for_provider,
    default_model_for_provider,
    extract_response_text,
    probe_local_runtime,
    resolve_runtime,
)

CREDENTIAL_ENV = "DEEPSEEK_API_KEY"
LOCAL_BASE_URL_ENV = "OLLAMA_BASE_URL"


def deepseek_config(model: str = "deepseek-flash") -> ModelConfig:
    """Return an explicitly remote config, independent of the engine defaults."""

    return ModelConfig(provider="deepseek", model=model)


def ollama_config(model: str = DEFAULT_MODEL) -> ModelConfig:
    """Return an explicitly local config, independent of the engine defaults."""

    return ModelConfig(provider="ollama", model=model)


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real credentials and endpoints out of these tests."""

    monkeypatch.delenv(CREDENTIAL_ENV, raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv(LOCAL_BASE_URL_ENV, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


def free_port() -> int:
    """Return a loopback port number that is currently unused."""

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def stub_server(
    handler: type[BaseHTTPRequestHandler],
) -> ThreadingHTTPServer:
    """Start a loopback HTTP server for the duration of a test."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def stop_server(server: ThreadingHTTPServer) -> None:
    """Shut a stub server down and wait for its thread to end."""

    server.shutdown()
    server.server_close()


class ChatStubHandler(BaseHTTPRequestHandler):
    """Capture one chat request and reply with a fixed completion."""

    captured: ClassVar[dict[str, Any]] = {}

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        type(self).captured["path"] = self.path
        type(self).captured["body"] = json.loads(self.rfile.read(length))
        type(self).captured["authorization"] = self.headers.get("Authorization")
        payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": type(self).captured["body"].get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"ok": true}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @override
    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default access log."""


class ModelListStubHandler(BaseHTTPRequestHandler):
    """Serve an OpenAI-compatible model list."""

    installed: tuple[str, ...] = ()

    def do_GET(self) -> None:
        payload = {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "library"}
                for name in type(self).installed
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @override
    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default access log."""


class RuntimeStubHandler(BaseHTTPRequestHandler):
    """Serve both the OpenAI-compatible list and Ollama's native ``/api/ps``.

    Set ``loaded_context`` to ``None`` to emulate a server that has no native
    route at all (any non-Ollama OpenAI-compatible backend).
    """

    installed: ClassVar[tuple[str, ...]] = ("qwen3:8b",)
    loaded_context: ClassVar[int | None] = None
    loaded_name: ClassVar[str | None] = None

    def do_GET(self) -> None:
        if self.path.endswith("/api/ps"):
            loaded = type(self).loaded_name or type(self).installed[0]
            if type(self).loaded_context is None:
                self.send_error(404)
                return
            payload = {
                "models": [
                    {
                        "name": loaded,
                        "model": loaded,
                        "context_length": type(self).loaded_context,
                    }
                ]
            }
        else:
            payload = {
                "object": "list",
                "data": [
                    {"id": name, "object": "model", "owned_by": "library"}
                    for name in type(self).installed
                ],
            }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @override
    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default access log."""


# --------------------------------------------------------------------------
# Engine selection
# --------------------------------------------------------------------------


def test_default_engine_is_local() -> None:
    """The default must keep the thesis on this host."""

    config = build_model_config()
    assert DEFAULT_ENGINE == ENGINE_LOCAL
    assert config.engine == ENGINE_LOCAL
    assert config.is_local is True
    assert config.requires_credential is False
    assert config.provider == "ollama"
    assert config.model == DEFAULT_MODEL == "qwen3:8b"


def test_remote_engine_requires_an_explicit_choice() -> None:
    config = build_model_config(engine=ENGINE_REMOTE)
    assert config.engine == ENGINE_REMOTE
    assert config.provider == "deepseek"
    assert config.model == "deepseek-flash"
    assert config.requires_credential is True


def test_naming_the_provider_switches_the_engine() -> None:
    """``--provider deepseek`` and ``--engine remote`` must be equivalent."""

    assert build_model_config(provider="deepseek").engine == ENGINE_REMOTE
    assert build_model_config(engine=ENGINE_REMOTE).provider == "deepseek"
    assert build_model_config(engine=ENGINE_LOCAL).engine == ENGINE_LOCAL
    # An unknown engine name falls back to the safe default rather than remote.
    assert build_model_config(engine="bogus").engine == ENGINE_LOCAL


def test_local_engine_needs_no_credential() -> None:
    config = ollama_config()
    assert config.api_key is None
    assert config.credential_source == "not-required"
    assert config.is_available() is True
    assert config.availability() == "credential_not_required"


def test_provider_defaults_do_not_carry_the_retired_alias() -> None:
    assert DEFAULT_MODEL not in RETIRED_DEEPSEEK_MODELS
    assert default_max_tokens_for_provider("ollama") == 1024
    assert default_max_tokens_for_provider("deepseek") == 2048
    assert build_model_config(engine=ENGINE_LOCAL).output_tokens == 1024
    assert build_model_config(engine=ENGINE_REMOTE).output_tokens == 2048
    assert build_model_config(engine=ENGINE_LOCAL, max_tokens=256).output_tokens == 256


def test_remote_budget_clears_the_largest_json_this_pipeline_asks_for() -> None:
    """A short budget truncates JSON silently, so it must not be short.

    ``finish_reason=length`` ends the reply mid-string, the parser raises and
    the caller degrades to the local rules -- measured against the live
    DeepSeek API, a 20-point rubric item already needed more than the 400 the
    remote engine used to request, and the deep review asks for six findings.
    """

    assert default_max_tokens_for_provider("deepseek") >= 2048
    assert default_max_tokens_for_provider("ollama") >= 1024


def test_both_construction_paths_agree_on_the_token_budget() -> None:
    """A directly built config must not silently differ from the factory."""

    for provider in ("ollama", "deepseek"):
        direct = ModelConfig(
            provider=provider, model=default_model_for_provider(provider)
        )
        assert (
            direct.output_tokens == build_model_config(provider=provider).output_tokens
        )
        assert direct.to_metadata()["max_tokens"] == direct.output_tokens


# --------------------------------------------------------------------------
# Remote credentials (unchanged contract)
# --------------------------------------------------------------------------


def test_credential_from_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CREDENTIAL_ENV, "sk-from-env")
    config = deepseek_config()
    assert config.is_available() is True
    assert config.api_key == "sk-from-env"
    assert config.credential_source == f"env:{CREDENTIAL_ENV}"


def test_credential_from_literal_config_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A literal value must resolve even though it is not an env var name."""

    monkeypatch.setitem(
        llm_module.PROVIDER_SETTINGS,
        "deepseek",
        ProviderSettings(api_key="sk-literal", base_url="https://api.deepseek.com"),
    )
    config = deepseek_config()
    assert config.is_available() is True
    assert config.api_key == "sk-literal"
    assert config.credential_source == "config"


def test_missing_credential_reports_unavailable() -> None:
    config = deepseek_config()
    assert config.api_key is None
    assert config.is_available() is False
    assert config.credential_source is None
    assert config.availability() == "credential_missing"


def test_metadata_never_contains_the_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata is persisted to data/history.json, so it must not leak keys."""

    credential_marker = "sk-must-not-be-persisted-1234567890"
    monkeypatch.setenv(CREDENTIAL_ENV, credential_marker)
    for config in (deepseek_config(), ollama_config()):
        metadata = config.to_metadata()
        serialized = json.dumps(metadata, ensure_ascii=False)
        assert credential_marker not in serialized
        assert "api_key" not in metadata
    metadata = deepseek_config().to_metadata()
    assert metadata["credential_source"] == f"env:{CREDENTIAL_ENV}"
    assert metadata["available"] is True
    assert metadata["engine"] == ENGINE_REMOTE


def test_local_metadata_is_self_describing() -> None:
    metadata = ollama_config().to_metadata()
    assert metadata["engine"] == ENGINE_LOCAL
    assert metadata["provider"] == "ollama"
    assert metadata["requires_credential"] is False
    assert metadata["available"] is True
    assert metadata["base_url"] == "http://127.0.0.1:11434/v1"
    # An unprobed config carries no window observation and no verdict.
    assert metadata["context_window"] is None
    assert metadata["context_window_adequate"] is None


def test_context_window_verdict_follows_the_observed_window() -> None:
    """The floor lives on the config so the frontend never restates it."""

    config = ollama_config()
    config.local_context_window = llm_module.MIN_RECOMMENDED_LOCAL_CONTEXT - 1
    assert config.context_window_adequate() is False
    assert config.to_metadata()["context_window_adequate"] is False
    config.local_context_window = llm_module.MIN_RECOMMENDED_LOCAL_CONTEXT
    assert config.context_window_adequate() is True
    assert ollama_config().context_window_adequate() is None


# --------------------------------------------------------------------------
# Request bodies
# --------------------------------------------------------------------------


def test_deepseek_model_disables_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CREDENTIAL_ENV, "sk-test")
    model = create_deepseek_chat_model(deepseek_config())
    assert model.model_name == "deepseek-flash"
    assert model.extra_body == {
        "thinking": {"type": "disabled"},
        "max_tokens": 2048,
    }
    assert model.temperature == 0.2
    assert model.request_timeout == 60.0


def test_ollama_model_disables_thinking() -> None:
    """Qwen3 defaults to thinking on, which would empty ``content``."""

    model = create_ollama_chat_model(ollama_config())
    assert model.model_name == "qwen3:8b"
    assert model.extra_body == {"reasoning_effort": "none", "max_tokens": 1024}
    assert str(model.openai_api_base) == "http://127.0.0.1:11434/v1"


def test_deepseek_wire_body_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert the JSON actually put on the wire, not just the client config."""

    server = stub_server(ChatStubHandler)
    try:
        monkeypatch.setenv(CREDENTIAL_ENV, "sk-test")
        monkeypatch.setenv(
            "DEEPSEEK_BASE_URL", f"http://127.0.0.1:{server.server_port}"
        )
        response = create_chat_model(deepseek_config()).invoke("ping")
        assert extract_response_text(response) == '{"ok": true}'
    finally:
        stop_server(server)

    captured = ChatStubHandler.captured
    assert captured["path"] == "/chat/completions"
    body = captured["body"]
    assert body["model"] == "deepseek-flash"
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 2048
    assert body["temperature"] == 0.2
    assert captured["authorization"] == "Bearer sk-test"


def test_ollama_wire_body_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local engine must reach the wire unauthenticated but well formed."""

    server = stub_server(ChatStubHandler)
    try:
        monkeypatch.setenv(
            LOCAL_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/v1"
        )
        response = create_chat_model(ollama_config()).invoke("ping")
        assert extract_response_text(response) == '{"ok": true}'
    finally:
        stop_server(server)

    captured = ChatStubHandler.captured
    assert captured["path"] == "/v1/chat/completions"
    body = captured["body"]
    assert body["model"] == "qwen3:8b"
    assert body["reasoning_effort"] == "none"
    assert body["max_tokens"] == 1024
    assert body["temperature"] == 0.2
    assert captured["authorization"] == f"Bearer {OLLAMA_PLACEHOLDER_API_KEY}"


# --------------------------------------------------------------------------
# Runtime probe and degradation
# --------------------------------------------------------------------------


def test_probe_is_skipped_for_a_remote_engine() -> None:
    assert probe_local_runtime(deepseek_config()) is None
    _, warning = resolve_runtime(deepseek_config())
    assert warning is None


def test_probe_reports_a_reachable_server_with_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ModelListStubHandler.installed = ("qwen3:8b", "llama3.2:3b")
    server = stub_server(ModelListStubHandler)
    try:
        monkeypatch.setenv(
            LOCAL_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/v1"
        )
        config, warning = resolve_runtime(ollama_config())
    finally:
        stop_server(server)

    assert warning is None
    assert config.is_available() is True
    assert config.availability() == "credential_not_required"


def test_probe_demotes_a_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server without the configured model would fail every single call."""

    ModelListStubHandler.installed = ("llama3.2:3b",)
    server = stub_server(ModelListStubHandler)
    try:
        monkeypatch.setenv(
            LOCAL_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/v1"
        )
        config, warning = resolve_runtime(ollama_config())
    finally:
        stop_server(server)

    assert warning is not None and "qwen3:8b" in warning
    assert config.is_available() is False
    assert config.availability() == "runtime_unreachable"


def test_probe_matches_a_bare_repository_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ModelListStubHandler.installed = ("qwen3:8b",)
    server = stub_server(ModelListStubHandler)
    try:
        monkeypatch.setenv(
            LOCAL_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/v1"
        )
        config, warning = resolve_runtime(ollama_config(model="qwen3"))
    finally:
        stop_server(server)

    assert warning is None
    assert config.is_available() is True


def test_probe_demotes_a_stopped_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closed port must degrade immediately instead of retrying each call."""

    monkeypatch.setenv(LOCAL_BASE_URL_ENV, f"http://127.0.0.1:{free_port()}/v1")
    config, warning = resolve_runtime(ollama_config())

    assert warning is not None and "not reachable" in warning
    assert config.is_available() is False
    assert config.availability() == "runtime_unreachable"


def test_local_context_hint_mentions_the_server_side_variable() -> None:
    hint = llm_module.local_context_hint(ollama_config())
    assert hint is not None and "OLLAMA_CONTEXT_LENGTH" in hint
    assert llm_module.local_context_hint(deepseek_config()) is None


def test_probe_reports_a_narrow_loaded_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reachable server with a 4096 window degrades silently, so it must warn.

    The server still answers, so the config stays available -- the point is to
    name the cause instead of letting every scoring call fall back unseen.
    """

    RuntimeStubHandler.installed = ("qwen3:8b",)
    RuntimeStubHandler.loaded_name = None
    RuntimeStubHandler.loaded_context = 4096
    server = stub_server(RuntimeStubHandler)
    try:
        monkeypatch.setenv(
            LOCAL_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/v1"
        )
        config, warning = resolve_runtime(ollama_config())
    finally:
        stop_server(server)

    assert warning is not None
    assert "4096" in warning and "OLLAMA_CONTEXT_LENGTH" in warning
    assert config.local_context_window == 4096
    assert config.is_available() is True
    # A concrete report replaces the generic prerequisite hint.
    assert llm_module.local_context_hint(config) is None


def test_probe_accepts_an_adequate_loaded_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    RuntimeStubHandler.installed = ("qwen3:8b",)
    RuntimeStubHandler.loaded_name = None
    RuntimeStubHandler.loaded_context = llm_module.MIN_RECOMMENDED_LOCAL_CONTEXT
    server = stub_server(RuntimeStubHandler)
    try:
        monkeypatch.setenv(
            LOCAL_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/v1"
        )
        config, warning = resolve_runtime(ollama_config())
    finally:
        stop_server(server)

    assert warning is None
    assert config.is_available() is True
    assert config.local_context_window == llm_module.MIN_RECOMMENDED_LOCAL_CONTEXT
    assert llm_module.local_context_hint(config) is None


def test_probe_tolerates_a_server_without_the_native_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-Ollama backend has no ``/api/ps``; that must not be an error."""

    RuntimeStubHandler.installed = ("qwen3:8b",)
    RuntimeStubHandler.loaded_name = None
    RuntimeStubHandler.loaded_context = None
    server = stub_server(RuntimeStubHandler)
    try:
        monkeypatch.setenv(
            LOCAL_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/v1"
        )
        config, warning = resolve_runtime(ollama_config())
    finally:
        stop_server(server)

    assert warning is None
    assert config.is_available() is True
    assert config.local_context_window is None
    # With no observation, the generic prerequisite still applies.
    assert llm_module.local_context_hint(config) is not None


def test_probe_ignores_a_window_from_another_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loaded *other* model must not lend its window to the configured one."""

    RuntimeStubHandler.installed = ("qwen3:8b", "llama3.2:3b")
    RuntimeStubHandler.loaded_context = 4096
    RuntimeStubHandler.loaded_name = "llama3.2:3b"
    server = stub_server(RuntimeStubHandler)
    try:
        monkeypatch.setenv(
            LOCAL_BASE_URL_ENV, f"http://127.0.0.1:{server.server_port}/v1"
        )
        config, warning = resolve_runtime(ollama_config())
    finally:
        stop_server(server)

    assert warning is None
    assert config.local_context_window is None


# --------------------------------------------------------------------------
# Provider registry (P2-B): every default is driven by one PROVIDER_SPECS table
# --------------------------------------------------------------------------


def test_local_providers_are_derived_from_the_spec_table() -> None:
    specs = llm_module.PROVIDER_SPECS
    assert (
        frozenset(name for name, spec in specs.items() if spec.is_local)
        == llm_module.LOCAL_PROVIDERS
    )
    # The local engine(s) need no credential; remote ones do.
    for name, spec in specs.items():
        config = ModelConfig(provider=name, model=spec.default_model)
        assert config.requires_credential is (not spec.is_local)


def test_base_url_table_matches_specs_and_wire_defaults() -> None:
    specs = llm_module.PROVIDER_SPECS
    expected = {
        name: spec.base_url
        for name, spec in specs.items()
        if spec.base_url is not None
    }
    assert expected == llm_module.DEFAULT_BASE_URLS
    assert specs["ollama"].base_url == llm_module.DEFAULT_OLLAMA_BASE_URL
    assert specs["deepseek"].base_url == llm_module.DEFAULT_DEEPSEEK_BASE_URL


def test_default_lookup_helpers_read_the_table() -> None:
    specs = llm_module.PROVIDER_SPECS
    for name, spec in specs.items():
        assert default_model_for_provider(name) == spec.default_model
        assert default_max_tokens_for_provider(name) == spec.max_tokens
    # Unknown providers keep the historic "use the name itself / floor budget"
    # behaviour instead of raising.
    assert default_model_for_provider("acme") == "acme"
    assert default_max_tokens_for_provider("acme") == llm_module.FALLBACK_MAX_TOKENS


def test_only_openai_compatible_providers_take_the_compatible_path() -> None:
    specs = llm_module.PROVIDER_SPECS
    for name, spec in specs.items():
        compatible = bool(spec.base_url and spec.extra_body_builder)
        assert llm_module._is_compatible_provider(name) is compatible
        if compatible:
            base_url, builder, _placeholder = llm_module._compatible_spec(name)
            assert base_url == spec.base_url
            assert builder is spec.extra_body_builder
    # Generic-SDK providers must not be treated as OpenAI-compatible.
    assert llm_module._is_compatible_provider("anthropic") is False
    with pytest.raises(ValueError):
        llm_module._compatible_spec("anthropic")


def test_adding_a_provider_needs_only_a_table_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the registry: one entry, no dispatch edits.

    Register a brand-new OpenAI-compatible provider purely by mutating the
    table, then confirm the derived views and the factory pick it up with no
    change to any ``if``/``elif``.
    """

    from thesisev.llm import ProviderSpec, create_chat_model

    monkeypatch.setitem(
        llm_module.PROVIDER_SPECS,
        "acme",
        ProviderSpec(
            default_model="acme-1",
            base_url="http://127.0.0.1:9/v1",
            max_tokens=777,
            is_local=True,
            placeholder_api_key="acme-placeholder",
            extra_body_builder=lambda max_tokens: {"max_tokens": max_tokens},
        ),
    )
    config = build_model_config(provider="acme")
    # is_local / base_url are resolved live from the table, so a runtime entry
    # is honoured even though the import-time LOCAL_PROVIDERS snapshot is stale.
    assert config.is_local is True
    assert config.base_url == "http://127.0.0.1:9/v1"
    assert default_model_for_provider("acme") == "acme-1"
    assert default_max_tokens_for_provider("acme") == 777
    assert config.model == "acme-1"
    assert config.output_tokens == 777
    assert config.engine == ENGINE_LOCAL
    # It routes through the OpenAI-compatible client, not init_chat_model.
    assert llm_module._is_compatible_provider("acme") is True
    model = create_chat_model(config)
    assert model.model_name == "acme-1"
    assert model.extra_body == {"max_tokens": 777}
