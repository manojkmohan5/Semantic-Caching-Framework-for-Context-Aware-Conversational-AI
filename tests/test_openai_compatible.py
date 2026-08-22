"""The OpenAI wire path in front of a local mock endpoint.

This path serves openai plus every OpenAI-compatible host (grok, deepseek, kimi,
glm, nvidia, groq, openrouter, together, ollama, lmstudio, custom), so a break
here breaks eleven providers at once. The mock keeps it offline and free.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from semcache import SemCache
from semcache.config import Config
from semcache.providers import (
    OPENAI_COMPATIBLE,
    ProviderError,
    build_provider,
    detect_provider,
)

CHUNKS = ["A ", "mock ", "answer ", "about ", "caching."]
ANSWER = "".join(CHUNKS)


class _Handler(BaseHTTPRequestHandler):
    calls: list = []

    def log_message(self, *_a):  # keep pytest output clean
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append((self.path, body))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(payload):
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        for word in CHUNKS:
            send({"choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]})
        send({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        # Token counts only arrive in a streamed response when the client asks
        # for them via stream_options -- this asserts we do.
        send({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 22}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture
def endpoint():
    _Handler.calls = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


@pytest.mark.parametrize("provider", ["grok", "deepseek", "kimi", "glm", "nvidia"])
def test_compatible_host_answers_then_caches(endpoint, tmp_path, provider):
    with SemCache(
        "sk-fake",
        provider=provider,
        base_url=endpoint,
        model="test-model-1",
        embedder="hash",
        home=tmp_path,
        project=provider,
        threshold=0.45,
    ) as cache:
        first = cache.ask("what causes cache misses on large carts")
        assert first.source == "miss"
        assert first.text == ANSWER
        assert (first.input_tokens, first.output_tokens) == (11, 22)

        again = cache.ask("cache misses large carts what causes")
        assert again.from_cache
        assert again.source == "semantic"
        assert again.text == ANSWER

    # One model call for two questions: the point of the whole project.
    assert len(_Handler.calls) == 1
    path, body = _Handler.calls[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "test-model-1"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_provider_name_is_recorded_not_flattened_to_openai(endpoint, tmp_path):
    with SemCache(
        "sk-fake",
        provider="deepseek",
        base_url=endpoint,
        model="deepseek-chat",
        embedder="hash",
        home=tmp_path,
    ) as cache:
        cache.ask("hello there world")
        # /stats must say deepseek, not openai.
        assert cache.metrics.session[0].provider == "deepseek"


def test_key_prefixes_that_are_unambiguous():
    assert detect_provider("xai-abc") == "grok"
    assert detect_provider("nvapi-abc") == "nvidia"
    # Must be checked before the bare "sk-" branch, or an OpenRouter key is
    # sent to api.openai.com and fails with a confusing auth error.
    assert detect_provider("sk-or-v1-abc") == "openrouter"
    assert detect_provider("sk-ant-abc") == "anthropic"
    # DeepSeek and Kimi also issue sk-... keys, so a prefix cannot disambiguate
    # them from OpenAI -- those require an explicit --provider.
    assert detect_provider("sk-abc") == "openai"


def test_every_preset_has_a_usable_endpoint():
    for name, url in OPENAI_COMPATIBLE.items():
        if name == "custom":
            continue
        assert url.startswith("http"), name
        provider = build_provider(name, "k", "m", Config.load())
        assert provider.base_url == url
        assert provider.name == name


def test_custom_requires_an_explicit_endpoint():
    with pytest.raises(ProviderError, match="base-url"):
        build_provider("custom", "k", "m", Config.load())


def test_base_url_overrides_the_preset():
    cfg = Config.load(base_url="https://my-gateway.internal/v1")
    assert build_provider("grok", "k", "m", cfg).base_url == ("https://my-gateway.internal/v1")


def test_provider_errors_are_one_readable_line():
    """SDKs stringify to the whole JSON body. Printing that put a wall of braces
    in the chat on every turn while the condition lasted."""
    from semcache.providers import _summarize

    raw = (
        "Error code: 402 - {'error': {'message': 'Insufficient credits. This "
        "account never purchased credits. Make sure your key is on the correct "
        "account, and purchase more at https://openrouter.ai/settings/credits', "
        "'code': 402}}"
    )
    assert _summarize(raw) == "Insufficient credits"

    payment = ProviderError(raw, "payment")
    assert "no credits" in payment.user_message
    assert "{" not in payment.user_message  # no JSON reaches the user
    assert "--model" in payment.user_message  # and it says what to do

    missing = ProviderError("nope", "not_found", detail="bad/model-1")
    assert "bad/model-1" in missing.user_message

    # An unmapped error still collapses to a sentence, not a blob.
    assert ProviderError(raw, "error").user_message == "Insufficient credits"
