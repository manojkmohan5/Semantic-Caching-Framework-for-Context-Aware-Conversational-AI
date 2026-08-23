"""Chat providers behind one streaming interface, picked from the shape of the API key.

Each provider uses its vendor's official SDK, imported lazily so that installing
one provider never drags in the other two. The SDKs already retry 429/5xx with
backoff and honour Retry-After, so this module does not hand-roll any of that --
it only configures them and maps their exceptions onto one error type.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

# ---------------------------------------------------------------- model catalog


@dataclass(frozen=True)
class ModelInfo:
    id: str
    blurb: str
    #: USD per million tokens. None means "unknown to us" -- we report no cost
    #: rather than inventing a number. Override in config.json to enable costs.
    price_in: float | None = None
    price_out: float | None = None
    #: Current Anthropic models removed the temperature parameter entirely and
    #: return 400 if it is sent. Not "deprecated" -- rejected.
    supports_temperature: bool = True


# Anthropic prices are first-party API rates as documented 2026-06-24.
# OpenAI and Google prices are intentionally left None: shipping a stale or
# guessed price would make the savings figures dishonest. Set them in
# ~/.semcache/config.json under "prices" to turn cost reporting on.
CATALOG: dict[str, list[ModelInfo]] = {
    "anthropic": [
        ModelInfo("claude-sonnet-5", "balanced, fastest of the three", 3.00, 15.00, False),
        ModelInfo("claude-opus-5", "most capable", 5.00, 25.00, False),
        ModelInfo("claude-haiku-4-5", "cheapest, lowest latency", 1.00, 5.00, True),
    ],
    "openai": [
        ModelInfo("gpt-5-mini", "small and fast"),
        ModelInfo("gpt-5", "most capable"),
    ],
    "gemini": [
        ModelInfo("gemini-2.5-flash", "fast and cheap"),
        ModelInfo("gemini-2.5-pro", "most capable"),
    ],
}

#: Providers that speak the OpenAI wire format. They all work through the same
#: client -- only the base URL differs -- so supporting them costs a dict rather
#: than a new provider class each. Override any of these with --base-url.
OPENAI_COMPATIBLE = {
    "grok": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "kimi": "https://api.moonshot.ai/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "custom": "",  # requires --base-url
}

#: Input context windows, only for models we can state with confidence. Anything
#: absent shows raw token counts instead of a percentage -- an invented window
#: would make the status bar lie.
CONTEXT_WINDOWS = {
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-haiku-4-5": 200_000,
}


def context_window(model: str) -> int | None:
    return CONTEXT_WINDOWS.get(model)


EMBEDDING_MODELS = {
    "openai": ("text-embedding-3-small", 1536),
    "gemini": ("gemini-embedding-001", 3072),
}


def detect_provider(api_key: str) -> str | None:
    """Guess the provider from the key prefix. Always confirmed by the user."""
    key = (api_key or "").strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    # Check the longer, unambiguous "sk-" variants before the bare prefix,
    # otherwise an OpenRouter key gets sent to api.openai.com and fails with a
    # confusing auth error.
    if key.startswith("sk-or-"):
        return "openrouter"
    if key.startswith(("sk-", "sess-")):
        return "openai"
    if key.startswith("AIza"):
        return "gemini"
    if key.startswith("nvapi-"):
        return "nvidia"
    if key.startswith("xai-"):
        return "grok"
    # DeepSeek, Kimi and most OpenAI-compatible hosts also issue "sk-..." keys,
    # so a prefix cannot tell them apart from OpenAI. Those need --provider.
    return None


def default_model(provider: str) -> str:
    return CATALOG[provider][0].id


def model_info(provider: str, model: str) -> ModelInfo:
    for info in CATALOG.get(provider, []):
        if info.id == model:
            return info
    # A model we do not know about is still usable -- we just cannot price it,
    # and we must not assume it tolerates `temperature`.
    return ModelInfo(model, "custom", None, None, provider != "anthropic")


def needs_explicit_model(provider: str) -> bool:
    """OpenAI-compatible hosts each have their own model names, which change
    often. Rather than ship a guess that 400s, ask once and save it."""
    return provider in OPENAI_COMPATIBLE


# --------------------------------------------------------------------- results


@dataclass
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    model: str

    @property
    def truncated(self) -> bool:
        """A truncated answer must never be cached -- it would be served forever."""
        return self.stop_reason in {"max_tokens", "length", "MAX_TOKENS"}


class ProviderError(Exception):
    """One error type for every SDK, carrying whether a retry could ever help."""

    def __init__(
        self,
        message: str,
        kind: str = "error",
        *,
        transient: bool = False,
        detail: str = "",
    ):
        super().__init__(message)
        # rate_limit | auth | payment | not_found | network | timeout | error
        self.kind = kind
        self.transient = transient
        #: Short human-readable summary, without the provider's JSON envelope.
        self.detail = detail or _summarize(message)

    @property
    def user_message(self) -> str:
        fixed = {
            "rate_limit": "the model is rate-limited",
            "auth": "the API key was rejected",
            "network": "cannot reach the model",
            "timeout": "the model timed out",
            "payment": "this account has no credits for that model"
            " -- add credits, or pick a cheaper/free model with --model",
            "not_found": f"the provider does not have a model called {self.detail!r}"
            " -- check the name and pass --model",
        }
        if self.kind in fixed:
            return fixed[self.kind]
        # Unknown failures still get one readable line rather than a wall of
        # provider JSON repeated on every turn.
        return self.detail or str(self)


def _summarize(text: str) -> str:
    """Pull the human sentence out of a provider error.

    SDKs stringify to the whole JSON body, so printing str(exc) put a wall of
    braces in the middle of the chat -- on every single turn while the condition
    lasted.
    """
    raw = str(text)
    for marker in ("'message': '", '"message": "'):
        if marker in raw:
            rest = raw.split(marker, 1)[1]
            end = rest.find("'") if marker.endswith("'") else rest.find('"')
            if end > 0:
                sentence = rest[:end]
                # Keep it to the first sentence; the rest is usually a URL hint.
                return sentence.split(". ")[0].strip().rstrip(".")
    return raw.splitlines()[0][:200]


def _model_of(exc) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("param") or err.get("code") or "")
    return ""


def _redact(text: str) -> str:
    """Strip anything key-shaped out of an error before it is shown or logged."""
    out = []
    for word in str(text).split():
        if len(word) > 20 and word.startswith(("sk-", "sk-ant-", "AIza", "sess-")):
            out.append(word[:7] + "...redacted")
        else:
            out.append(word)
    return " ".join(out)


# ------------------------------------------------------------------- providers


class Provider:
    """Base class. One instance per session; `stream_chat` is not re-entrant.

    ponytail: single-threaded by construction (one REPL, one provider instance).
    Give each caller its own instance if this ever serves concurrent requests.
    """

    name = "base"

    def __init__(self, api_key: str, model: str, cfg, base_url: str | None = None):
        self.api_key = api_key
        self.model = model
        self.cfg = cfg
        #: Set for OpenAI-compatible hosts (DeepSeek, Kimi, GLM, NVIDIA, Ollama...).
        self.base_url = base_url
        self.result: ChatResult | None = None

    def stream_chat(self, prompt: str, history: list[dict]) -> Iterator[str]:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ProviderError(f"{self.name} has no embedding API", "error")

    # Chat history is sent as plain alternating turns -- every SDK accepts that shape.
    def _messages(self, prompt: str, history: list[dict]) -> list[dict]:
        return [*history, {"role": "user", "content": prompt}]


class AnthropicProvider(Provider):
    name = "anthropic"

    def _client(self):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - install-time path
            raise ProviderError(
                'anthropic SDK not installed. Run: pip install "semcache[anthropic]"', "error"
            ) from exc
        self._sdk = anthropic
        return anthropic.Anthropic(
            api_key=self.api_key,
            timeout=self.cfg.timeout_read,
            max_retries=self.cfg.max_retries,
        )

    def stream_chat(self, prompt: str, history: list[dict]) -> Iterator[str]:
        client = self._client()
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.cfg.max_tokens,
            "messages": self._messages(prompt, history),
        }
        # Current Anthropic models reject `temperature` with a 400.
        if model_info("anthropic", self.model).supports_temperature:
            kwargs["temperature"] = self.cfg.temperature
        if self.cfg.effort:
            kwargs["output_config"] = {"effort": self.cfg.effort}

        try:
            with client.messages.stream(**kwargs) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except Exception as exc:
            raise _map_anthropic(self._sdk, exc) from exc

        self.result = ChatResult(
            text="".join(b.text for b in final.content if b.type == "text"),
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            stop_reason=final.stop_reason or "end_turn",
            model=final.model,
        )


class OpenAIProvider(Provider):
    name = "openai"

    def _client(self):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                'openai SDK not installed. Run: pip install "semcache[openai]"', "error"
            ) from exc
        self._sdk = openai
        kwargs = {
            "api_key": self.api_key or "not-needed",
            "timeout": self.cfg.timeout_read,
            "max_retries": self.cfg.max_retries,
        }
        base_url = self.base_url or getattr(self.cfg, "base_url", None)
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)

    def stream_chat(self, prompt: str, history: list[dict]) -> Iterator[str]:
        client = self._client()
        chunks: list[str] = []
        usage = None
        finish = "stop"
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=self._messages(prompt, history),
                max_completion_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                stream=True,
                # Without this, a streamed response carries no token counts at all.
                stream_options={"include_usage": True},
            )
            for event in stream:
                if event.usage:
                    usage = event.usage
                for choice in event.choices or []:
                    if choice.finish_reason:
                        finish = choice.finish_reason
                    piece = choice.delta.content if choice.delta else None
                    if piece:
                        chunks.append(piece)
                        yield piece
        except Exception as exc:
            raise _map_openai(self._sdk, exc) from exc

        self.result = ChatResult(
            text="".join(chunks),
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            stop_reason=finish,
            model=self.model,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._client()
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODELS["openai"][0], input=texts)
        except Exception as exc:
            raise _map_openai(self._sdk, exc) from exc
        return [item.embedding for item in resp.data]


class GeminiProvider(Provider):
    name = "gemini"

    def _client(self):
        try:
            from google import genai
            from google.genai import errors as genai_errors
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                'google-genai SDK not installed. Run: pip install "semcache[gemini]"', "error"
            ) from exc
        self._errors = genai_errors
        return genai.Client(api_key=self.api_key)

    def stream_chat(self, prompt: str, history: list[dict]) -> Iterator[str]:
        client = self._client()
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in self._messages(prompt, history)
        ]
        chunks: list[str] = []
        usage = None
        finish = "STOP"
        try:
            for event in client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config={
                    "temperature": self.cfg.temperature,
                    "max_output_tokens": self.cfg.max_tokens,
                },
            ):
                if getattr(event, "usage_metadata", None):
                    usage = event.usage_metadata
                for cand in getattr(event, "candidates", None) or []:
                    if cand.finish_reason:
                        finish = str(cand.finish_reason)
                if event.text:
                    chunks.append(event.text)
                    yield event.text
        except Exception as exc:
            raise _map_gemini(self._errors, exc) from exc

        self.result = ChatResult(
            text="".join(chunks),
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            stop_reason=finish,
            model=self.model,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._client()
        try:
            resp = client.models.embed_content(model=EMBEDDING_MODELS["gemini"][0], contents=texts)
        except Exception as exc:
            raise _map_gemini(self._errors, exc) from exc
        return [list(e.values) for e in resp.embeddings]


class StubProvider(Provider):
    """Deterministic canned answers. Powers the test suite and `bench --offline`,
    so CI never needs a key, a network, or a cent."""

    name = "stub"
    delay = 0.8

    def __init__(self, api_key: str = "", model: str = "stub-1", cfg=None):
        super().__init__(api_key, model, cfg)

    def stream_chat(self, prompt: str, history: list[dict]) -> Iterator[str]:
        text = f"Stub answer about {prompt.strip()[:60]}. " * 3
        per_chunk = self.delay / 8
        for word in text.split(" "):
            time.sleep(per_chunk / 8)
            yield word + " "
        self.result = ChatResult(
            text=text,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
            stop_reason="end_turn",
            model=self.model,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        from .embedders import HashEmbedder

        return [list(v) for v in HashEmbedder().encode(texts)]


_CLASSES = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "stub": StubProvider,
}


def build_provider(provider: str, api_key: str, model: str, cfg) -> Provider:
    if cfg is not None and getattr(cfg, "offline", False):
        return StubProvider(cfg=cfg)

    if provider in OPENAI_COMPATIBLE:
        base_url = getattr(cfg, "base_url", None) or OPENAI_COMPATIBLE[provider]
        if not base_url:
            raise ProviderError(
                f"{provider} needs an endpoint. Pass --base-url https://...", "error"
            )
        host = OpenAIProvider(api_key, model, cfg, base_url=base_url)
        host.name = provider  # so metrics and /stats say "deepseek", not "openai"
        return host

    try:
        cls = _CLASSES[provider]
    except KeyError:
        known = ", ".join([*_CLASSES, *OPENAI_COMPATIBLE])
        raise ProviderError(f"unknown provider {provider!r}. Known: {known}", "error") from None
    return cls(api_key, model, cfg)


def all_provider_names() -> list[str]:
    return [*CATALOG, *OPENAI_COMPATIBLE]


# ------------------------------------------------------- exception translation


def _map_anthropic(sdk, exc: Exception) -> ProviderError:
    msg = _redact(exc)
    if isinstance(exc, sdk.RateLimitError):
        return ProviderError(msg, "rate_limit", transient=True)
    if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return ProviderError(msg, "auth")
    if isinstance(exc, sdk.APITimeoutError):
        return ProviderError(msg, "timeout", transient=True)
    if isinstance(exc, sdk.APIConnectionError):
        return ProviderError(msg, "network", transient=True)
    if isinstance(exc, sdk.APIStatusError):
        if exc.status_code == 402:
            return ProviderError(msg, "payment")
        if exc.status_code == 404:
            return ProviderError(msg, "not_found", detail=_model_of(exc))
        return ProviderError(msg, "error", transient=exc.status_code >= 500)
    return ProviderError(msg, "error")


def _map_openai(sdk, exc: Exception) -> ProviderError:
    msg = _redact(exc)
    if isinstance(exc, sdk.RateLimitError):
        return ProviderError(msg, "rate_limit", transient=True)
    if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
        return ProviderError(msg, "auth")
    if isinstance(exc, sdk.APITimeoutError):
        return ProviderError(msg, "timeout", transient=True)
    if isinstance(exc, sdk.APIConnectionError):
        return ProviderError(msg, "network", transient=True)
    if isinstance(exc, sdk.APIStatusError):
        if exc.status_code == 402:
            return ProviderError(msg, "payment")
        if exc.status_code == 404:
            return ProviderError(msg, "not_found", detail=_model_of(exc))
        return ProviderError(msg, "error", transient=exc.status_code >= 500)
    return ProviderError(msg, "error")


def _map_gemini(errors, exc: Exception) -> ProviderError:
    msg = _redact(exc)
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return ProviderError(msg, "rate_limit", transient=True)
    if code == 402:
        return ProviderError(msg, "payment")
    if code == 404:
        return ProviderError(msg, "not_found")
    if code in (401, 403):
        return ProviderError(msg, "auth")
    if isinstance(exc, getattr(errors, "ServerError", ())):
        return ProviderError(msg, "error", transient=True)
    if isinstance(exc, (TimeoutError,)):
        return ProviderError(msg, "timeout", transient=True)
    if isinstance(exc, (ConnectionError, OSError)) and not isinstance(
        exc, getattr(errors, "APIError", ())
    ):
        return ProviderError(msg, "network", transient=True)
    return ProviderError(msg, "error")
