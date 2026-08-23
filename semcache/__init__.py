"""semcache -- a semantic caching layer that sits in front of any LLM chat model.

Use it from the terminal (`semcache`), or drop it into your own project:

    from semcache import SemCache

    cache = SemCache(api_key="sk-ant-...", project="my-service")
    answer = cache.ask("What causes memory fragmentation in Python?")
    answer = cache.ask("Why do Python services fragment the heap?")  # served locally
    print(cache.report())

Answers you have effectively asked before come back in milliseconds without an
API call, and every request is measured.
"""

from __future__ import annotations

import os as _os

# Set before anything can import huggingface_hub: its progress bars are decided
# at import time, and a download bar from the background warm-up thread printed
# straight through the dashboard.
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"

__all__ = ["SemCache", "Answer", "__version__"]


class Answer:
    """The result of one `ask`, with enough detail to log or display."""

    __slots__ = (
        "text",
        "source",
        "similarity",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "cost_saved_usd",
        "model",
        "note",
    )

    def __init__(self, turn):
        rec = turn.record
        self.text = turn.text
        #: "exact" | "semantic" | "degraded" | "miss" | "error"
        self.source = turn.outcome
        self.similarity = rec.similarity
        self.latency_ms = rec.total_ms
        self.input_tokens = rec.prompt_tokens
        self.output_tokens = rec.response_tokens
        self.cost_usd = rec.cost_usd
        self.cost_saved_usd = rec.cost_saved_usd
        self.model = rec.model
        self.note = turn.note

    @property
    def from_cache(self) -> bool:
        return self.source in {"exact", "semantic", "degraded"}

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return (
            f"Answer(source={self.source!r}, latency_ms={self.latency_ms:.1f}, "
            f"from_cache={self.from_cache})"
        )


class SemCache:
    """Embeddable façade over the same pipeline the CLI uses.

    Everything is optional: with no arguments it reads the key from the
    environment (ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY), picks the
    provider from the key's shape, and stores the cache under ~/.semcache.

    Not thread-safe -- construct one per worker. Call close() when done, or use
    it as a context manager, so the index is flushed to disk.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        project: str | None = None,
        session_id: str = "default",
        offline: bool = False,
        **options,
    ):
        import os

        from .cache import SemanticCache
        from .chat import ChatSession
        from .config import Config
        from .embedders import build_embedder
        from .metrics import Metrics
        from .providers import build_provider, default_model, detect_provider

        cfg = Config.load(
            project=project, offline=offline or None, model=model, provider=provider, **options
        )

        key = api_key or ""
        if not key and not cfg.offline:
            for var in (
                "SEMCACHE_API_KEY",
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "GEMINI_API_KEY",
            ):
                key = os.environ.get(var) or ""
                if key:
                    break

        name = cfg.provider or provider or detect_provider(key) or "stub"
        chosen = cfg.model or model or ("stub-1" if name == "stub" else default_model(name))

        self.config = cfg
        self._provider = build_provider(name, key, chosen, cfg)
        self._embedder = build_embedder(cfg, self._provider)
        self.cache = SemanticCache(cfg, self._embedder)
        self.metrics = Metrics(cfg.metrics_path, log_prompts=cfg.log_prompts)
        self._session = ChatSession(
            cfg, self._provider, self._embedder, self.cache, self.metrics, session_id
        )

    def ask(self, prompt: str, on_chunk=None) -> Answer:
        """Answer one prompt, from the cache when possible.

        `on_chunk` is called with each piece of streamed text as it arrives.
        """
        return Answer(self._session.ask(prompt, on_chunk=on_chunk))

    def warm(self) -> None:
        """Load the embedding model now rather than on the first ask."""
        self._embedder.warm()

    def stats(self) -> dict:
        agg = self.metrics.aggregate()
        cache = self.cache.stats()
        return {
            "requests": agg.total,
            "from_cache": agg.hits,
            "hit_rate": agg.hit_rate,
            "calls_avoided": agg.calls_avoided,
            "cost_spent": agg.cost_spent,
            "cost_saved": agg.cost_saved,
            "entries": cache["entries"],
            "evicted": cache["evicted"],
        }

    def report(self) -> str:
        """The same plain-English summary the CLI prints for /stats."""
        from .metrics import render_plain

        return render_plain(self.metrics.aggregate(), self.cache.stats())

    def clear(self) -> int:
        return self.cache.clear()

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> SemCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
