"""The request pipeline: normalize, exact tier, semantic tier, model, store.

This is the whole product. Everything else in the package serves this file.
"""

from __future__ import annotations

import contextlib
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable

from .cache import SemanticCache
from .metrics import Metrics, Record, estimate_cost
from .providers import ProviderError, model_info

#: Prompts that cannot stand alone need conversation context folded in before
#: embedding. Everything else is embedded bare, which is what lets two different
#: sessions share one answer -- where the real savings are.
_ANAPHORA = re.compile(
    r"\b(it|its|that|this|they|them|those|these|he|she|him|her|"
    r"the (?:previous|above|last|former|latter)|"
    r"(?:previous|above|last) (?:one|answer|result|question))\b",
    re.IGNORECASE,
)
_MIN_STANDALONE_WORDS = 5


def needs_context(prompt: str) -> bool:
    """True when the prompt leans on earlier turns to make sense.

    ponytail: keyword heuristic, not coreference resolution. See bonus.md Part A
    for the limits this inherits.
    """
    words = prompt.split()
    return len(words) < _MIN_STANDALONE_WORDS or bool(_ANAPHORA.search(prompt))


@dataclass
class Turn:
    """One completed request, ready to print and to record."""

    outcome: str  # exact | semantic | degraded | miss | error
    text: str
    record: Record
    note: str = ""  # the human-readable tail of the status line

    @property
    def from_cache(self) -> bool:
        return self.outcome in {"exact", "semantic", "degraded"}


class ChatSession:
    """Owns the cache, the provider and the conversation history for one session."""

    def __init__(
        self,
        cfg,
        provider,
        embedder,
        cache: SemanticCache,
        metrics: Metrics,
        session_id: str = "default",
    ):
        self.cfg = cfg
        self.provider = provider
        self.embedder = embedder
        self.cache = cache
        self.metrics = metrics
        self.session_id = session_id
        self.history: list[dict] = []
        self._warm_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ warmup
    def warm_async(self) -> None:
        """Load the embedding model while the user is still typing, so its
        one-time cost never shows up as latency on a real question."""

        def _warm():
            # A failed warm-up just means the first encode pays the load cost.
            with contextlib.suppress(Exception):
                self.embedder.warm()

        self._warm_thread = threading.Thread(target=_warm, daemon=True)
        self._warm_thread.start()

    # ------------------------------------------------------------------- embed
    def _embed_text(self, prompt: str) -> str:
        if not self.cfg.context_turns or not needs_context(prompt) or not self.history:
            return prompt
        recent = [t["content"] for t in self.history[-(self.cfg.context_turns * 2) :]]
        return " ".join([*recent, prompt]).strip()

    # ----------------------------------------------------------------- the flow
    def ask(self, prompt: str, on_chunk: Callable[[str], None] | None = None) -> Turn:
        """Answer one prompt. `on_chunk` receives streamed text as it arrives."""
        started = time.perf_counter()
        info = model_info(self.provider.name, self.provider.model)
        rec = Record(
            session_id=self.session_id,
            provider=self.provider.name,
            model=self.provider.model,
            embedder=self.embedder.id,
            prompt=prompt,
        )

        # --- tier 1: exact, no embedding at all
        hit, _ = self.cache.lookup(prompt, None, self.session_id)
        vector = None
        if hit is None:
            t0 = time.perf_counter()
            try:
                vector = self.embedder.encode_one(self._embed_text(prompt))
            except ProviderError as exc:
                return self._fail(rec, exc, started)
            rec.embed_ms = (time.perf_counter() - t0) * 1000

            # --- tier 2: semantic
            t0 = time.perf_counter()
            hit, best = self.cache.lookup(prompt, vector, self.session_id)
            rec.search_ms = (time.perf_counter() - t0) * 1000
            if hit is None and best > 0:
                rec.best_below = best

        if hit is not None:
            return self._from_cache(hit, rec, info, started, on_chunk)

        # --- tier 3: the model
        return self._from_model(prompt, vector, rec, info, started, on_chunk)

    # ------------------------------------------------------------- cache branch
    def _from_cache(self, hit, rec, info, started, on_chunk) -> Turn:
        text = hit.response
        if on_chunk:
            on_chunk(text)
        rec.outcome = hit.kind
        rec.similarity = hit.similarity
        rec.matched_id = hit.entry.id
        rec.prompt_tokens = hit.entry.prompt_tokens
        rec.response_tokens = hit.entry.response_tokens
        # What we did not pay for is exactly what the original call cost.
        rec.cost_saved_usd = estimate_cost(info, hit.entry.prompt_tokens, hit.entry.response_tokens)
        rec.total_ms = (time.perf_counter() - started) * 1000
        self._remember(rec.prompt, text)
        note = ""
        if hit.similarity is not None:
            note = f"{hit.similarity * 100:.0f}% match"
        if hit.entry.model and hit.entry.model != self.provider.model:
            note += (
                f" (answered by {hit.entry.model})" if note else f"answered by {hit.entry.model}"
            )
        return Turn(hit.kind, text, self.metrics.record(rec), note)

    # ------------------------------------------------------------- model branch
    def _from_model(self, prompt, vector, rec, info, started, on_chunk) -> Turn:
        chunks: list[str] = []
        first_token_at: float | None = None
        t0 = time.perf_counter()
        try:
            for piece in self.provider.stream_chat(prompt, self.history):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                chunks.append(piece)
                if on_chunk:
                    on_chunk(piece)
        except ProviderError as exc:
            return self._degrade_or_fail(exc, vector, rec, info, started, on_chunk)

        rec.llm_ms = (time.perf_counter() - t0) * 1000
        if first_token_at is not None:
            rec.first_token_ms = (first_token_at - started) * 1000

        result = self.provider.result
        text = (result.text if result else "".join(chunks)) or "".join(chunks)
        rec.outcome = "miss"
        rec.prompt_tokens = result.input_tokens if result else 0
        rec.response_tokens = result.output_tokens if result else 0
        rec.cost_usd = estimate_cost(info, rec.prompt_tokens, rec.response_tokens)
        rec.total_ms = (time.perf_counter() - started) * 1000

        note = ""
        if rec.best_below:
            note = f"closest saved answer was only {rec.best_below * 100:.0f}% match"

        # A truncated answer must never be cached -- it would be served forever.
        if result is not None and result.truncated:
            note = (note + " · " if note else "") + "answer hit the length limit, not saved"
        elif vector is not None:
            self.cache.put(
                prompt=prompt,
                response=text,
                vector=vector,
                session_id=self.session_id,
                provider=self.provider.name,
                model=self.provider.model,
                prompt_tokens=rec.prompt_tokens,
                response_tokens=rec.response_tokens,
            )

        self._remember(prompt, text)
        return Turn("miss", text, self.metrics.record(rec), note)

    # ---------------------------------------------------------- failure branch
    def _degrade_or_fail(self, exc, vector, rec, info, started, on_chunk) -> Turn:
        """The provider failed. Rather than dead-ending, serve the closest saved
        answer if one clears the lower bar -- clearly labelled, and never counted
        as a clean hit."""
        degraded = self.cache.fallback_lookup(vector, self.session_id)
        if degraded is not None:
            if on_chunk:
                on_chunk(degraded.response)
            rec.outcome = "degraded"
            rec.similarity = degraded.similarity
            rec.matched_id = degraded.entry.id
            rec.prompt_tokens = degraded.entry.prompt_tokens
            rec.response_tokens = degraded.entry.response_tokens
            rec.error = exc.kind
            rec.total_ms = (time.perf_counter() - started) * 1000
            self._remember(rec.prompt, degraded.response)
            note = (
                f"{degraded.similarity * 100:.0f}% match · "
                f"{exc.user_message}, so this is the closest saved answer"
            )
            return Turn("degraded", degraded.response, self.metrics.record(rec), note)
        return self._fail(rec, exc, started)

    def _fail(self, rec, exc, started) -> Turn:
        rec.outcome = "error"
        rec.error = exc.kind
        rec.total_ms = (time.perf_counter() - started) * 1000
        note = exc.user_message
        if rec.best_below:
            note += f" · no close saved answer (best was {rec.best_below * 100:.0f}%)"
        return Turn("error", "", self.metrics.record(rec), note)

    # ----------------------------------------------------------------- history
    def _remember(self, prompt: str | None, answer: str) -> None:
        if not prompt:
            return
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": answer})
        # Keep only what the context window of the embedder actually uses, plus
        # a little slack for the provider's own conversational continuity.
        limit = max(self.cfg.context_turns, 4) * 2
        if len(self.history) > limit:
            self.history = self.history[-limit:]


def stream_to(writer) -> Callable[[str], None]:
    """Adapter so callers can hand `ask()` a file-like sink."""

    def _write(piece: str) -> None:
        writer.write(piece)
        writer.flush()

    return _write


def iter_chunks(session: ChatSession, prompt: str) -> Iterator[str]:
    """Generator form, used by tests that want the pieces rather than a callback."""
    out: list[str] = []
    session.ask(prompt, on_chunk=out.append)
    yield from out
