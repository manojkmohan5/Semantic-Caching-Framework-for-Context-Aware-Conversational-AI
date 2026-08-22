"""The cache itself: exact tier, semantic tier, LRU eviction, TTL, drift repair."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .index import VectorIndex
from .store import Entry, Store

EXACT = "exact"
SEMANTIC = "semantic"
DEGRADED = "degraded"


@dataclass
class Hit:
    entry: Entry
    kind: str  # exact | semantic | degraded
    similarity: float | None  # None for an exact hit -- no vector was compared

    @property
    def response(self) -> str:
        return self.entry.response


class SemanticCache:
    """Composes the vector index and the metadata store.

    ponytail: one lock around all mutations. Right for a single-user REPL; shard
    per namespace if this ever serves concurrent traffic.
    """

    def __init__(self, cfg, embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.namespace = embedder.namespace
        cfg.ensure_dirs()
        self.store = Store(cfg.db_path, self.namespace)
        self.index = VectorIndex(
            embedder.dim, cfg.index_dir / f"{self.namespace}.faiss"
        )
        self._lock = threading.Lock()
        self.evicted = 0
        self._repair_drift()
        self._expire()

    # ------------------------------------------------------------- consistency
    def _repair_drift(self) -> None:
        """SQLite and FAISS are two files that can disagree if a process died
        mid-write. SQLite wins -- it holds the vectors."""
        if self.store.count() == len(self.index):
            return
        ids, vectors = self.store.all_vectors(self.embedder.dim)
        self.index.rebuild(ids, vectors)

    def _expire(self) -> None:
        if not self.cfg.ttl_seconds:
            return
        stale = self.store.expired_ids(self.cfg.ttl_seconds)
        if stale:
            self._drop(stale)

    # -------------------------------------------------------------------- read
    def lookup(self, prompt: str, vector: np.ndarray | None, session_id: str):
        """Return (hit_or_None, best_score_seen).

        best_score is reported even when it loses, because on a miss it is the
        single most useful tuning signal: it says whether the threshold was the
        reason you missed.
        """
        exact = self.store.by_hash(prompt)
        if exact is not None and self._in_scope(exact, session_id):
            self.store.touch(exact.id)
            return Hit(exact, EXACT, None), 1.0

        if vector is None:
            return None, 0.0

        best_score = 0.0
        for entry_id, score in self.index.search(vector, self.cfg.top_k):
            entry = self.store.by_id(entry_id)
            if entry is None or not self._in_scope(entry, session_id):
                continue
            best_score = max(best_score, score)
            if score >= self.cfg.threshold:
                self.store.touch(entry.id)
                return Hit(entry, SEMANTIC, score), score
        return None, best_score

    def fallback_lookup(self, vector: np.ndarray | None, session_id: str):
        """Best match above the *lower* bar, used only when the provider is
        unreachable. Answering slightly off beats not answering at all -- but it
        is reported as degraded and never counted as a clean hit."""
        if vector is None:
            return None
        for entry_id, score in self.index.search(vector, self.cfg.top_k):
            entry = self.store.by_id(entry_id)
            if entry is None or not self._in_scope(entry, session_id):
                continue
            if score >= self.cfg.fallback_threshold:
                self.store.touch(entry.id)
                return Hit(entry, DEGRADED, score)
        return None

    def _in_scope(self, entry: Entry, session_id: str) -> bool:
        return self.cfg.scope != "session" or entry.session_id == session_id

    # ------------------------------------------------------------------- write
    def put(
        self,
        *,
        prompt: str,
        response: str,
        vector: np.ndarray,
        session_id: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        response_tokens: int,
    ) -> int | None:
        """Store an answer, then evict if over capacity. Returns the new id, or
        None if the response was rejected as uncacheable."""
        if not self.is_cacheable(response):
            return None
        with self._lock:
            entry_id = self.store.add(
                prompt=prompt,
                response=response,
                vector=vector,
                session_id=session_id,
                provider=provider,
                model=model,
                embedder=self.embedder.id,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
            )
            self.index.add([entry_id], vector)
            self._evict_if_full()
            self.index.save(force=self.index.pending >= self.cfg.flush_every)
        return entry_id

    def is_cacheable(self, response: str) -> bool:
        """An empty or oversized answer cached once would be served forever."""
        if not response or not response.strip():
            return False
        return len(response.encode("utf-8")) <= self.cfg.max_cache_bytes

    def _evict_if_full(self) -> None:
        over = self.store.count() - self.cfg.max_entries
        if over > 0:
            self.evicted += len(self._drop(self.store.lru_ids(over)))

    def _drop(self, ids: list[int]) -> list[int]:
        """Remove from both stores. Doing one without the other is the drift bug
        _repair_drift exists to clean up, so they stay together here."""
        if not ids:
            return []
        self.index.remove(ids)
        self.store.delete(ids)
        return ids

    # ------------------------------------------------------------------- admin
    def stats(self) -> dict:
        return {
            "entries": self.store.count(),
            "capacity": self.cfg.max_entries,
            "evicted": self.evicted,
            "bytes": self.store.size_bytes(),
            "top_reused": self.store.top_reused(),
            "namespace": self.namespace,
            "backend": self.index.backend,
        }

    def clear(self) -> int:
        with self._lock:
            removed = self.store.clear()
            self.index.rebuild([], np.zeros((0, self.embedder.dim), dtype="float32"))
            self.evicted = 0
        return removed

    def close(self) -> None:
        with self._lock:
            self.index.save(force=True)
            self.store.close()
