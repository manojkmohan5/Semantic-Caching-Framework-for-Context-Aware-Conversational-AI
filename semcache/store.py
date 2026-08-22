"""SQLite metadata: the exact-match tier, LRU ordering, and the vector of record.

Vectors live here as BLOBs as well as in FAISS. That redundancy is deliberate:
it makes SQLite the single source of truth, so a lost or corrupt index file is a
rebuild rather than data loss.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SCHEMA_VERSION

_WS = re.compile(r"\s+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace       TEXT    NOT NULL,
    prompt_hash     TEXT    NOT NULL,
    prompt          TEXT    NOT NULL,
    response        TEXT    NOT NULL,
    vector          BLOB    NOT NULL,
    session_id      TEXT,
    provider        TEXT,
    model           TEXT,
    embedder        TEXT,
    created_at      REAL    NOT NULL,
    last_used_at    REAL    NOT NULL,
    hit_count       INTEGER NOT NULL DEFAULT 0,
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    response_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_hash ON entries(namespace, prompt_hash);
CREATE INDEX IF NOT EXISTS idx_lru  ON entries(namespace, last_used_at);
"""


def normalize(prompt: str) -> str:
    """Canonical form for exact matching. NORMALIZE_VERSION guards changes here."""
    return _WS.sub(" ", (prompt or "").strip()).casefold()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(normalize(prompt).encode("utf-8")).hexdigest()


@dataclass
class Entry:
    id: int
    prompt: str
    response: str
    session_id: str
    provider: str
    model: str
    created_at: float
    last_used_at: float
    hit_count: int
    prompt_tokens: int
    response_tokens: int


_COLS = (
    "id, prompt, response, session_id, provider, model, "
    "created_at, last_used_at, hit_count, prompt_tokens, response_tokens"
)


def _row(row) -> Entry:
    return Entry(*row)


class Store:
    def __init__(self, path: Path, namespace: str):
        self.path = Path(path)
        self.namespace = namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so a background warm-up thread can read.
        # timeout: WAL allows one writer at a time, so a second semcache process
        # must wait rather than fail with "database is locked".
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=15.0)
        # WAL gives crash-atomic commits and lets readers run during writes.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._check_schema()
        self.db.commit()

    def _check_schema(self) -> None:
        cur = self.db.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if row is None:
            self.db.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        elif int(row[0]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"cache at {self.path} uses schema v{row[0]}, this build expects "
                f"v{SCHEMA_VERSION}. Move it aside or run: semcache clear"
            )

    # ------------------------------------------------------------------- reads
    def by_hash(self, prompt: str) -> Entry | None:
        """The exact-match tier. Costs no embedding at all."""
        cur = self.db.execute(
            f"SELECT {_COLS} FROM entries WHERE namespace=? AND prompt_hash=? "
            "ORDER BY last_used_at DESC LIMIT 1",
            (self.namespace, prompt_hash(prompt)),
        )
        row = cur.fetchone()
        return _row(row) if row else None

    def by_id(self, entry_id: int) -> Entry | None:
        cur = self.db.execute(
            f"SELECT {_COLS} FROM entries WHERE id=? AND namespace=?",
            (entry_id, self.namespace),
        )
        row = cur.fetchone()
        return _row(row) if row else None

    def count(self) -> int:
        cur = self.db.execute("SELECT COUNT(*) FROM entries WHERE namespace=?", (self.namespace,))
        return int(cur.fetchone()[0])

    def all_vectors(self, dim: int) -> tuple[list[int], np.ndarray]:
        """Everything needed to rebuild the FAISS index from scratch."""
        cur = self.db.execute(
            "SELECT id, vector FROM entries WHERE namespace=? ORDER BY id",
            (self.namespace,),
        )
        ids, blobs = [], []
        for entry_id, blob in cur:
            ids.append(int(entry_id))
            blobs.append(np.frombuffer(blob, dtype="float32"))
        if not ids:
            return [], np.zeros((0, dim), dtype="float32")
        return ids, np.vstack(blobs)

    def top_reused(self, limit: int = 3) -> list[tuple[str, int]]:
        cur = self.db.execute(
            "SELECT prompt, hit_count FROM entries WHERE namespace=? AND hit_count > 0 "
            "ORDER BY hit_count DESC, last_used_at DESC LIMIT ?",
            (self.namespace, limit),
        )
        return [(p, int(h)) for p, h in cur]

    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    # ------------------------------------------------------------------ writes
    def add(
        self,
        *,
        prompt: str,
        response: str,
        vector: np.ndarray,
        session_id: str,
        provider: str,
        model: str,
        embedder: str,
        prompt_tokens: int,
        response_tokens: int,
    ) -> int:
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO entries (namespace, prompt_hash, prompt, response, vector,"
            " session_id, provider, model, embedder, created_at, last_used_at,"
            " hit_count, prompt_tokens, response_tokens)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?,?)",
            (
                self.namespace,
                prompt_hash(prompt),
                prompt,
                response,
                np.ascontiguousarray(vector, dtype="float32").tobytes(),
                session_id,
                provider,
                model,
                embedder,
                now,
                now,
                prompt_tokens,
                response_tokens,
            ),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def touch(self, entry_id: int) -> None:
        """Record a reuse. This is what makes eviction LRU rather than FIFO."""
        self.db.execute(
            "UPDATE entries SET last_used_at=?, hit_count=hit_count+1 WHERE id=?",
            (time.time(), entry_id),
        )
        self.db.commit()

    def lru_ids(self, over_by: int) -> list[int]:
        if over_by <= 0:
            return []
        cur = self.db.execute(
            "SELECT id FROM entries WHERE namespace=? ORDER BY last_used_at ASC LIMIT ?",
            (self.namespace, over_by),
        )
        return [int(r[0]) for r in cur]

    def expired_ids(self, ttl_seconds: float) -> list[int]:
        cutoff = time.time() - ttl_seconds
        cur = self.db.execute(
            "SELECT id FROM entries WHERE namespace=? AND created_at < ?",
            (self.namespace, cutoff),
        )
        return [int(r[0]) for r in cur]

    def delete(self, ids: list[int]) -> int:
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        cur = self.db.execute(f"DELETE FROM entries WHERE id IN ({marks})", tuple(ids))
        self.db.commit()
        return cur.rowcount or 0

    def clear(self) -> int:
        cur = self.db.execute("DELETE FROM entries WHERE namespace=?", (self.namespace,))
        self.db.commit()
        return cur.rowcount or 0

    def close(self) -> None:
        try:
            self.db.commit()
        finally:
            self.db.close()
