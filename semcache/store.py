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

META_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Created after the version check, because the indexes below reference columns a
# migration may still need to add to an older table.
ENTRIES_SCHEMA = """
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
    response_tokens INTEGER NOT NULL DEFAULT 0,
    -- Monotonic use counter. last_used_at is a wall clock with ~15.6ms
    -- resolution on Windows, so entries written in quick succession tie and
    -- "ORDER BY last_used_at" could evict the most-recently-used entry. This
    -- never ties, so LRU ordering is exact.
    use_seq         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_hash ON entries(namespace, prompt_hash);
CREATE INDEX IF NOT EXISTS idx_lru  ON entries(namespace, use_seq);
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
        # Order matters: meta first so the version is readable, then any
        # migration, and only then the entries table and its indexes -- the LRU
        # index references a column that a v1 cache does not have yet.
        self.db.executescript(META_SCHEMA)
        self._check_schema()
        self.db.executescript(ENTRIES_SCHEMA)
        self.db.commit()

    def _check_schema(self) -> None:
        """Record the version, or migrate an older cache in place.

        Refusing to open an old cache and telling the user to wipe it was the
        wrong trade twice over: it threw away answers they had already paid for,
        and `semcache clear` had to open the same database to do it, so the
        suggested fix could not work either.
        """
        cur = self.db.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if row is None:
            self.db.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            return

        found = int(row[0])
        if found == SCHEMA_VERSION:
            return
        if found > SCHEMA_VERSION:
            raise RuntimeError(
                f"cache at {self.path} was written by a newer semcache "
                f"(schema v{found}, this build understands v{SCHEMA_VERSION}). "
                "Upgrade semcache, or point --home somewhere else."
            )
        self._migrate(found)

    def _migrate(self, found: int) -> None:
        """Upgrade an older cache, keeping every stored answer."""
        if found < 2:
            # v2 added entries.use_seq for tie-free LRU ordering. Backfill it
            # from last_used_at so the existing eviction order is preserved.
            columns = {r[1] for r in self.db.execute("PRAGMA table_info(entries)")}
            if "use_seq" not in columns:
                self.db.execute("ALTER TABLE entries ADD COLUMN use_seq INTEGER NOT NULL DEFAULT 0")
            self.db.execute(
                "UPDATE entries SET use_seq = ("
                "  SELECT COUNT(*) FROM entries older"
                "  WHERE older.last_used_at <= entries.last_used_at"
                ")"
            )
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_lru ON entries(namespace, use_seq)")

        self.db.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),)
        )
        self.db.commit()

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
            " hit_count, prompt_tokens, response_tokens, use_seq)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?,?,"
            " (SELECT IFNULL(MAX(use_seq), 0) + 1 FROM entries))",
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
        """Record a reuse: bumps both the display timestamp and the monotonic
        use counter. This is what makes eviction LRU rather than FIFO."""
        self.db.execute(
            "UPDATE entries SET last_used_at=?, hit_count=hit_count+1,"
            " use_seq=(SELECT IFNULL(MAX(use_seq), 0) + 1 FROM entries)"
            " WHERE id=?",
            (time.time(), entry_id),
        )
        self.db.commit()

    def lru_ids(self, over_by: int) -> list[int]:
        if over_by <= 0:
            return []
        cur = self.db.execute(
            "SELECT id FROM entries WHERE namespace=? ORDER BY use_seq ASC, id ASC LIMIT ?",
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
