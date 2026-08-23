"""FAISS vector index with id-based removal, atomic saves and a numpy fallback."""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

import numpy as np

try:
    import faiss

    _HAVE_FAISS = True
    # Below ~50k vectors, spawning OpenMP threads costs more than the search.
    faiss.omp_set_num_threads(1)
except Exception:  # pragma: no cover - exercised only where the wheel is missing
    faiss = None
    _HAVE_FAISS = False


class VectorIndex:
    """Exact cosine search over L2-normalized vectors, keyed by int64 id.

    Wraps IndexIDMap2 rather than a bare IndexFlatIP because a bare flat index
    has no remove_ids -- and without removal, LRU eviction cannot shrink the
    index, so it would grow forever while SQLite shrank beside it.

    ponytail: exact brute-force scan; remove_ids compacts in O(n). Correct and
    fast to ~100k entries. Switch to IndexIVFFlat past that.
    """

    def __init__(self, dim: int, path: Path | None = None):
        self.dim = dim
        self.path = Path(path) if path else None
        self._pending = 0
        self._index = None
        self._ids: list[int] = []  # fallback only
        self._vecs: np.ndarray | None = None  # fallback only
        self._load_or_create()

    # ------------------------------------------------------------------ set up
    def _new_index(self):
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    def _load_or_create(self) -> None:
        if not _HAVE_FAISS:
            self._vecs = np.zeros((0, self.dim), dtype="float32")
            return
        if self.path and self.path.exists():
            try:
                # Deliberately NOT IO_FLAG_MMAP: mmap holds an open handle on
                # the file, and on Windows that makes another process's
                # os.replace() fail with a sharing violation. Measured startup
                # with a full read is 2-25ms at our scale, so mmap buys nothing
                # worth that hazard.
                loaded = faiss.read_index(str(self.path))
                if loaded.d == self.dim:
                    self._index = loaded
                    return
                # A dimension mismatch means this file belongs to another
                # embedder. Namespacing should prevent it; if it happens anyway,
                # discard rather than misread.
            except Exception:
                pass  # corrupt or truncated -- caller rebuilds from SQLite
        self._index = self._new_index()

    @property
    def backend(self) -> str:
        return "faiss" if _HAVE_FAISS else "numpy"

    def __len__(self) -> int:
        if _HAVE_FAISS:
            return int(self._index.ntotal)
        return len(self._ids)

    # ------------------------------------------------------------------ writes
    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if _HAVE_FAISS:
            self._index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))
        else:
            self._ids.extend(int(i) for i in ids)
            self._vecs = np.vstack([self._vecs, vectors])
        self._pending += len(ids)

    def remove(self, ids: list[int]) -> int:
        if not ids:
            return 0
        if _HAVE_FAISS:
            removed = int(self._index.remove_ids(np.asarray(ids, dtype="int64")))
        else:
            drop = set(int(i) for i in ids)
            keep = [i for i, entry in enumerate(self._ids) if entry not in drop]
            removed = len(self._ids) - len(keep)
            self._vecs = self._vecs[keep] if keep else np.zeros((0, self.dim), dtype="float32")
            self._ids = [self._ids[i] for i in keep]
        self._pending += removed
        return removed

    def rebuild(self, ids: list[int], vectors: np.ndarray) -> None:
        """Discard everything and rebuild from SQLite, the source of truth."""
        if _HAVE_FAISS:
            self._index = self._new_index()
        else:
            self._ids, self._vecs = [], np.zeros((0, self.dim), dtype="float32")
        if len(ids):
            self.add(ids, vectors)
        self.save(force=True)

    # ------------------------------------------------------------------ search
    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return (id, cosine) best-first. Vectors are unit length, so the inner
        product IS the cosine -- there is no separate similarity function."""
        if len(self) == 0:
            return []
        query = np.ascontiguousarray(vector, dtype="float32").reshape(1, -1)
        k = max(1, min(k, len(self)))
        if _HAVE_FAISS:
            scores, ids = self._index.search(query, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        sims = (self._vecs @ query[0]).astype("float32")
        order = np.argsort(-sims)[:k]
        return [(self._ids[i], float(sims[i])) for i in order]

    # ------------------------------------------------------------------- flush
    def save(self, force: bool = False) -> bool:
        """Debounced, atomic write. Skipping fsync on every turn saves ~20-40ms
        per question; SQLite already holds the vectors, so a crash costs a
        rebuild rather than data."""
        if self.path is None or not _HAVE_FAISS:
            return False
        if not force and self._pending < 1:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The pid makes the temp file private to this process. Sharing one
        # ".tmp" path let two processes write the same file at once, which
        # corrupted it and raised PermissionError on Windows.
        tmp = self.path.with_suffix(f"{self.path.suffix}.{os.getpid()}.tmp")
        try:
            faiss.write_index(self._index, str(tmp))
            for attempt in range(4):
                try:
                    # Atomic: no half-written index is ever observable.
                    os.replace(tmp, self.path)
                    self._pending = 0
                    return True
                except PermissionError:
                    # Another process holds the target open. Back off briefly;
                    # if it never clears, losing this flush is harmless.
                    if attempt == 3:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        except (OSError, RuntimeError):
            # A failed index flush is NOT data loss: SQLite holds the vectors and
            # the index is rebuilt on the next open. Never fail a user's question
            # over a cache-file write.
            return False
        finally:
            if tmp.exists():
                with contextlib.suppress(OSError):
                    tmp.unlink()
        return False

    @property
    def pending(self) -> int:
        return self._pending
