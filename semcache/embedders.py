"""Text to vectors, three ways, behind one interface.

All three L2-normalize their output, which is what lets FAISS inner product BE
cosine similarity -- no separate cosine function anywhere in the codebase.
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np

from .config import make_namespace
from .providers import EMBEDDING_MODELS, ProviderError

_WORD = re.compile(r"[a-z0-9']+")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so a dot product equals cosine similarity."""
    matrix = np.asarray(matrix, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction; leave it at zero rather than dividing by 0.
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


class Embedder:
    """id + dim + encode(). id and dim together decide cache namespace identity."""

    id = "base"
    dim = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def namespace(self) -> str:
        return make_namespace(self.id, self.dim)

    def warm(self) -> None:
        """Optional: pay any one-time load cost now rather than on first use."""


class HashEmbedder(Embedder):
    """Deterministic, offline, dependency-free. Hashes word unigrams and bigrams
    into a fixed-width vector.

    This is what lets the test suite and CI exercise the whole pipeline with no
    API key, no network and no 130MB model download. It captures word overlap,
    not meaning -- two paraphrases with no shared words score ~0 -- so it is
    correct for testing cache *mechanics* and wrong for real use.

    ponytail: bag-of-words hashing, no semantics. Real runs use local/api.
    """

    id = "hash:v1"
    dim = 256

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            words = _WORD.findall(text.lower())
            grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
            for gram in grams:
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest()
                bucket = int.from_bytes(digest, "big") % self.dim
                out[row, bucket] += 1.0
        return l2_normalize(out)


class LocalEmbedder(Embedder):
    """fastembed / ONNX. No torch, no network after the first run."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.id = f"local:{model_name}"
        self.dim = 384  # bge-small-en-v1.5; corrected from the model on first encode
        self._model = None

    def _load(self):
        if self._model is None:
            # Windows without Developer Mode cannot create the symlinks the HF
            # cache prefers; it falls back correctly but logs a scary
            # WinError 1314 first. Silence the noise before importing.
            # Windows without Developer Mode cannot create symlinks, and the
            # hub logs an ERROR-level WinError 1314 before falling back. Telling
            # it to copy instead avoids the failed attempt altogether.
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ProviderError(
                    'local embedder needs fastembed. Run: pip install "semcache[local]"'
                    " -- or use --embedder api",
                    "error",
                ) from exc
            kwargs = {"model_name": self.model_name}
            if self.cache_dir is not None:
                kwargs["cache_dir"] = str(self.cache_dir)
            self._model = TextEmbedding(**kwargs)
        return self._model

    def warm(self) -> None:
        self._load()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._load().embed(list(texts))), dtype="float32")
        if vectors.shape[1] != self.dim:
            self.dim = int(vectors.shape[1])
        return l2_normalize(vectors)


class ApiEmbedder(Embedder):
    """The provider's own embedding endpoint. Costs a network round-trip per lookup."""

    def __init__(self, provider):
        name = getattr(provider, "name", "unknown")
        if name not in EMBEDDING_MODELS:
            raise ProviderError(
                f"{name} has no embedding API. Anthropic does not offer embeddings -- "
                "use --embedder local (or --embedder hash for tests).",
                "error",
            )
        self.provider = provider
        model, dim = EMBEDDING_MODELS[name]
        self.id = f"api:{model}"
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(self.provider.embed(list(texts)), dtype="float32")
        if vectors.ndim == 2 and vectors.shape[1] != self.dim:
            self.dim = int(vectors.shape[1])
        return l2_normalize(vectors)


def build_embedder(cfg, provider=None) -> Embedder:
    """Resolve cfg.embedder. "auto" prefers local, then api, then hash."""
    choice = (cfg.embedder or "auto").lower()

    if choice == "hash":
        return HashEmbedder()
    if choice == "local":
        return LocalEmbedder(cache_dir=cfg.models_dir)
    if choice == "api":
        return ApiEmbedder(provider)

    if choice != "auto":
        raise ProviderError(
            f"unknown embedder {cfg.embedder!r} (expected auto|local|api|hash)", "error"
        )

    # auto: local is preferred because a cache hit then touches the network zero
    # times, which is the entire point of the cache.
    try:
        import fastembed  # noqa: F401

        return LocalEmbedder(cache_dir=cfg.models_dir)
    except ImportError:
        pass
    if provider is not None and getattr(provider, "name", "") in EMBEDDING_MODELS:
        return ApiEmbedder(provider)
    return HashEmbedder()
