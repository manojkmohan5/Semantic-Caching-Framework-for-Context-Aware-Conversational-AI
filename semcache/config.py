"""Paths, defaults, and the CLI > env > config file > default resolution order.

Also owns the two version constants that decide cache compatibility. Bump either
one and existing entries land in a fresh namespace instead of being misread.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

SCHEMA_VERSION = 2  # 2 added entries.use_seq for tie-free LRU ordering
#: Bump when prompt normalization changes. Old vectors were built from differently
#: normalized text, so they must not be compared against new ones.
NORMALIZE_VERSION = 1

ENV_PREFIX = "SEMCACHE_"


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_opt_float(value: str) -> float | None:
    value = value.strip().lower()
    return None if value in {"", "none", "off", "0"} else float(value)


# Only fields that are not plain strings need an entry here.
_COERCE = {
    "threshold": float,
    "fallback_threshold": float,
    "max_entries": int,
    "max_tokens": int,
    "ttl_seconds": _as_opt_float,
    "context_turns": int,
    "temperature": float,
    "top_k": int,
    "flush_every": int,
    "max_cache_bytes": int,
    "timeout_connect": float,
    "timeout_read": float,
    "max_retries": int,
    "stream": _as_bool,
    "log_prompts": _as_bool,
    "offline": _as_bool,
    "debug": _as_bool,
}


@dataclass
class Config:
    """Everything tunable. Defaults are chosen so a normal user never passes a flag."""

    home: Path = Path.home() / ".semcache"

    # --- matching ---
    #: Cosine score required to reuse an answer. High on purpose: a wrong answer
    #: served confidently is far worse than an extra API call.
    #:
    #: 0.95 is measured, not guessed. On data/queries.jsonl with
    #: bge-small-en-v1.5, 0.90 admitted 6 of 12 trap questions while 0.95 admits
    #: 1 and still keeps 23 of 24 real paraphrases. Re-run `semcache bench` after
    #: changing this.
    threshold: float = 0.95
    #: Lower bar used *only* when the provider is unreachable or rate-limited.
    fallback_threshold: float = 0.75
    top_k: int = 5
    scope: str = "global"  # "global" reuses across sessions; "session" filters to one
    context_turns: int = 2

    # --- capacity ---
    max_entries: int = 5000
    ttl_seconds: float | None = None
    max_cache_bytes: int = 256_000

    # --- model ---
    embedder: str = "auto"  # auto | local | api | hash
    provider: str | None = None
    model: str | None = None
    #: Endpoint for OpenAI-compatible hosts. Overrides the built-in preset.
    base_url: str | None = None
    #: A cache implies a stable answer for a stable question. Note this is only
    #: sent to models that still accept it -- current Anthropic models reject it.
    temperature: float = 0.0
    max_tokens: int = 16000
    #: Anthropic reasoning effort: low|medium|high|xhigh|max. None = server default.
    #: "low" measurably cuts miss latency on thinking-by-default models.
    effort: str | None = None
    stream: bool = True

    # --- io ---
    flush_every: int = 25  # index writes are debounced this many inserts
    timeout_connect: float = 5.0
    timeout_read: float = 120.0
    max_retries: int = 2

    # --- misc ---
    #: Isolates one project's cache from another. Unset (the default) means all
    #: projects share one cache, which is where the savings are largest -- opt
    #: into isolation only when answers should not cross project boundaries.
    project: str | None = None
    #: Where the ONNX model is cached. Separate from `home` so a container can
    #: bake the model into the image outside the mounted cache volume, which
    #: would otherwise mask it.
    model_cache_dir: str | None = None
    log_prompts: bool = True
    offline: bool = False
    debug: bool = False

    # ------------------------------------------------------------------ paths
    @property
    def db_path(self) -> Path:
        return self.home / "cache.db"

    @property
    def index_dir(self) -> Path:
        return self.home / "index"

    @property
    def metrics_path(self) -> Path:
        return self.home / "metrics.jsonl"

    @property
    def models_dir(self) -> Path:
        if self.model_cache_dir:
            return Path(self.model_cache_dir).expanduser()
        return self.home / "models"

    @property
    def env_path(self) -> Path:
        return self.home / ".env"

    @property
    def config_path(self) -> Path:
        return self.home / "config.json"

    def ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- resolution
    @classmethod
    def load(cls, **overrides) -> Config:
        """Defaults, then config.json, then SEMCACHE_* env vars, then CLI overrides."""
        # Explicit argument first: the documented order is CLI > env > file >
        # default, and reading the env var first inverted it for `home` alone.
        home = Path(overrides.get("home") or os.environ.get(f"{ENV_PREFIX}HOME") or cls.home)
        cfg = cls(home=Path(home).expanduser())

        cfg = _overlay(cfg, _from_file(cfg.config_path))
        cfg = _overlay(cfg, _from_env())
        cfg = _overlay(cfg, {k: v for k, v in overrides.items() if v is not None})

        if cfg.project:
            # Scope the whole cache -- db, index and metrics -- under the project,
            # then re-read that project's own config so a model saved there sticks.
            # `home` is deliberately excluded from these re-overlays: re-applying
            # it would undo the scoping we just did.
            # The embedding model is not project data. Pin it to the root home
            # before scoping, or every project re-downloads its own 130MB copy.
            if not cfg.model_cache_dir:
                cfg = replace(cfg, model_cache_dir=str(cfg.home / "models"))
            cfg = replace(cfg, home=cfg.home / "projects" / slug(cfg.project))
            cfg = _overlay(cfg, _drop_home(_from_file(cfg.config_path)))
            cfg = _overlay(cfg, _drop_home(_from_env()))
            cfg = _overlay(
                cfg,
                _drop_home({k: v for k, v in overrides.items() if v is not None}),
            )
        return cfg

    def as_dict(self) -> dict:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = str(value) if isinstance(value, Path) else value
        return out


def _drop_home(values: dict) -> dict:
    return {k: v for k, v in values.items() if k != "home"}


def slug(name: str) -> str:
    """Filesystem-safe project name."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.strip())
    return cleaned.strip("-").lower() or "default"


def _from_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt config must not stop the app -- defaults are always valid.
        return {}
    return data if isinstance(data, dict) else {}


def _from_env() -> dict:
    known = {f.name for f in fields(Config)} - {"home"}
    found = {}
    for name in known:
        raw = os.environ.get(f"{ENV_PREFIX}{name.upper()}")
        if raw is None:
            continue
        # An unparseable env var falls back to the default rather than crashing.
        with contextlib.suppress(ValueError):
            found[name] = _COERCE.get(name, str)(raw)
    return found


def _overlay(cfg: Config, values: dict) -> Config:
    known = {f.name for f in fields(Config)}
    clean = {k: v for k, v in values.items() if k in known}
    if "home" in clean:
        clean["home"] = Path(clean["home"]).expanduser()
    return replace(cfg, **clean) if clean else cfg


def make_namespace(embedder_id: str, dim: int) -> str:
    """Identity of a set of comparable vectors.

    Vectors from different embedders (or different dimensions, or a changed
    normalization scheme) are not comparable at all -- mixing them silently
    returns nonsense. Keying the index and every row by this makes switching
    embedders start a clean namespace instead of corrupting the old one.
    """
    raw = f"{embedder_id}|{dim}|norm{NORMALIZE_VERSION}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
