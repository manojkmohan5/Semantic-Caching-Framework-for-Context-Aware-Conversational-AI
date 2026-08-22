"""One runnable check per piece of non-trivial logic. No key, no network, no model.

Everything here uses HashEmbedder + StubProvider, so it runs anywhere CI runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from semcache.cache import SemanticCache
from semcache.config import Config, make_namespace
from semcache.embedders import HashEmbedder, l2_normalize
from semcache.metrics import Aggregate, Record, percentile
from semcache.providers import ChatResult, StubProvider, detect_provider, model_info
from semcache.store import Store, normalize, prompt_hash


def make_cache(tmp_path, **over) -> SemanticCache:
    cfg = Config.load(home=tmp_path, embedder="hash", offline=True, **over)
    return SemanticCache(cfg, HashEmbedder())


def add(cache, prompt, response="answer", session="s1"):
    vec = cache.embedder.encode_one(prompt)
    return cache.put(
        prompt=prompt,
        response=response,
        vector=vec,
        session_id=session,
        provider="stub",
        model="stub-1",
        prompt_tokens=4,
        response_tokens=9,
    )


def look(cache, prompt, session="s1"):
    return cache.lookup(prompt, cache.embedder.encode_one(prompt), session)


# ------------------------------------------------------------------ primitives


def test_normalization_folds_into_the_hash():
    assert normalize("  What   IS  a  Cache? ") == "what is a cache?"
    assert prompt_hash("a b") == prompt_hash("A    B")
    assert prompt_hash("a b") != prompt_hash("a c")


def test_vectors_are_unit_length_so_dot_product_is_cosine():
    v = HashEmbedder().encode(["hello world", "different text here"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    # identical text -> cosine 1
    assert float(v[0] @ HashEmbedder().encode_one("hello world")) == pytest.approx(1.0, abs=1e-5)


def test_zero_vector_does_not_divide_by_zero():
    assert not np.isnan(l2_normalize(np.zeros((1, 8)))).any()


def test_namespace_separates_incomparable_vectors():
    # Same embedder name, different dim -> must never share an index.
    assert make_namespace("e", 384) != make_namespace("e", 1536)
    assert make_namespace("a", 384) != make_namespace("b", 384)


def test_anthropic_models_are_not_sent_temperature():
    # The API removed the parameter and returns 400. An unknown Anthropic model
    # must be assumed to reject it too.
    assert model_info("anthropic", "claude-sonnet-5").supports_temperature is False
    assert model_info("anthropic", "claude-made-up").supports_temperature is False
    assert model_info("openai", "gpt-5").supports_temperature is True


def test_key_prefix_detection():
    assert detect_provider("sk-ant-abc") == "anthropic"
    assert detect_provider("sk-proj-abc") == "openai"
    assert detect_provider("AIzaabc") == "gemini"
    assert detect_provider("garbage") is None


# ----------------------------------------------------------------- the 3 tiers


def test_exact_hit_is_case_and_whitespace_insensitive(tmp_path):
    cache = make_cache(tmp_path)
    add(cache, "What is a semantic cache?", "the answer")
    hit, score = cache.lookup("what is   A SEMANTIC cache?", None, "s1")
    assert hit is not None and hit.kind == "exact"
    assert hit.response == "the answer"
    assert hit.similarity is None  # nothing was compared
    cache.close()


def test_semantic_hit_above_threshold_and_miss_below(tmp_path):
    cache = make_cache(tmp_path, threshold=0.40)
    add(cache, "checkout api returns 504 on large carts", "the 504 answer")

    hit, score = look(cache, "checkout api 504 large carts")
    assert hit is not None and hit.kind == "semantic"
    assert hit.response == "the 504 answer"
    assert score >= 0.40

    hit, best = look(cache, "how do I bake sourdough bread")
    assert hit is None
    assert best < 0.40  # and reported, so the threshold can be tuned
    cache.close()


def test_threshold_boundary_is_inclusive(tmp_path):
    cache = make_cache(tmp_path, threshold=0.40)
    add(cache, "alpha beta gamma", "a")
    _, score = look(cache, "alpha beta delta")
    # Re-open with the threshold set exactly at the observed score: must hit.
    cache.close()
    cache = make_cache(tmp_path, threshold=score)
    hit, _ = look(cache, "alpha beta delta")
    assert hit is not None, f"score {score} should hit at threshold {score}"
    cache.close()


def test_miss_reports_best_subthreshold_score(tmp_path):
    cache = make_cache(tmp_path, threshold=0.99)
    add(cache, "one two three four", "a")
    hit, best = look(cache, "one two three five")
    assert hit is None
    assert 0.0 < best < 0.99  # the tuning signal
    cache.close()


# -------------------------------------------------------------------- rejection


def test_empty_and_oversized_responses_are_never_cached(tmp_path):
    cache = make_cache(tmp_path, max_cache_bytes=100)
    assert add(cache, "q1", "") is None
    assert add(cache, "q2", "   \n ") is None
    assert add(cache, "q3", "x" * 200) is None
    assert cache.store.count() == 0
    assert add(cache, "q4", "fine") is not None
    cache.close()


def test_truncated_results_are_flagged():
    assert ChatResult("t", 1, 1, "max_tokens", "m").truncated is True
    assert ChatResult("t", 1, 1, "length", "m").truncated is True
    assert ChatResult("t", 1, 1, "end_turn", "m").truncated is False


# -------------------------------------------------------------------- eviction


def test_lru_eviction_removes_from_both_stores(tmp_path):
    cache = make_cache(tmp_path, max_entries=3)
    for i in range(3):
        add(cache, f"prompt number {i}", f"answer {i}")
    assert cache.store.count() == 3
    assert len(cache.index) == 3

    # Reuse entry 0 so it is the most-recently-used, then overflow.
    assert look(cache, "prompt number 0")[0] is not None
    add(cache, "prompt number 3", "answer 3")

    assert cache.store.count() == 3, "capacity must hold"
    assert len(cache.index) == 3, "index must shrink with the store, not drift"
    assert cache.evicted == 1

    # LRU, not FIFO: the touched entry survived, the untouched oldest went.
    assert look(cache, "prompt number 0")[0] is not None
    assert cache.store.by_hash("prompt number 1") is None
    cache.close()


def test_ttl_expires_old_entries(tmp_path):
    cache = make_cache(tmp_path)
    add(cache, "will expire", "a")
    cache.close()
    # Reopen with a TTL of 0: everything is already older than the cutoff.
    cache = make_cache(tmp_path, ttl_seconds=0.000001)
    assert cache.store.count() == 0
    cache.close()


# ----------------------------------------------------------------- persistence


def test_cache_survives_a_restart(tmp_path):
    cache = make_cache(tmp_path)
    add(cache, "persist me", "kept")
    cache.close()

    reopened = make_cache(tmp_path)
    hit, _ = reopened.lookup("persist me", None, "s1")
    assert hit is not None and hit.response == "kept"
    assert len(reopened.index) == 1
    reopened.close()


def test_deleting_the_index_rebuilds_it_from_sqlite(tmp_path):
    cache = make_cache(tmp_path, threshold=0.40)
    add(cache, "rebuild me from sqlite", "kept")
    ns = cache.namespace
    cache.close()

    index_file = tmp_path / "index" / f"{ns}.faiss"
    assert index_file.exists()
    index_file.unlink()  # simulate a lost or corrupt index

    reopened = make_cache(tmp_path, threshold=0.40)
    assert len(reopened.index) == 1, "must rebuild from the vectors in SQLite"
    hit, _ = look(reopened, "rebuild me from sqlite")
    assert hit is not None and hit.response == "kept"
    reopened.close()


def test_switching_embedder_isolates_instead_of_corrupting(tmp_path):
    """A different embedder means incomparable vectors. It must open a clean
    namespace rather than mixing dimensions into one index."""
    cache = make_cache(tmp_path)
    add(cache, "shared question", "from hash embedder")
    first_ns = cache.namespace
    cache.close()

    class OtherEmbedder(HashEmbedder):
        id = "hash:other"
        dim = 64

        def encode(self, texts):
            return l2_normalize(np.ones((len(texts), self.dim), dtype="float32"))

    cfg = Config.load(home=tmp_path, embedder="hash", offline=True)
    other = SemanticCache(cfg, OtherEmbedder())
    assert other.namespace != first_ns
    assert other.store.count() == 0, "must not see the other embedder's entries"
    other.close()


# --------------------------------------------------------------------- scoping


def test_session_scope_filters_and_global_shares(tmp_path):
    shared = make_cache(tmp_path, scope="global")
    add(shared, "cross session question", "shared answer", session="alice")
    hit, _ = shared.lookup("cross session question", None, "bob")
    assert hit is not None, "global scope is where the savings come from"
    shared.close()

    scoped = make_cache(tmp_path, scope="session")
    assert scoped.lookup("cross session question", None, "bob")[0] is None
    assert scoped.lookup("cross session question", None, "alice")[0] is not None
    scoped.close()


# -------------------------------------------------------------------- degraded


def test_degraded_lookup_uses_the_lower_bar_only(tmp_path):
    cache = make_cache(tmp_path, threshold=0.99, fallback_threshold=0.30)
    add(cache, "gateway timeout at checkout", "the answer")

    # Normal lookup misses at 0.99...
    hit, best = look(cache, "checkout gateway timeout problem")
    assert hit is None
    # ...but the provider-down path still answers, marked degraded.
    vec = cache.embedder.encode_one("checkout gateway timeout problem")
    degraded = cache.fallback_lookup(vec, "s1")
    assert degraded is not None
    assert degraded.kind == "degraded"
    assert degraded.similarity >= 0.30
    cache.close()


def test_degraded_still_refuses_when_nothing_is_close(tmp_path):
    cache = make_cache(tmp_path, fallback_threshold=0.75)
    add(cache, "gateway timeout at checkout", "the answer")
    vec = cache.embedder.encode_one("entirely unrelated sourdough baking")
    assert cache.fallback_lookup(vec, "s1") is None
    cache.close()


# --------------------------------------------------------------------- metrics


def test_percentile_matches_hand_computed_values():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(data, 50) == pytest.approx(5.5)
    assert percentile(data, 0) == 1
    assert percentile(data, 100) == 10
    assert percentile([], 50) == 0.0
    assert percentile([42], 95) == 42


def test_aggregate_counts_and_savings(tmp_path):
    records = [
        Record(
            outcome="miss",
            total_ms=2000,
            llm_ms=1990,
            cost_usd=0.01,
            prompt_tokens=10,
            response_tokens=100,
            provider="stub",
            model="m",
        ),
        Record(
            outcome="semantic",
            total_ms=5,
            cost_saved_usd=0.01,
            prompt_tokens=10,
            response_tokens=100,
            provider="stub",
            model="m",
        ),
        Record(
            outcome="exact",
            total_ms=1,
            cost_saved_usd=0.01,
            prompt_tokens=10,
            response_tokens=100,
            provider="stub",
            model="m",
        ),
        Record(outcome="error", total_ms=50, provider="stub", model="m", error="rate_limit"),
    ]
    agg = Aggregate.of(records)
    assert agg.total == 4
    assert agg.hits == 2
    assert agg.misses == 1
    assert agg.errors == 1
    assert agg.calls_avoided == 2
    assert agg.cost_spent == pytest.approx(0.01)
    assert agg.cost_saved == pytest.approx(0.02)
    assert agg.tokens_avoided_out == 200
    # 2 of 3 answerable requests came from cache
    assert agg.hit_rate == pytest.approx(2 / 3)


def test_aggregate_of_nothing_does_not_divide_by_zero():
    agg = Aggregate.of([])
    assert agg.total == 0
    assert agg.hit_rate == 0.0
    assert agg.speedup == 0.0


# ----------------------------------------------------------------------- stub


def test_stub_provider_streams_and_reports_usage():
    p = StubProvider()
    chunks = list(p.stream_chat("hello there", []))
    assert chunks, "must yield something to stream"
    assert p.result is not None
    assert p.result.output_tokens > 0
    assert not p.result.truncated


def test_schema_version_mismatch_is_a_clear_error(tmp_path):
    store = Store(tmp_path / "c.db", "ns")
    store.db.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    store.db.commit()
    store.close()
    with pytest.raises(RuntimeError, match="schema"):
        Store(tmp_path / "c.db", "ns")


# ------------------------------------------------------- concurrency regression


def test_index_temp_file_is_private_to_this_process(tmp_path):
    """Two processes sharing one ".tmp" path clobbered each other's write, which
    failed the request on Windows. The pid keeps them apart."""
    from semcache.index import VectorIndex

    index = VectorIndex(4, tmp_path / "n.faiss")
    index.add([1], np.eye(4, dtype="float32")[0])
    index.save(force=True)
    assert str(os.getpid()) in str(index.path.with_suffix(f"{index.path.suffix}.{os.getpid()}.tmp"))
    # The real temp file must not survive a successful save.
    assert not list(tmp_path.glob("*.tmp"))
    assert index.path.exists()


def test_failed_index_flush_is_not_fatal(tmp_path, monkeypatch):
    """SQLite holds the vectors, so a lost flush is not lost data. A user's
    question must never fail because a cache file could not be written."""
    from semcache import index as index_mod

    index = index_mod.VectorIndex(4, tmp_path / "n.faiss")
    index.add([1], np.eye(4, dtype="float32")[0])

    def boom(*_a, **_k):
        raise PermissionError("held by another process")

    monkeypatch.setattr(index_mod.os, "replace", boom)
    assert index.save(force=True) is False  # reported, not raised
    assert len(index) == 1  # in-memory state is untouched


def test_index_rebuilds_after_a_corrupt_file(tmp_path):
    from semcache.index import VectorIndex

    path = tmp_path / "n.faiss"
    index = VectorIndex(4, path)
    index.add([1, 2], np.eye(4, dtype="float32")[:2])
    index.save(force=True)
    path.write_bytes(b"not a faiss index")

    reopened = VectorIndex(4, path)
    assert len(reopened) == 0, "a corrupt file must be discarded, not misread"


# ------------------------------------------------------ non-interactive startup


def test_explicit_provider_is_not_re_confirmed(tmp_path, monkeypatch):
    """An explicit --provider is already the answer. Re-asking made scripting
    impossible and re-prompted every returning user."""
    from semcache import cli

    def no_input(*_a, **_k):
        raise AssertionError("must not prompt when the provider is explicit")

    monkeypatch.setattr("builtins.input", no_input)
    cfg = Config.load(home=tmp_path, provider="anthropic")
    key, provider = cli.resolve_key(cfg, "sk-ant-whatever")
    assert provider == "anthropic"
    assert key == "sk-ant-whatever"


def test_no_tty_never_prompts(tmp_path, monkeypatch):
    from semcache import cli

    monkeypatch.setattr("builtins.input", lambda *_a, **_k: pytest.fail("prompted"))
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    cfg = Config.load(home=tmp_path)
    _key, provider = cli.resolve_key(cfg, "sk-ant-abc")
    assert provider == "anthropic"  # taken from the key shape, unattended
    # and the model picker falls back to the documented default
    assert cli.choose_model(cfg, "anthropic") == "claude-sonnet-5"


def test_offline_needs_no_key_at_all(tmp_path):
    from semcache import cli

    key, provider = cli.resolve_key(Config.load(home=tmp_path, offline=True), None)
    assert (key, provider) == ("", "stub")
