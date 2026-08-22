"""Replay a query set cold and warm, then report and optionally assert targets.

Runs against a throwaway cache directory so it never pollutes the real one.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from .cache import SemanticCache
from .chat import ChatSession
from .embedders import build_embedder
from .metrics import Aggregate, Metrics, percentile, render_detail
from .providers import build_provider


def default_queries() -> Path:
    """The bundled query set, found whether running from a checkout or an
    installed wheel. Walking up from __file__ lands in site-packages once
    installed, so the file has to travel with the package."""
    packaged = Path(__file__).resolve().parent / "data" / "queries.jsonl"
    if packaged.exists():
        return packaged
    return Path(__file__).resolve().parent.parent / "data" / "queries.jsonl"


#: Asserted with --assert-targets. Measured on the dev machine, not aspirational:
#: a semantic hit is dominated by the cost of embedding the query, which differs
#: by an order of magnitude between backends. bge-small takes ~25ms for one short
#: query, so a single <10ms figure for every backend would be fiction.
TARGETS = {
    "exact_hit_ms": 1.0,
    "warm_start_ms": 400.0,
}
HIT_MS_BY_EMBEDDER = {"hash": 5.0, "local": 60.0, "api": 500.0}

#: The hash embedder is lexical, not semantic: real paraphrases score ~0.45, so
#: the production 0.90 threshold would never hit. Bench lowers it for that
#: backend only, and says so in the output.
HASH_THRESHOLD = 0.55


def load_queries(path: Path) -> list[dict]:
    groups = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                groups.append(json.loads(line))
    return groups


def _pass(session: ChatSession, groups: list[dict]) -> tuple[list, int, int]:
    """Ask every prompt once. Returns (records, gated_false_hits, hard_trap_hits)."""
    records = []
    false_hits = 0
    hard_hits = 0
    for group in groups:
        prompts = [group["canonical"], *group.get("paraphrases", [])]
        for prompt in prompts:
            turn = session.ask(prompt)
            records.append(turn.record)
        # A trap is a similar-sounding question with a genuinely different
        # answer. Reusing a cached answer for one is the failure this catches.
        for trap in group.get("traps", []):
            turn = session.ask(trap)
            records.append(turn.record)
            # Only a SEMANTIC hit is a failure here. An exact hit just means the
            # trap was asked before and matched itself, which is correct reuse.
            if turn.outcome in ("semantic", "degraded"):
                false_hits += 1
        # hard_traps are documented known-hard cases (same words, reversed
        # relation). Reported so regressions stay visible, but not gated --
        # embeddings discard word order, so no threshold separates them.
        for trap in group.get("hard_traps", []):
            turn = session.ask(trap)
            records.append(turn.record)
            if turn.outcome in ("semantic", "degraded"):
                hard_hits += 1
    return records, false_hits, hard_hits


def run_bench(cfg, args) -> int:
    queries_path = Path(args.queries) if args.queries else default_queries()
    if not queries_path.exists():
        print(f"  no query set at {queries_path}")
        return 1
    groups = load_queries(queries_path)

    embedder_name = cfg.embedder if cfg.embedder != "auto" else "hash"
    threshold = HASH_THRESHOLD if embedder_name == "hash" else cfg.threshold

    with tempfile.TemporaryDirectory(prefix="semcache-bench-") as tmp:
        bench_cfg = replace(
            cfg,
            home=Path(tmp),
            embedder=embedder_name,
            threshold=threshold,
            offline=True if cfg.offline else cfg.offline,
        )
        provider = build_provider(
            "stub" if bench_cfg.offline else (cfg.provider or "stub"),
            "",
            cfg.model or "stub-1",
            bench_cfg,
        )
        embedder = build_embedder(bench_cfg, provider)

        prompts_total = sum(
            1
            + len(g.get("paraphrases", []))
            + len(g.get("traps", []))
            + len(g.get("hard_traps", []))
            for g in groups
        )
        print(
            f"\nreplaying {queries_path.name} · {len(groups)} groups · "
            f"{prompts_total} prompts · {provider.name} · {embedder.id}"
        )
        if embedder_name == "hash":
            print(
                f"  note: hash embedder is lexical only, so the threshold is "
                f"{threshold} rather than {cfg.threshold}"
            )

        # Warm start is measured on an already-populated cache, below.
        cache = SemanticCache(bench_cfg, embedder)
        metrics = Metrics(bench_cfg.metrics_path)
        session = ChatSession(bench_cfg, provider, embedder, cache, metrics)
        embedder.warm()

        cold_records, cold_traps, cold_hard = _pass(session, groups)
        cold = Aggregate.of(cold_records)

        warm_records, warm_traps, warm_hard = _pass(session, groups)
        warm = Aggregate.of(warm_records)

        namespace = cache.namespace
        cache.close()

        # Reopen to time a warm start against a populated cache on disk.
        import time as _time

        t0 = _time.perf_counter()
        reopened = SemanticCache(bench_cfg, embedder)
        warm_start_ms = (_time.perf_counter() - t0) * 1000
        entries = reopened.store.count()
        reopened.close()

    exact_ms = percentile([r.total_ms for r in warm_records if r.outcome == "exact"], 50)
    hit_ms = percentile(
        [r.total_ms for r in warm_records if r.outcome in ("exact", "semantic")], 50
    )

    _print_table(cold, warm)
    print(render_detail(warm, "warm pass detail"))

    hit_target = HIT_MS_BY_EMBEDDER.get(embedder_name, 60.0)
    checks = [
        ("exact hit typical", exact_ms, TARGETS["exact_hit_ms"], "<="),
        (f"cache hit typical ({embedder_name})", hit_ms, hit_target, "<="),
        ("warm start", warm_start_ms, TARGETS["warm_start_ms"], "<="),
    ]
    # Trap safety is a property of a *semantic* embedder. The hash backend is
    # bag-of-words and the traps differ by a single word, so it cannot possibly
    # separate them -- gating on it would assert the impossible. Its job is to
    # exercise cache mechanics offline, not semantic quality.
    if embedder_name != "hash":
        checks.insert(0, ("no false hits on trap questions", cold_traps + warm_traps, 0, "=="))
    print("targets")
    failures = 0
    for label, actual, target, op in checks:
        ok = actual == target if op == "==" else actual <= target
        unit = "" if op == "==" else " ms"
        print(
            f"  {'ok  ' if ok else 'FAIL'} {label:<32} "
            f"{actual:.2f}{unit} (target {op} {target}{unit})"
        )
        failures += 0 if ok else 1

    if warm.hits == 0:
        print("  FAIL warm pass produced no cache hits at all")
        failures += 1

    if embedder_name == "hash":
        print(
            f"  note {cold_traps + warm_traps} trap questions reused an answer -- "
            "expected, the hash embedder is lexical and cannot separate them"
        )

    hard_total = sum(len(g.get("hard_traps", [])) for g in groups) * 2
    if hard_total:
        print(
            f"  note {cold_hard + warm_hard}/{hard_total} known-hard traps reused an "
            "answer (same words, reversed relation -- embeddings are order-blind)"
        )

    report = {
        "queries": str(queries_path),
        "groups": len(groups),
        "embedder": embedder.id,
        "threshold": threshold,
        "entries_after": entries,
        "namespace": namespace,
        "cold": _summary(cold),
        "warm": _summary(warm),
        "warm_start_ms": round(warm_start_ms, 2),
        "false_hits": cold_traps + warm_traps,
        "hard_trap_hits": cold_hard + warm_hard,
        "failures": failures,
    }
    try:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.report}")
    except OSError:
        pass

    if getattr(args, "assert_targets", False) and failures:
        print(f"\n{failures} target(s) failed")
        return 1
    return 0


def _summary(agg: Aggregate) -> dict:
    return {
        "requests": agg.total,
        "hits": agg.hits,
        "exact": agg.exact,
        "semantic": agg.semantic,
        "misses": agg.misses,
        "hit_rate": round(agg.hit_rate, 4),
        "median_ms": round(percentile(agg.hit_latencies + agg.miss_latencies, 50), 2),
        "hit_median_ms": round(percentile(agg.hit_latencies, 50), 2),
        "miss_median_ms": round(percentile(agg.miss_latencies, 50), 2),
        "p95_ms": round(percentile(agg.hit_latencies + agg.miss_latencies, 95), 2),
        "model_calls": agg.misses,
    }


def _print_table(cold: Aggregate, warm: Aggregate) -> None:
    def row(label: str, a, b, suffix: str = "") -> str:
        return f"  {label:<22} {a:>9}{suffix} {b:>9}{suffix}"

    all_cold = cold.hit_latencies + cold.miss_latencies
    all_warm = warm.hit_latencies + warm.miss_latencies
    mean_cold = sum(all_cold) / len(all_cold) if all_cold else 0
    mean_warm = sum(all_warm) / len(all_warm) if all_warm else 0
    change = f"{(mean_warm - mean_cold) / mean_cold * 100:.1f}%" if mean_cold else "-"

    print("")
    print(f"  {'':<22} {'cold':>9} {'warm':>9}")
    print(row("requests", cold.total, warm.total))
    print(row("answered from cache", cold.hits, warm.hits))
    print(row("hit rate", f"{cold.hit_rate * 100:.1f}%", f"{warm.hit_rate * 100:.1f}%"))
    print(row("model calls", cold.misses, warm.misses))
    print(
        row(
            "typical response",
            f"{percentile(all_cold, 50):.1f}",
            f"{percentile(all_warm, 50):.1f}",
            " ms",
        )
    )
    print(
        row(
            "slowest 1 in 20",
            f"{percentile(all_cold, 95):.1f}",
            f"{percentile(all_warm, 95):.1f}",
            " ms",
        )
    )
    print(row("average", f"{mean_cold:.1f}", f"{mean_warm:.1f}", " ms"))
    print(f"  {'average change':<22} {'':>9} {change:>9}")
