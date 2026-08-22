"""One record per request, one pure aggregator, two renderers.

The split matters: /stats must read like plain English while `bench` needs real
percentiles. Both read the same Aggregate, so the numbers can never disagree --
only the wording does.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Record:
    """Everything measured about one request."""

    ts: float = field(default_factory=time.time)
    session_id: str = ""
    outcome: str = "miss"  # exact | semantic | degraded | miss | error
    provider: str = ""
    model: str = ""
    embedder: str = ""
    similarity: float | None = None
    #: Best score that did NOT clear the threshold. On a miss this is the single
    #: most useful tuning signal -- it says whether the threshold caused the miss.
    best_below: float | None = None
    embed_ms: float = 0.0
    search_ms: float = 0.0
    llm_ms: float = 0.0
    first_token_ms: float | None = None
    total_ms: float = 0.0
    prompt_tokens: int = 0
    response_tokens: int = 0
    cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    matched_id: int | None = None
    error: str | None = None
    prompt: str | None = None  # omitted when log_prompts is off

    @property
    def is_hit(self) -> bool:
        return self.outcome in {"exact", "semantic"}


HIT_KINDS = {"exact", "semantic"}


def percentile(values, pct: float) -> float:
    """Linear-interpolated percentile. Small enough not to pull in a dependency."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return 0.0
    if len(data) == 1:
        return float(data[0])
    pos = (len(data) - 1) * (pct / 100.0)
    low = int(pos)
    high = min(low + 1, len(data) - 1)
    frac = pos - low
    return float(data[low] * (1 - frac) + data[high] * frac)


@dataclass
class Aggregate:
    total: int = 0
    exact: int = 0
    semantic: int = 0
    degraded: int = 0
    misses: int = 0
    errors: int = 0
    hit_latencies: list = field(default_factory=list)
    miss_latencies: list = field(default_factory=list)
    cost_spent: float = 0.0
    cost_saved: float = 0.0
    tokens_sent_in: int = 0
    tokens_sent_out: int = 0
    tokens_avoided_in: int = 0
    tokens_avoided_out: int = 0
    provider_ok: int = 0
    provider_fail: int = 0
    error_kinds: dict = field(default_factory=dict)
    provider: str = ""
    model: str = ""

    @classmethod
    def of(cls, records) -> Aggregate:
        agg = cls()
        for r in records:
            agg.total += 1
            agg.provider = r.provider or agg.provider
            agg.model = r.model or agg.model
            if r.outcome == "exact":
                agg.exact += 1
            elif r.outcome == "semantic":
                agg.semantic += 1
            elif r.outcome == "degraded":
                agg.degraded += 1
            elif r.outcome == "error":
                agg.errors += 1
            else:
                agg.misses += 1

            if r.outcome in HIT_KINDS:
                agg.hit_latencies.append(r.total_ms)
                agg.cost_saved += r.cost_saved_usd
                agg.tokens_avoided_in += r.prompt_tokens
                agg.tokens_avoided_out += r.response_tokens
            elif r.outcome == "miss":
                agg.miss_latencies.append(r.total_ms)
                agg.cost_spent += r.cost_usd
                agg.tokens_sent_in += r.prompt_tokens
                agg.tokens_sent_out += r.response_tokens
                agg.provider_ok += 1
            elif r.outcome == "error":
                agg.provider_fail += 1
                agg.error_kinds[r.error or "error"] = agg.error_kinds.get(r.error or "error", 0) + 1
            elif r.outcome == "degraded":
                # A degraded answer means the provider was tried and failed.
                agg.provider_fail += 1
                agg.error_kinds[r.error or "unavailable"] = (
                    agg.error_kinds.get(r.error or "unavailable", 0) + 1
                )
        return agg

    # ------------------------------------------------------------- derivations
    @property
    def hits(self) -> int:
        return self.exact + self.semantic

    @property
    def answerable(self) -> int:
        """Requests that could have been answered -- errors could not."""
        return self.hits + self.misses + self.degraded

    @property
    def hit_rate(self) -> float:
        base = self.hits + self.misses
        return self.hits / base if base else 0.0

    @property
    def calls_avoided(self) -> int:
        return self.hits

    @property
    def hit_typical(self) -> float:
        return percentile(self.hit_latencies, 50)

    @property
    def miss_typical(self) -> float:
        return percentile(self.miss_latencies, 50)

    @property
    def speedup(self) -> float:
        """How many times faster a cached answer came back."""
        if not self.hit_latencies or not self.miss_latencies or self.hit_typical <= 0:
            return 0.0
        return self.miss_typical / self.hit_typical

    @property
    def cost_total(self) -> float:
        return self.cost_spent + self.cost_saved

    @property
    def saved_pct(self) -> float:
        return (self.cost_saved / self.cost_total * 100) if self.cost_total else 0.0


class Metrics:
    """Appends one JSON object per request. Append-only so a crash cannot lose
    earlier records, and re-sliceable later with `semcache stats`."""

    def __init__(self, path: Path, log_prompts: bool = True):
        self.path = Path(path)
        self.log_prompts = log_prompts
        self.session: list[Record] = []

    def record(self, rec: Record) -> Record:
        if not self.log_prompts:
            rec.prompt = None
        self.session.append(rec)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        except OSError:
            pass  # never let a metrics write break the user's actual request
        return rec

    def aggregate(self) -> Aggregate:
        return Aggregate.of(self.session)

    def load_all(self, since: float | None = None) -> list[Record]:
        out: list[Record] = []
        if not self.path.exists():
            return out
        keys = set(Record.__dataclass_fields__)
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn final line must not break the report
                if since and data.get("ts", 0) < since:
                    continue
                out.append(Record(**{k: v for k, v in data.items() if k in keys}))
        return out


# ---------------------------------------------------------------- presentation


def _ms(value: float) -> str:
    if value <= 0:
        return "-"
    if value < 10:
        return f"{value:.1f} ms"  # sub-ms hits must not render as "0 ms"
    if value < 1000:
        return f"{value:.0f} ms"
    return f"{value / 1000:.1f} s"


def _size(num_bytes: int) -> str:
    if num_bytes < 1_048_576:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes / 1_048_576:.1f} MB"


def _money(value: float) -> str:
    return f"${value:.4f}"


def _tokens(value: int) -> str:
    return f"{value:,}"


def render_plain(agg: Aggregate, cache_stats: dict | None = None) -> str:
    """The /stats screen. Plain English, no percentiles, no jargon."""
    if agg.total == 0:
        return "No questions asked yet."

    lines = [
        "",
        f"Questions asked          {agg.total}",
        f"  answered from cache    {agg.hits:<5} ({agg.hit_rate * 100:.0f}%)",
        f"  asked the model        {agg.misses}",
    ]
    if agg.degraded:
        lines.append(f"  closest saved answer   {agg.degraded}  (model was unavailable)")
    if agg.errors:
        lines.append(f"  failed                 {agg.errors}")

    if agg.hit_latencies or agg.miss_latencies:
        lines += ["", "Speed"]
        if agg.hit_latencies:
            lines.append(
                f"  from cache             {_ms(agg.hit_typical)} typical"
                f"    (fastest {_ms(min(agg.hit_latencies))},"
                f" slowest {_ms(max(agg.hit_latencies))})"
            )
        if agg.miss_latencies:
            lines.append(
                f"  from the model         {_ms(agg.miss_typical)} typical"
                f"    (fastest {_ms(min(agg.miss_latencies))},"
                f" slowest {_ms(max(agg.miss_latencies))})"
            )
        if agg.speedup > 1:
            lines.append(f"  cache answers came back about {agg.speedup:.0f}x faster")

    lines += [
        "",
        "Tokens",
        f"  sent to the model      {_tokens(agg.tokens_sent_in)} in"
        f" / {_tokens(agg.tokens_sent_out)} out",
        f"  avoided by the cache   {_tokens(agg.tokens_avoided_in)} in"
        f" / {_tokens(agg.tokens_avoided_out)} out",
    ]

    if agg.cost_total > 0:
        lines += [
            "",
            "Money",
            f"  spent                  {_money(agg.cost_spent)}",
            f"  saved by the cache     {_money(agg.cost_saved)}"
            f"   ({agg.saved_pct:.0f}% of what this would have cost)",
        ]
    else:
        lines += [
            "",
            "Money",
            "  not tracked for this model -- add prices to config.json to enable",
        ]

    if agg.provider_ok or agg.provider_fail:
        total_calls = agg.provider_ok + agg.provider_fail
        detail = ""
        if agg.error_kinds:
            detail = " - " + ", ".join(f"{n} {k}" for k, n in agg.error_kinds.items())
        lines += [
            "",
            "Model calls",
            f"  {agg.provider_ok} of {total_calls} succeeded{detail}",
        ]

    if cache_stats:
        lines += ["", "Cache"]
        lines.append(
            f"  {cache_stats['entries']} saved answers, {_size(cache_stats['bytes'])}"
            f"   (room for {cache_stats['capacity']:,})"
        )
        if cache_stats.get("evicted"):
            lines.append(
                f"  {cache_stats['evicted']} old answers dropped to make room, least-used first"
            )
        top = cache_stats.get("top_reused") or []
        for i, (prompt, count) in enumerate(top):
            label = "  reused most:" if i == 0 else "              "
            short = prompt if len(prompt) <= 44 else prompt[:41] + "..."
            lines.append(f'{label}  "{short}"   {count} times')

    lines.append("")
    return "\n".join(lines)


def render_detail(agg: Aggregate, title: str = "detail") -> str:
    """The bench / --detail view. Percentiles, each labelled in words."""
    lines = [
        "",
        f"{title}",
        f"  requests                 {agg.total}",
        f"  from cache               {agg.hits}  (exact {agg.exact}, semantic {agg.semantic})",
        f"  asked the model          {agg.misses}",
        f"  degraded / errors        {agg.degraded} / {agg.errors}",
        f"  hit rate                 {agg.hit_rate * 100:.1f}%",
    ]
    for label, values in (("cache", agg.hit_latencies), ("model", agg.miss_latencies)):
        if not values:
            continue
        lines += [
            f"  {label} latency",
            f"    typical (median)       {_ms(percentile(values, 50))}",
            f"    slowest 1 in 20 (p95)  {_ms(percentile(values, 95))}",
            f"    slowest 1 in 100 (p99) {_ms(percentile(values, 99))}",
            f"    fastest / slowest      {_ms(min(values))} / {_ms(max(values))}",
        ]
    lines += [
        f"  model calls avoided      {agg.calls_avoided}",
        f"  tokens avoided           {_tokens(agg.tokens_avoided_in)} in"
        f" / {_tokens(agg.tokens_avoided_out)} out",
    ]
    if agg.cost_total > 0:
        lines.append(
            f"  cost spent / saved       {_money(agg.cost_spent)} / {_money(agg.cost_saved)}"
        )
    lines.append("")
    return "\n".join(lines)


def estimate_cost(info, in_tokens: int, out_tokens: int) -> float:
    """USD for one call, or 0.0 when we have no trustworthy price."""
    if info is None or info.price_in is None or info.price_out is None:
        return 0.0
    return (in_tokens * info.price_in + out_tokens * info.price_out) / 1_000_000
