# semcache

A semantic caching layer that sits in front of any LLM. Ask a question once; ask
it again in different words and the answer comes back from local disk in
milliseconds, with no API call, no tokens billed, and no rate limit.

```
you > Why is our checkout API returning 504 on large carts?

  asked the model · 2.4s · 52 in / 890 out · $0.0055
  A 504 on large carts usually points to a request exceeding an upstream
  timeout rather than a hard failure. Check three things in order...

you > checkout endpoint gives 504 errors on big carts, why?

  from cache · 27ms · 52 in / 890 out · saved $0.0055 · 96% match
  A 504 on large carts usually points to a request exceeding an upstream
  timeout rather than a hard failure. Check three things in order...
```

Different words, no shared phrasing, no question mark — reused anyway. That is
the whole product.

---

## Install

Pick one. All three give you the same `semcache` command.

```bash
# 1. from source, with the offline embedder and every provider
pip install -e ".[all]"

# 2. minimal: core + just the provider you use
pip install -e ".[local,anthropic]"     # or [openai] / [gemini]

# 3. Docker, nothing installed on the host
docker build -t semcache .
docker run -it -v semcache-data:/data -e ANTHROPIC_API_KEY=sk-ant-... semcache
```

Then run it:

```bash
semcache                 # the REPL
python -m semcache       # identical, no install needed
```

## First run

```
$ semcache

Paste your LLM API key (hidden): ****************************
  This looks like an anthropic key. Use it? [Y/n] y

  Pick a model:
    1) claude-sonnet-5      $3.00/$15.00 per Mtok      balanced, fastest  (default)
    2) claude-opus-5        $5.00/$25.00 per Mtok      most capable
    3) claude-haiku-4-5     $1.00/$5.00 per Mtok       cheapest, lowest latency
  Model [1-3, Enter for 1]: 1
  Save the key to /home/you/.semcache/.env? [Y/n] y
  Saved. You won't be asked again.

Ready. 0 answers cached.
Type your question. /help for commands, /exit to quit.
```

Paste a key, confirm the provider, pick a model. Every later start is two lines.
The key is read from `--api-key`, then `SEMCACHE_API_KEY`, then
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY`, then `.env` — so in
CI or Docker you never see a prompt at all.

## Commands

| | |
|---|---|
| `semcache` | the REPL |
| `semcache ask "question"` | answer one question and exit |
| `semcache stats [--since 7d] [--detail]` | what the cache has saved you |
| `semcache bench [--offline]` | replay a query set, cold vs warm |
| `semcache clear` | forget every saved answer |

In the REPL: `/stats`, `/clear`, `/help`, `/exit`. That is the whole surface.

## What you get back

Every answer carries the same four facts: where it came from, how long it took,
tokens in/out, and what it cost or saved.

```
  asked the model · 2.4s · 52 in / 890 out · $0.0055
  from cache · 0.2ms · 52 in / 890 out · saved $0.0055
  from cache · 27ms · 52 in / 890 out · saved $0.0055 · 96% match
  asked the model · 1.9s · 44 in / 620 out · $0.0039 · closest saved answer was only 81% match
  from cache · 31ms · 84% match · the model is rate-limited, so this is the closest saved answer
```

Cached answers report the original answer's token counts, because that is
exactly what you did not pay for. The fourth line is the important one: when
nothing is close enough, it says how close it got, which is how you tune the
threshold.

`/stats` is plain English, no percentiles:

```
Questions asked          312
  answered from cache    218    (70%)
  asked the model         94

Speed
  from cache             27 ms typical    (fastest 0.2 ms, slowest 41 ms)
  from the model         2.2 s typical    (fastest 1.1 s, slowest 5.3 s)
  cache answers came back about 81x faster

Tokens
  sent to the model      6,180 in / 71,400 out
  avoided by the cache   11,900 in / 168,300 out

Money
  spent                  $1.14
  saved by the cache     $2.64   (70% of what this would have cost)

Model calls
  91 of 94 succeeded - 3 rate_limit

Cache
  94 saved answers, 3.1 MB   (room for 5,000)
  reused most:  "Why is our checkout API returning 504..."   19 times
```

Percentiles still exist — `semcache stats --detail` and `bench` print them,
labelled in words ("slowest 1 in 20"). They are kept out of the chat flow.

## How it works

```
prompt
 → normalize (trim, collapse whitespace, casefold) → sha256
 → tier 1  exact hash lookup in SQLite        → ~0.2ms, no embedding at all
 → embed   local ONNX ~25ms | provider API ~150ms
 → tier 2  FAISS top-k, score >= threshold    → ~1ms
 → tier 3  call the model, then store it
 → record a metric for every outcome, errors included
```

- **Vectors are L2-normalized on write**, so FAISS inner product *is* cosine
  similarity. There is no cosine function anywhere in the codebase.
- **The index is `IndexIDMap2(IndexFlatIP)`**, not a bare flat index, because a
  bare one has no `remove_ids` — and without removal, LRU eviction could never
  shrink the index while SQLite shrank beside it.
- **LRU, not FIFO.** Every hit bumps `last_used_at`, so an old answer that keeps
  getting reused survives and a recent one nobody wants gets dropped.
- **SQLite is the source of truth.** Vectors are stored there as well as in
  FAISS, so a deleted or corrupt index file is a rebuild, not data loss. Delete
  `~/.semcache/index/*.faiss` and the next start rebuilds it.
- **Context is used sparingly.** The bare prompt is embedded by default, and
  conversation context is folded in only when the prompt cannot stand alone
  (anaphora, or under five words). Embedding `context + prompt` unconditionally
  destroys cross-session reuse, and reuse across sessions is where the savings
  are.
- **Failure does not dead-end.** On a rate limit, timeout or outage, the closest
  answer above a lower bar is served, clearly labelled, and counted separately —
  never folded into the hit rate.
- **Bad answers are never cached.** Empty, oversized, or length-truncated
  responses are rejected, because one cached truncated answer is served forever.

## Multiple projects, multiple people

The cache is shared by default — that is where the savings are largest. Opt into
isolation when answers should not cross a boundary:

```bash
semcache --project api-service        # ~/.semcache/projects/api-service/
SEMCACHE_PROJECT=data-pipeline semcache
```

Each project gets its own database, index and metrics. Switching the embedder or
its dimension also opens a fresh namespace automatically, because vectors from
different embedders are not comparable and mixing them returns nonsense.

## In your own code

```python
from semcache import SemCache

with SemCache(project="my-service") as cache:
    answer = cache.ask("What causes memory fragmentation in Python?")
    print(answer.text, answer.source, answer.latency_ms)  # 'miss'  2410.5

    again = cache.ask("Why do Python services fragment the heap?")
    print(again.source, again.similarity)  # 'semantic'  0.96

    print(cache.report())
```

`ask()` takes an `on_chunk` callback for streaming. With no arguments `SemCache()`
reads the key from the environment and infers the provider from its shape.
Construct one per worker; it is not thread-safe.

## Configuration

Resolution order: CLI flag → `SEMCACHE_*` env var → `~/.semcache/config.json` →
default. Any field can be set by any of them.

| | default | |
|---|---|---|
| `--threshold` | `0.95` | cosine score needed to reuse an answer |
| `--max-entries` | `5000` | capacity before LRU eviction |
| `--embedder` | `auto` | `local` (ONNX) · `api` · `hash` (tests) |
| `--scope` | `global` | `session` restricts reuse to one session |
| `--ttl-seconds` | off | expire answers after this long |
| `--project` | none | isolate this project's cache |
| `--effort` | server default | Anthropic reasoning effort; `low` cuts miss latency |
| `--offline` | off | built-in stub model: no key, no network, no cost |
| `--no-log-prompts` | off | record hashes only, never prompt text |

Prices for cost reporting live in `config.json` under `prices`. Anthropic rates
ship built in; OpenAI and Google are deliberately left unset rather than
guessed, so `/stats` says "not tracked" instead of showing an invented number.

## Benchmark

```
$ semcache bench --offline --embedder local --assert-targets

replaying queries.jsonl · 12 groups · 48 prompts · stub · local:BAAI/bge-small-en-v1.5

                              cold      warm
  requests                      48        48
  answered from cache            0        48
  hit rate                    0.0%    100.0%
  model calls                   48         0
  average                    244.8 ms      17.7 ms
  average change                      -92.8%

targets
  ok   no false hits on trap questions  0.00 (target == 0)
  ok   exact hit typical                0.18 ms (target <= 1.0 ms)
  ok   cache hit typical (local)        13.67 ms (target <= 60.0 ms)
  ok   warm start                       16.83 ms (target <= 400.0 ms)
```

`--offline` uses a stub model, so this runs in CI with no key and no spend.
Latency targets are per-backend and measured, not aspirational: a semantic hit is
dominated by embedding the query, which costs ~25ms with bge-small and ~150ms
over an API.

The **trap questions** are the important row. Each group carries a
similar-sounding question with a genuinely different answer ("returning 504" vs
"returning 401", "add an index" vs "drop an index"). Serving a cached answer to
one of those is the failure mode that matters, and it fails the build.

## Known limitation: reversed relations

Threshold 0.95 was chosen by measurement, not taste. On the bundled query set
with `bge-small-en-v1.5`:

| threshold | real paraphrases kept | trap questions admitted |
|---|---|---|
| 0.90 | 23 / 24 | 6 / 12 |
| **0.95** | **23 / 24** | **1 / 12** |
| 0.98 | 18 / 24 | 1 / 12 |

The one trap that survives every threshold:

> "Why is my Postgres query **doing a sequential scan instead of using the index**?"
> "Why is my Postgres query **using the index instead of a sequential scan**?"

These score **0.989** — higher than six genuine paraphrases. Same words,
reversed relation. Embeddings discard word order by design, so no threshold
separates them, and raising the threshold high enough to try would throw away
real paraphrases. It is tracked in the query set as a `hard_trap`, reported on
every benchmark run so a regression stays visible, and excluded from the gate
because gating on something unachievable would just mean a permanently red
build. If your domain is full of such inversions, raise `--threshold` and expect
a lower hit rate.

Other deliberate ceilings, each marked in the source:

- One lock around cache mutations — right for a single-user REPL, shard per
  namespace for concurrent traffic.
- `IndexFlatIP` is an exact brute-force scan and `remove_ids` compacts in O(n) —
  correct and fast to ~100k entries, then switch to `IndexIVFFlat`.
- Anaphora detection is a keyword heuristic, not coreference resolution. See
  [bonus.md](bonus.md) Part A.
- The `hash` embedder is bag-of-words. It exists so tests and CI can exercise
  cache mechanics with no key, no network and no model download; it has no
  semantic understanding and is not for real use.

## Development

```bash
pip install -e ".[all,dev]"
pytest                                              # no key, no network needed
ruff check . && ruff format --check .
semcache bench --offline --embedder hash --assert-targets
docker run --rm --network none semcache bench --offline --embedder local --assert-targets
```

CI runs lint, the test suite on Linux and Windows across Python 3.9/3.11/3.12, an
offline smoke test, the real-embedder benchmark, and a Docker build that verifies
the image works with `--network none` and that the cache survives across separate
containers.

## Requirements

Python 3.9+. Core install is `numpy`, `faiss-cpu`, `python-dotenv`. The local
embedder uses `fastembed` (ONNX, ~50MB) rather than `sentence-transformers`,
which would pull in ~2.5GB of PyTorch. Provider SDKs are optional extras,
imported lazily, so installing one never drags in the others.

## Licence

MIT — see [LICENSE](LICENSE).
