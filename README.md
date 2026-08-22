# semcache

A semantic caching layer that sits in front of any LLM. Ask a question once; ask
it again in different words and the answer comes back from local disk in
milliseconds — no API call, no tokens billed, no rate limit.

```
you > What causes memory fragmentation in long-running Python processes?

  asked the model · 15.8s · 21 in / 2,518 out
  Memory fragmentation in long-running Python processes comes from...

you > What causes memory fragmentation in long running Python processes?

  from cache · 28ms · 21 in / 2,518 out · 100% match
  Memory fragmentation in long-running Python processes comes from...
```

Measured on a real session: **cache hits 0.3–90ms, model calls 5–13s.** Two
orders of magnitude, and the hits cost nothing.

---

## Contents

- [Quickstart](#quickstart) · [How it works](#how-it-works) · [Providers](#providers)
- [Commands](#commands) · [Reading the output](#reading-the-output) · [Configuration](#configuration)
- [Use it in your own code](#use-it-in-your-own-code) · [Docker](#docker)
- [Tuning the threshold](#tuning-the-threshold) ← read before changing it
- [Limitations](#limitations) · [Troubleshooting](#troubleshooting) · [Development](#development)

---

## Quickstart

Requires Python 3.9+.

```bash
git clone https://github.com/manojkmohan5/Semantic-Caching-Framework-for-Context-Aware-Conversational-AI.git
cd Semantic-Caching-Framework-for-Context-Aware-Conversational-AI
pip install -e ".[all]"
```

Check it works without spending anything or needing a key:

```bash
semcache bench --offline --embedder local --assert-targets
```

That replays 48 questions cold then warm and asserts its latency targets. All
`ok` means you are good.

Then start it:

```bash
semcache
```

First run asks three things, once:

```
semcache 0.1.0  -  no API key found
  (the paste stays hidden, so nothing appears as you type)
Paste your LLM API key (hidden):
  This looks like an anthropic key. Use it? [Y/n] y

  Pick a model:
    1) claude-sonnet-5      $3.00/$15.00 per Mtok      balanced, fastest  (default)
    2) claude-opus-5        $5.00/$25.00 per Mtok      most capable
    3) claude-haiku-4-5     $1.00/$5.00 per Mtok       cheapest, lowest latency
  Model [1-3, Enter for 1]: 1
  Save the key to ~/.semcache/.env? [Y/n] y
  Saved. You won't be asked again.

Ready. 0 answers cached.
Type your question. /help for commands, /exit to quit.
```

Every later run goes straight to `Ready.` Type questions; `/stats` shows what you
have saved; `/exit` quits.

**Install only what you need** — provider SDKs and the local embedder are optional
extras, lazily imported:

```bash
pip install -e ".[local,anthropic]"    # offline embedder + Claude
pip install -e ".[openai]"             # OpenAI and every compatible host
pip install -e ".[all]"                # everything
```

---

## How it works

```
prompt
 → normalize (trim, collapse whitespace, casefold) → sha256
 → tier 1  exact hash lookup in SQLite     → ~0.3ms, no embedding at all
 → embed   local ONNX ~30ms | provider API ~150ms
 → tier 2  FAISS top-k, score >= threshold → ~0.5ms
 → tier 3  call the model, then store it
 → record a metric for every outcome, errors included
```

- **Vectors are L2-normalized on write**, so FAISS inner product *is* cosine
  similarity. There is no cosine function anywhere in the codebase.
- **The index is `IndexIDMap2(IndexFlatIP)`**, not a bare flat index — a bare one
  has no `remove_ids`, so LRU eviction could never shrink it while SQLite shrank
  beside it.
- **LRU, not FIFO.** Every hit bumps a monotonic counter, so an old answer that
  keeps getting reused survives while a recent one nobody wants is dropped. The
  counter is deliberately not a wall clock: `time.time()` has ~15.6ms resolution
  on Windows, so timestamps tie, and ties made eviction pick the wrong entry.
- **SQLite is the source of truth.** Vectors live there as well as in FAISS, so a
  deleted or corrupt index is a rebuild, not data loss. Delete
  `~/.semcache/index/*.faiss` and the next start rebuilds it.
- **Context is used sparingly.** The bare prompt is embedded by default; recent
  turns are folded in only when the prompt cannot stand alone (anaphora, or under
  five words). Embedding `context + prompt` unconditionally destroys
  cross-session reuse, and that reuse is where the savings are.
- **Failure does not dead-end.** On a rate limit, timeout or outage, the closest
  answer above a lower bar is served, clearly labelled, and counted separately —
  never folded into the hit rate.
- **Bad answers are never cached.** Empty, oversized and length-truncated
  responses are rejected, because one cached truncated answer is served forever.

---

## Providers

Three native SDK paths, plus every OpenAI-compatible host through one client.

| | how |
|---|---|
| Anthropic, OpenAI, Gemini | detected from the key, native SDK |
| **Grok** (xAI) | detected from `xai-...` keys |
| **OpenRouter** | detected from `sk-or-...`; `--model anthropic/claude-sonnet-4.5` |
| **NVIDIA NIM** | detected from `nvapi-...` keys |
| **DeepSeek** | `--provider deepseek --model deepseek-chat` |
| **Kimi** (Moonshot) | `--provider kimi --model kimi-k2-0905-preview` |
| **GLM** (Zhipu) | `--provider glm --model glm-4.6` |
| Groq, Together | `--provider groq` / `together` |
| **Ollama, LM Studio** (local) | `--provider ollama --model llama3.1` |
| anything else | `--provider custom --base-url https://your-host/v1` |

```bash
semcache --provider grok --model grok-4
semcache --provider ollama --model llama3.1        # fully offline, no key
```

`--base-url` overrides any preset, so a corporate gateway or proxy works too.

Only `sk-ant-`, `sk-or-`, `xai-`, `nvapi-` and `AIza` prefixes are unambiguous.
DeepSeek, Kimi and most compatible hosts also issue `sk-...` keys, so those need
`--provider`. Provider and model are saved after the first run.

These hosts rename models often, so semcache asks once rather than shipping a
guess that would 404.

---

## Commands

| | |
|---|---|
| `semcache` | the chat loop (default) |
| `semcache ask "question"` | answer one question and exit |
| `semcache stats [--since 7d] [--detail]` | what the cache has saved you |
| `semcache bench [--offline]` | replay a query set, cold vs warm |
| `semcache clear` | forget every saved answer |

In the chat loop: `/stats`, `/clear`, `/help`, `/exit`.

---

## Reading the output

Every answer carries the same four facts: where it came from, how long it took,
tokens in/out, and what it cost or saved.

```
  asked the model · 2.4s · 52 in / 890 out · $0.0055
  from cache · 0.3ms · 52 in / 890 out · saved $0.0055
  from cache · 28ms · 52 in / 890 out · saved $0.0055 · 100% match
  asked the model · 1.9s · 44 in / 620 out · closest saved answer was only 81.8% match
  from cache · 31ms · 84.2% match · the model is rate-limited, so this is the closest saved answer
```

Cached answers report the original answer's token counts, because that is exactly
what you did not pay for.

**The fourth line is the one to watch.** `closest saved answer was only 81.8%
match` means the cache *was* consulted, found its nearest neighbour, and correctly
refused it. It is also how you tune — see
[Tuning the threshold](#tuning-the-threshold).

`/stats` in plain English:

```
Questions asked          13
  answered from cache    4     (31%)
  asked the model        9

Speed
  from cache             34 ms typical    (fastest 0.3 ms, slowest 90 ms)
  from the model         5.8 s typical    (fastest 4.9 s, slowest 13.4 s)
  cache answers came back about 171x faster

Tokens
  sent to the model      6,180 in / 71,400 out
  avoided by the cache   11,900 in / 168,300 out

Money
  spent                  $1.14
  saved by the cache     $2.64   (70% of what this would have cost)

Cache
  9 saved answers, 28 KB   (room for 5,000)
  reused most:  "What is the fastest car?"   3 times
```

Percentiles are deliberately kept out of the chat flow. `semcache stats --detail`
and `bench` print them, labelled in words ("slowest 1 in 20").

Every request is also appended to `~/.semcache/metrics.jsonl`, one JSON object
per line, so you can re-slice a session later.

---

## Configuration

CLI flag → `SEMCACHE_*` env var → `~/.semcache/config.json` → default. Any field
can be set by any of them.

| | default | |
|---|---|---|
| `--threshold` | `0.95` | cosine score needed to reuse an answer |
| `--max-entries` | `5000` | capacity before LRU eviction |
| `--embedder` | `auto` | `local` (ONNX) · `api` · `hash` (tests only) |
| `--scope` | `global` | `session` restricts reuse to one session |
| `--ttl-seconds` | off | expire answers after this long |
| `--project` | none | isolate this project's cache from others |
| `--base-url` | preset | endpoint for an OpenAI-compatible host |
| `--effort` | server default | Anthropic reasoning effort; `low` cuts miss latency |
| `--offline` | off | built-in stub model: no key, no network, no cost |
| `--no-log-prompts` | off | record hashes only, never prompt text |

**Multiple projects.** The cache is shared by default, because that is where the
savings are largest. Opt into isolation when answers should not cross a boundary:

```bash
semcache --project api-service        # ~/.semcache/projects/api-service/
SEMCACHE_PROJECT=data-pipeline semcache
```

Each project gets its own database, index and metrics; the ONNX model is shared.
Switching embedder or dimension also opens a fresh namespace automatically,
because vectors from different embedders are not comparable.

**Where things live** — under `~/.semcache/`: `cache.db` (answers + vectors),
`index/*.faiss`, `metrics.jsonl`, `models/` (ONNX model), `.env` (your key,
`0600` on POSIX), `config.json`.

---

## Use it in your own code

```python
from semcache import SemCache

with SemCache(project="my-service") as cache:
    answer = cache.ask("What causes memory fragmentation in Python?")
    print(answer.text)
    print(answer.source, answer.latency_ms)  # 'miss'  2410.5

    again = cache.ask("What causes memory fragmentation in Python")
    print(again.source, again.similarity)  # 'semantic'  0.99

    print(cache.report())
```

`answer` carries `.text`, `.source` (`exact`/`semantic`/`degraded`/`miss`/`error`),
`.similarity`, `.latency_ms`, `.input_tokens`, `.output_tokens`, `.cost_usd`,
`.cost_saved_usd`, `.from_cache`. Pass `on_chunk=print` to stream.

With no arguments, `SemCache()` reads the key from the environment and infers the
provider from its shape. Construct one per worker — it is not thread-safe.

Runnable version: [examples/use_the_api.py](examples/use_the_api.py)

```bash
python examples/use_the_api.py --offline        # no key, no cost
python examples/use_the_api.py --key sk-... --provider openrouter --model "z-ai/glm-5.2:free"
```

---

## Docker

```bash
docker build -t semcache .
docker run -it -v semcache-data:/data -e ANTHROPIC_API_KEY=sk-ant-... semcache
```

The `/data` volume is what makes the cache worth having — answers survive
container restarts. The ONNX model is baked into the image at build time and lives
*outside* `/data`, so a fresh volume never triggers a re-download. Verified in CI:
the image passes the full benchmark with `--network none`.

`docker compose run --rm semcache` also works; the compose file forwards
`SEMCACHE_API_KEY`, `SEMCACHE_PROVIDER`, `SEMCACHE_MODEL`, `SEMCACHE_BASE_URL` and
`SEMCACHE_PROJECT`.

---

## Tuning the threshold

**Do not lower it without running `bench`.** This is the one setting that can make
semcache serve a confidently wrong answer.

Every group in the bundled query set carries a *trap*: a similar-sounding question
with a genuinely different answer ("returning 504" vs "returning 401", "add an
index" vs "drop an index"). Measured with `bge-small-en-v1.5`:

| threshold | paraphrases reused | traps wrongly served |
|---|---|---|
| 0.88 | most | **12 of 12** |
| 0.90 | 60% | **10** |
| 0.92 | 54% | **4** |
| 0.93 | 52% | **2** |
| **0.94** | 50% | **0** ← measured floor |
| **0.95** (default) | 50% | **0** |

Below 0.94, "why is checkout returning 504" starts being answered with the 401
answer. The default is 0.95 because a confidently wrong answer is worse than an
extra API call.

If you want more reuse, **phrase questions more consistently** rather than
lowering the bar. Reproduce the table yourself:

```bash
semcache bench --offline --embedder local --threshold 0.90
```

---

## Limitations

**Reversed relations are the known blind spot.** These two score **0.989** — higher
than six genuine paraphrases:

> "Why is my Postgres query **doing a sequential scan instead of using the index**?"
> "Why is my Postgres query **using the index instead of a sequential scan**?"

Same words, reversed relation. Embeddings discard word order by design, so no
threshold separates them, and raising it high enough to try would throw away real
paraphrases. It is tracked in the query set as a `hard_trap`, reported on every
benchmark run so a regression stays visible, and excluded from the pass/fail gate
because gating on something unachievable only means a permanently red build.

Other deliberate ceilings, each marked in the source:

- One lock around cache mutations — right for a single-user REPL; shard per
  namespace for concurrent traffic.
- `IndexFlatIP` is an exact brute-force scan and `remove_ids` compacts in O(n) —
  correct and fast to ~100k entries, then switch to `IndexIVFFlat`.
- Anaphora detection is a keyword heuristic, not coreference resolution. See
  [bonus.md](bonus.md) Part A.
- The `hash` embedder is bag-of-words. It exists so tests and CI can exercise cache
  mechanics with no key, no network and no model download; it has no semantic
  understanding and is not for real use.
- Cost is reported only for models with a known price. Anthropic rates ship built
  in; OpenAI, Gemini and the compatible hosts are left unset rather than guessed,
  so `/stats` says "not tracked" instead of showing an invented number.
- Misses get more expensive deeper into a conversation, because chat history is
  sent with each call. Cache hits therefore save more than the raw output token
  counts suggest.

---

## Troubleshooting

**`402 Insufficient credits`** — the selected model is paid and the account has no
balance. Pick a free or cheaper one: `semcache --model "z-ai/glm-5.2:free"`.

**`the provider does not have a model called ...`** — model ids are hand-typed for
OpenAI-compatible hosts and get renamed often. Check the provider's model list and
pass `--model`.

**Nothing appears while pasting the key** — that is `getpass`; the paste is hidden
by design.

**Everything is a miss** — expected until you repeat or paraphrase something. The
`closest saved answer was only X% match` line proves the cache is being consulted.
Ask the same question twice to see a hit.

**Paraphrases are not hitting** — check the reported match percentage. Scores vary
a lot by wording: two phrasings of one idea measured 0.90 and 0.837 in the same
session. Read [Tuning the threshold](#tuning-the-threshold) before lowering it.

**`cache was written by a newer semcache`** — the cache is from a later version.
Upgrade, or point `--home` elsewhere. Older caches migrate automatically.

**Wipe everything and start over** — `rm -rf ~/.semcache` (Windows:
`Remove-Item -Recurse -Force "$env:USERPROFILE\.semcache"`).

---

## Development

```bash
pip install -e ".[all,dev]"
pytest                    # no key, no network, no model download
ruff check . && ruff format --check .
semcache bench --offline --embedder hash --assert-targets
docker run --rm --network none semcache bench --offline --embedder local --assert-targets
```

CI runs lint, the test suite on Linux and Windows across Python 3.9 and 3.12, an
offline smoke test, the real-embedder benchmark, and a Docker build that verifies
the image works with no network and that the cache survives across separate
containers.

Layout:

```
semcache/
  cli.py         REPL, first-run setup, subcommands
  chat.py        the three-tier request pipeline
  cache.py       lookup, LRU eviction, TTL, drift repair
  index.py       FAISS IndexIDMap2(IndexFlatIP), atomic saves
  store.py       SQLite metadata, exact tier, migrations
  embedders.py   local ONNX | provider API | deterministic hash
  providers.py   Anthropic | OpenAI | Gemini | 11 compatible hosts | stub
  metrics.py     per-request records, one aggregator, two renderers
  config.py      CLI > env > file > default resolution
  bench.py       cold/warm replay with asserted targets
```

## Licence

MIT — see [LICENSE](LICENSE).
