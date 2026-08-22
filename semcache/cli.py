"""The terminal app: first-run setup, the REPL, and the bench/stats/clear commands.

Design rule for everything printed here: one status line per answer carrying the
same four facts (where it came from, how long, tokens in/out, cost or saving),
and no jargon anywhere in the chat flow.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import time

from . import __version__
from .cache import SemanticCache
from .chat import ChatSession
from .config import Config
from .embedders import build_embedder
from .metrics import Aggregate, Metrics, render_detail, render_plain
from .providers import (
    CATALOG,
    ProviderError,
    build_provider,
    default_model,
    detect_provider,
)

KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _setup_console() -> None:
    """A default Windows console is cp1252: one non-ASCII character in a model
    response would raise UnicodeEncodeError and kill the REPL."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass


def out(text: str = "") -> None:
    print(text, flush=True)


# --------------------------------------------------------------------- key flow


def _load_env_files(cfg: Config) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(cfg.env_path, override=False)
    load_dotenv(".env", override=False)


def resolve_key(cfg: Config, cli_key: str | None) -> tuple[str, str]:
    """Find a key and settle on a provider. Returns (key, provider).

    Order: --api-key, SEMCACHE_API_KEY, provider-specific env, .env, then ask.
    The detected provider is always confirmed rather than assumed.
    """
    if cfg.offline:
        return "", "stub"

    _load_env_files(cfg)
    key = cli_key or os.environ.get("SEMCACHE_API_KEY") or ""
    if not key:
        for provider, var in KEY_ENV.items():
            found = os.environ.get(var)
            if found:
                key, guess = found, provider
                break
        else:
            guess = None
    else:
        guess = None

    if not key:
        out(f"semcache {__version__}  -  no API key found")
        try:
            key = getpass.getpass("Paste your LLM API key (hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nNo key given.") from None
    if not key:
        raise SystemExit("No key given.")

    provider = cfg.provider or guess or detect_provider(key)
    provider = _confirm_provider(provider)
    return key, provider


def _confirm_provider(guess: str | None) -> str:
    names = list(CATALOG)
    if guess:
        answer = input(f"  This looks like an {guess} key. Use it? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            return guess
    out("  Which provider is this key for?")
    for i, name in enumerate(names, 1):
        out(f"    {i}) {name}")
    while True:
        choice = input(f"  Provider [1-{len(names)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        if choice in names:
            return choice


def choose_model(cfg: Config, provider: str) -> str:
    """Let the user pick, once. Saved to config.json so it is never asked twice."""
    if cfg.model:
        return cfg.model
    if provider == "stub":
        return "stub-1"

    options = CATALOG.get(provider, [])
    if not options:
        return default_model(provider)

    out("")
    out("  Pick a model:")
    for i, info in enumerate(options, 1):
        price = (
            f"${info.price_in:.2f}/${info.price_out:.2f} per Mtok"
            if info.price_in is not None
            else "price not tracked"
        )
        default = "  (default)" if i == 1 else ""
        out(f"    {i}) {info.id:<20} {price:<26} {info.blurb}{default}")
    choice = input(f"  Model [1-{len(options)}, Enter for 1]: ").strip()
    picked = options[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(
        options
    ) else options[0]
    return picked.id


def offer_to_save(cfg: Config, key: str, provider: str, model: str) -> None:
    if cfg.offline or cfg.env_path.exists():
        _persist_model(cfg, provider, model)
        return
    answer = input(f"  Save the key to {cfg.env_path}? [Y/n] ").strip().lower()
    if answer in ("", "y", "yes"):
        try:
            cfg.home.mkdir(parents=True, exist_ok=True)
            cfg.env_path.write_text(f"{KEY_ENV[provider]}={key}\n", encoding="utf-8")
            try:
                os.chmod(cfg.env_path, 0o600)  # no-op on Windows, correct on POSIX
            except OSError:
                pass
            out("  Saved. You won't be asked again.")
        except OSError as exc:
            out(f"  Could not save the key ({exc}). It will be asked for next time.")
    _persist_model(cfg, provider, model)


def _persist_model(cfg: Config, provider: str, model: str) -> None:
    """Remember the provider/model choice so startup stays two lines next time."""
    import json

    try:
        cfg.home.mkdir(parents=True, exist_ok=True)
        existing = {}
        if cfg.config_path.exists():
            existing = json.loads(cfg.config_path.read_text(encoding="utf-8"))
        existing.update({"provider": provider, "model": model})
        cfg.config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass  # a config we cannot write just means one extra question next run


# ------------------------------------------------------------------ status line


def status_line(turn) -> str:
    r = turn.record
    tokens = f"{r.prompt_tokens:,} in / {r.response_tokens:,} out"
    took = _short(r.total_ms)

    if turn.outcome == "error":
        return f"  {turn.note}"
    if turn.outcome in {"exact", "semantic", "degraded"}:
        bits = ["from cache", took, tokens]
        if r.cost_saved_usd:
            bits.append(f"saved ${r.cost_saved_usd:.4f}")
        if turn.note:
            bits.append(turn.note)
        return "  " + " · ".join(bits)

    bits = ["asked the model", took, tokens]
    if r.cost_usd:
        bits.append(f"${r.cost_usd:.4f}")
    if turn.note:
        bits.append(turn.note)
    return "  " + " · ".join(bits)


def _short(ms: float) -> str:
    if ms < 10:
        return f"{ms:.1f}ms"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


# -------------------------------------------------------------------- the REPL


HELP = """
  Just type a question and press Enter.

  /stats   what the cache has saved you
  /clear   forget every saved answer
  /help    this
  /exit    quit  (Ctrl-D also works)
"""


def build_session(cfg: Config, key: str, provider_name: str, model: str) -> ChatSession:
    provider = build_provider(provider_name, key, model, cfg)
    embedder = build_embedder(cfg, provider)
    cache = SemanticCache(cfg, embedder)
    metrics = Metrics(cfg.metrics_path, log_prompts=cfg.log_prompts)
    session = ChatSession(cfg, provider, embedder, cache, metrics)
    return session


def run_repl(cfg: Config, session: ChatSession) -> int:
    stats = session.cache.stats()
    out("")
    out(f"Ready. {stats['entries']} answers cached.")
    out("Type your question. /help for commands, /exit to quit.")
    session.warm_async()

    while True:
        try:
            prompt = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            out("")
            break

        if not prompt:
            continue
        lowered = prompt.lower()
        if lowered in ("/exit", "/quit"):
            break
        if lowered == "/help":
            out(HELP)
            continue
        if lowered == "/stats":
            out(render_plain(session.metrics.aggregate(), session.cache.stats()))
            continue
        if lowered == "/clear":
            confirm = input("  Forget every saved answer? [y/N] ").strip().lower()
            if confirm in ("y", "yes"):
                removed = session.cache.clear()
                out(f"  Cleared {removed} saved answers.")
            continue

        _answer(session, prompt)

    session.cache.close()
    out(f"Saved. {session.cache.store.count()} answers cached.")
    return 0


def _answer(session: ChatSession, prompt: str) -> None:
    """Print a live status line, stream the answer under it, then finalise."""
    started = time.perf_counter()
    state = {"started": False}

    def on_chunk(piece: str) -> None:
        if not state["started"]:
            # Clear the live line before the answer starts arriving.
            sys.stdout.write("\r" + " " * 60 + "\r")
            state["started"] = True
        sys.stdout.write(piece)
        sys.stdout.flush()

    sys.stdout.write("  asking the model...")
    sys.stdout.flush()
    turn = session.ask(prompt, on_chunk=on_chunk)
    if not state["started"]:
        sys.stdout.write("\r" + " " * 60 + "\r")
    elapsed = time.perf_counter() - started
    del elapsed

    out("")
    out(status_line(turn))


# ------------------------------------------------------------------ subcommands


def cmd_chat(cfg: Config, args) -> int:
    key, provider = resolve_key(cfg, args.api_key)
    model = choose_model(cfg, provider)
    if not cfg.offline:
        offer_to_save(cfg, key, provider, model)
    session = build_session(cfg, key, provider, model)
    return run_repl(cfg, session)


def cmd_ask(cfg: Config, args) -> int:
    """One-shot: answer a single prompt and exit. Used by CI's smoke test."""
    key, provider = resolve_key(cfg, args.api_key)
    model = cfg.model or (default_model(provider) if provider != "stub" else "stub-1")
    session = build_session(cfg, key, provider, model)
    turn = session.ask(args.prompt)
    out(turn.text.strip())
    out(status_line(turn))
    session.cache.close()
    return 0 if turn.outcome != "error" else 1


def cmd_stats(cfg: Config, args) -> int:
    metrics = Metrics(cfg.metrics_path)
    since = time.time() - _parse_since(args.since) if args.since else None
    records = metrics.load_all(since)
    if not records:
        out("No recorded questions yet.")
        return 0
    agg = Aggregate.of(records)
    out(render_detail(agg, "all recorded requests") if args.detail else render_plain(agg))
    return 0


def cmd_clear(cfg: Config, args) -> int:
    embedder = build_embedder(Config.load(home=cfg.home, embedder="hash"), None)
    cache = SemanticCache(cfg, embedder)
    removed = cache.clear()
    cache.close()
    out(f"Cleared {removed} saved answers from {cfg.db_path}")
    return 0


def cmd_bench(cfg: Config, args) -> int:
    from .bench import run_bench

    return run_bench(cfg, args)


def _parse_since(text: str) -> float:
    units = {"m": 60, "h": 3600, "d": 86400}
    text = text.strip().lower()
    if text and text[-1] in units and text[:-1].replace(".", "").isdigit():
        return float(text[:-1]) * units[text[-1]]
    return float(text)


# ----------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semcache",
        description="A semantic cache in front of any LLM. Answers you have "
        "effectively asked before come back in milliseconds without an API call.",
    )
    parser.add_argument("--version", action="version", version=f"semcache {__version__}")

    # Tuning knobs live here so a normal user never needs one.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--api-key", help="provider key (else env, .env, or a prompt)")
    common.add_argument("--provider", choices=list(CATALOG) + ["stub"])
    common.add_argument("--model")
    common.add_argument("--embedder", choices=["auto", "local", "api", "hash"])
    common.add_argument("--threshold", type=float, help="cosine score needed to reuse (0.90)")
    common.add_argument("--max-entries", type=int, dest="max_entries")
    common.add_argument("--ttl-seconds", type=float, dest="ttl_seconds")
    common.add_argument("--scope", choices=["global", "session"])
    common.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    common.add_argument("--home", help="cache directory (default ~/.semcache)")
    common.add_argument("--offline", action="store_true", default=None,
                        help="use the built-in stub model: no key, no network, no cost")
    common.add_argument("--no-log-prompts", action="store_false", dest="log_prompts",
                        default=None, help="record only hashes, not prompt text")
    common.add_argument("--debug", action="store_true", default=None)

    subs = parser.add_subparsers(dest="command")

    subs.add_parser("chat", parents=[common], help="start the interactive REPL (default)")

    ask = subs.add_parser("ask", parents=[common], help="answer one prompt and exit")
    ask.add_argument("prompt")

    stats = subs.add_parser("stats", parents=[common], help="report on recorded requests")
    stats.add_argument("--since", help="e.g. 7d, 12h, 30m")
    stats.add_argument("--detail", action="store_true", help="show percentiles")

    subs.add_parser("clear", parents=[common], help="forget every saved answer")

    bench = subs.add_parser("bench", parents=[common], help="replay a query set")
    bench.add_argument("--queries", help="path to a JSONL query set")
    bench.add_argument("--assert-targets", action="store_true",
                       help="exit non-zero if a latency target or trap pair regresses")
    bench.add_argument("--report", default="bench-report.json")

    return parser


_HANDLERS = {
    "chat": cmd_chat,
    "ask": cmd_ask,
    "stats": cmd_stats,
    "clear": cmd_clear,
    "bench": cmd_bench,
}

_CONFIG_KEYS = (
    "provider", "model", "embedder", "threshold", "max_entries", "ttl_seconds",
    "scope", "effort", "home", "offline", "log_prompts", "debug",
)


def main(argv: list[str] | None = None) -> int:
    _setup_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "chat"
    if args.command is None:
        # Bare `semcache` means chat, but argparse has not parsed the shared
        # flags in that case -- reparse so `semcache --offline` still works.
        args = parser.parse_args(["chat", *(argv if argv is not None else sys.argv[1:])])

    cfg = Config.load(**{k: getattr(args, k, None) for k in _CONFIG_KEYS})
    try:
        return _HANDLERS[command](cfg, args)
    except ProviderError as exc:
        if cfg.debug:
            raise
        out(f"  {exc.user_message}")
        return 1
    except KeyboardInterrupt:
        out("")
        return 130
    except RuntimeError as exc:  # schema mismatch and similar, already worded
        if cfg.debug:
            raise
        out(f"  {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
