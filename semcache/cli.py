"""The terminal app: first-run setup, the REPL, and the bench/stats/clear commands.

Design rule for everything printed here: one status line per answer carrying the
same four facts (where it came from, how long, tokens in/out, cost or saving),
and no jargon anywhere in the chat flow.
"""

from __future__ import annotations

import argparse
import contextlib
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
    all_provider_names,
    build_provider,
    default_model,
    detect_provider,
    needs_explicit_model,
)
from .ui import (
    Style,
    answer_marker,
    dashboard,
    footer,
    header,
    redraw_last_line,
    user_row,
)
from .ui import prompt as ui_prompt

#: Resolved once at import; colour is dropped automatically when stdout is not a
#: terminal, so piped output and log files stay clean.
STYLE = Style()

KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _setup_console() -> None:
    """A default Windows console is cp1252: one non-ASCII character in a model
    response would raise UnicodeEncodeError and kill the REPL."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")


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
    The guess is confirmed interactively, but never when the provider was given
    explicitly or when there is no terminal to ask at.
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
        out("  (the paste stays hidden, so nothing appears as you type)")
        try:
            key = getpass.getpass("Paste your LLM API key (hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nNo key given.") from None
    if not key:
        raise SystemExit("No key given.")

    # An explicitly configured provider (--provider, SEMCACHE_PROVIDER, or one
    # saved from a previous run) is already the user's answer -- asking again
    # would make every later start interactive, and would break scripting.
    if cfg.provider:
        return key, cfg.provider

    detected = guess or detect_provider(key)
    if not _interactive():
        if not detected:
            raise SystemExit(
                "Could not tell which provider this key is for. "
                "Pass --provider anthropic|openai|gemini."
            )
        return key, detected
    return key, _confirm_provider(detected)


def _prompt(question: str, default: str = "") -> str:
    """input() that cannot crash the app.

    isatty() can report a terminal where no human is typing (piped input, CI
    with a pseudo-tty), and then input() raises EOFError. Treating EOF and
    Ctrl-C as "take the default" keeps every prompt non-fatal.
    """
    try:
        return input(question).strip()
    except (EOFError, KeyboardInterrupt):
        return default


def _interactive() -> bool:
    """False under a pipe, in CI, or in a non-tty container."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _confirm_provider(guess: str | None) -> str:
    names = list(CATALOG)
    if guess:
        answer = _prompt(f"  This looks like an {guess} key. Use it? [Y/n] ", "y").lower()
        if answer in ("", "y", "yes"):
            return guess
    out("  Which provider is this key for?")
    for i, name in enumerate(names, 1):
        out(f"    {i}) {name}")
    while True:
        choice = _prompt(f"  Provider [1-{len(names)}]: ")
        if not choice:
            raise SystemExit("No provider chosen. Pass --provider.")
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        if choice in names:
            return choice


#: Shown when asking for a model, so the question is answerable without
#: leaving the terminal to go and read provider docs.
_MODEL_EXAMPLES = {
    "openrouter": "anthropic/claude-sonnet-4.5",
    "deepseek": "deepseek-chat",
    "kimi": "kimi-k2-0905-preview",
    "glm": "glm-4.6",
    "grok": "grok-4",
    "nvidia": "meta/llama-3.1-70b-instruct",
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "ollama": "llama3.1",
    "lmstudio": "local-model",
}


def choose_model(cfg: Config, provider: str) -> str:
    """Let the user pick, once. Saved to config.json so it is never asked twice."""
    if cfg.model:
        return cfg.model
    if provider == "stub" or cfg.offline:
        return "stub-1"

    options = CATALOG.get(provider, [])
    if not options:
        if not needs_explicit_model(provider):
            return default_model(provider)
        # These hosts each name their models differently and rename them often.
        # Shipping a guess would 400; ask once and save it to config.json.
        example = _MODEL_EXAMPLES.get(provider, "the provider's model id")
        if not _interactive():
            raise SystemExit(f"{provider} needs a model name. Pass --model {example}")
        out("")
        out(f"  Which {provider} model? Example: {example}")
        typed = _prompt("  Model: ")
        if not typed:
            raise SystemExit(f"{provider} needs a model name. Pass --model {example}")
        return typed
    if not _interactive():
        return options[0].id  # nobody to ask; take the documented default

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
    choice = _prompt(f"  Model [1-{len(options)}, Enter for 1]: ")
    picked = (
        options[int(choice) - 1]
        if choice.isdigit() and 1 <= int(choice) <= len(options)
        else options[0]
    )
    return picked.id


def offer_to_save(cfg: Config, key: str, provider: str, model: str) -> None:
    if cfg.offline or cfg.env_path.exists():
        _persist_model(cfg, provider, model)
        return
    if not _interactive():
        _persist_model(cfg, provider, model)
        return
    answer = _prompt(f"  Save the key to {cfg.env_path}? [Y/n] ", "n").lower()
    if answer in ("", "y", "yes"):
        try:
            cfg.home.mkdir(parents=True, exist_ok=True)
            cfg.env_path.write_text(f"{_key_var(provider)}={key}\n", encoding="utf-8")
            # No-op on Windows, correct on POSIX.
            with contextlib.suppress(OSError):
                os.chmod(cfg.env_path, 0o600)
            out("  Saved. You won't be asked again.")
        except OSError as exc:
            out(f"  Could not save the key ({exc}). It will be asked for next time.")
    _persist_model(cfg, provider, model)


def _key_var(provider: str) -> str:
    """Env var name to save the key under.

    Grok, DeepSeek, OpenRouter and the other OpenAI-compatible hosts have no
    dedicated variable, so they use the generic one. Indexing KEY_ENV directly
    raised KeyError for every one of them.
    """
    return KEY_ENV.get(provider, "SEMCACHE_API_KEY")


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


def status_line(turn, style: Style | None = None) -> str:
    """One line, always the same four facts: source, time, tokens, cost/saving."""
    st = style or STYLE
    r = turn.record
    tokens = st.faint(f"{r.prompt_tokens:,} in / {r.response_tokens:,} out")
    took = _short(r.total_ms)
    sep = st.faint(" · ")

    if turn.outcome == "error":
        return "  " + st.error("✗ " + turn.note)

    if turn.outcome == "degraded":
        bits = [st.warn("~ from cache"), st.warn(took), tokens]
        if turn.note:
            bits.append(st.warn(turn.note))
        return "  " + sep.join(bits)

    if turn.outcome in {"exact", "semantic"}:
        bits = [st.hit("✓ from cache"), st.hit(st.strong(took)), tokens]
        if r.cost_saved_usd:
            bits.append(st.hit(f"saved ${r.cost_saved_usd:.4f}"))
        if turn.note:
            bits.append(st.faint(turn.note))
        return "  " + sep.join(bits)

    bits = [st.miss("→ asked the model"), took, tokens]
    if r.cost_usd:
        bits.append(f"${r.cost_usd:.4f}")
    if turn.note:
        bits.append(st.faint(turn.note))
    return "  " + sep.join(bits)


def _short(ms: float) -> str:
    if ms < 10:
        return f"{ms:.1f}ms"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


# -------------------------------------------------------------------- the REPL


def help_text(style: Style) -> str:
    """One source of truth for the command list -- see COMMANDS."""
    lines = ["", "  " + style.faint("Type a question and press Enter."), ""]
    lines += [f"  {style.strong(cmd.ljust(8))} {style.faint(what)}" for cmd, what in COMMANDS]
    lines.append("")
    return "\n".join(lines)


def build_session(cfg: Config, key: str, provider_name: str, model: str) -> ChatSession:
    provider = build_provider(provider_name, key, model, cfg)
    embedder = build_embedder(cfg, provider)
    cache = SemanticCache(cfg, embedder)
    metrics = Metrics(cfg.metrics_path, log_prompts=cfg.log_prompts)
    session = ChatSession(cfg, provider, embedder, cache, metrics)
    return session


COMMANDS = (
    ("/stats", "what the cache has saved you"),
    ("/dash", "show this dashboard again"),
    ("/clear", "forget every saved answer"),
    ("/help", "the command list"),
    ("/exit", "quit  (Ctrl-D also works)"),
)


def _dashboard(cfg: Config, session: ChatSession) -> str:
    """Three panels: what is loaded, what is cached, what it has saved so far.

    Savings are lifetime, read back from metrics.jsonl, because "what has this
    cache done for me" is a more useful thing to land on than a row of zeros.
    """
    st = STYLE
    stats = session.cache.stats()
    lifetime = Aggregate.of(session.metrics.load_all())

    model = st.strong(session.provider.model)
    n = stats["entries"]
    entries = st.hit(f"{n} answer{'' if n == 1 else 's'}") if n else st.faint("empty")
    saved = (
        st.hit(f"${lifetime.cost_saved:.4f} saved")
        if lifetime.cost_saved
        else st.faint("cost not tracked")
    )
    panels = [
        (
            "model",
            [
                st.strong(session.provider.name),
                model,
                st.faint(f"threshold {cfg.threshold:g}"),
            ],
        ),
        (
            "cache",
            [
                entries,
                st.faint(f"{stats['bytes'] / 1024:.0f} KB / {stats['capacity']:,} max"),
                st.faint(session.embedder.id.split("/")[-1]),
            ],
        ),
        (
            "saved so far",
            [
                st.hit(f"{lifetime.hit_rate * 100:.0f}% hit rate")
                if lifetime.total
                else st.faint("no history yet"),
                st.faint(
                    f"{lifetime.calls_avoided} call"
                    f"{'' if lifetime.calls_avoided == 1 else 's'} avoided"
                ),
                saved,
            ],
        ),
    ]
    return dashboard(st, panels, list(COMMANDS))


def run_repl(cfg: Config, session: ChatSession) -> int:
    # Land on the identity block, not the full dashboard: it is what you read
    # every launch, so it stays short. /dash opens the detailed panels.
    lifetime = Aggregate.of(session.metrics.load_all())
    out(
        header(
            STYLE,
            __version__,
            session.provider.name,
            session.provider.model,
            str(cfg.home),
            session.cache.stats()["entries"],
            lifetime.hit_rate,
        )
    )
    session.warm_async()

    while True:
        try:
            prompt = input("\n" + ui_prompt(STYLE)).strip()
        except (EOFError, KeyboardInterrupt):
            out("")
            break

        if not prompt:
            continue
        lowered = prompt.lower()
        if lowered in ("/exit", "/quit"):
            break
        if lowered == "/help":
            out(help_text(STYLE))
            continue
        if lowered == "/stats":
            out(render_plain(session.metrics.aggregate(), session.cache.stats()))
            continue
        if lowered == "/clear":
            confirm = _prompt("  Forget every saved answer? [y/N] ", "n").lower()
            if confirm in ("y", "yes"):
                removed = session.cache.clear()
                out(f"  Cleared {removed} saved answers.")
            continue

        # Re-render the question as a framed block. The terminal already echoed
        # what was typed, so the echo is erased first; when colour is off there
        # is no cursor control to rely on, so the echo is simply left as-is.
        if STYLE.enabled:
            sys.stdout.write(redraw_last_line())
            out(user_row(STYLE, prompt))

        _answer(session, prompt)

        stats = session.cache.stats()
        out(
            footer(
                STYLE,
                session.provider.name,
                session.provider.model,
                stats["entries"],
                session.metrics.aggregate().hit_rate,
            )
        )

    # Read the count BEFORE closing: close() shuts the SQLite connection, and
    # querying it afterwards raised ProgrammingError, so every clean /exit ended
    # in a traceback and a non-zero exit code.
    remaining = session.cache.store.count()
    session.cache.close()
    out(STYLE.faint(f"  saved · {remaining} answers cached"))
    return 0


def _erase_line() -> str:
    """Clear the live status line. Uses the ANSI erase when the terminal supports
    it; padding with spaces left a visible row of blanks in piped output."""
    return "\r\033[2K" if STYLE.enabled else "\r" + " " * 70 + "\r"


def _answer(session: ChatSession, prompt: str) -> None:
    """Print a live status line, stream the answer under it, then finalise."""
    started = time.perf_counter()
    state = {"started": False}

    def on_chunk(piece: str) -> None:
        if not state["started"]:
            # Replace the live line with the answer marker, so the answer reads
            # as a distinct block rather than continuing the status text.
            sys.stdout.write(_erase_line() + "  " + answer_marker(STYLE))
            state["started"] = True
        sys.stdout.write(piece)
        sys.stdout.flush()

    sys.stdout.write("  " + STYLE.faint("thinking..."))
    sys.stdout.flush()
    turn = session.ask(prompt, on_chunk=on_chunk)
    if not state["started"]:
        sys.stdout.write(_erase_line())
    del started

    out("")
    out(status_line(turn))


# ------------------------------------------------------------------ subcommands


def setup(cfg: Config, args) -> tuple[str, str, str]:
    """Resolve key, provider and model once, then remember them.

    Shared by every subcommand. Previously only `chat` persisted the key, so
    every single `ask` re-prompted for it -- and `ask` exited with "pass
    --model" instead of just asking, even though `chat` asked.
    """
    key, provider = resolve_key(cfg, args.api_key)
    model = choose_model(cfg, provider)
    if not cfg.offline:
        offer_to_save(cfg, key, provider, model)
    return key, provider, model


def cmd_chat(cfg: Config, args) -> int:
    key, provider, model = setup(cfg, args)
    session = build_session(cfg, key, provider, model)
    return run_repl(cfg, session)


def cmd_ask(cfg: Config, args) -> int:
    """One-shot: answer a single prompt and exit."""
    key, provider, model = setup(cfg, args)
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
    common.add_argument(
        "--provider",
        choices=all_provider_names() + ["stub"],
        help="anthropic/openai/gemini, or an OpenAI-compatible host "
        "(deepseek, kimi, glm, nvidia, groq, openrouter, together, ollama, "
        "lmstudio, custom)",
    )
    common.add_argument(
        "--base-url",
        dest="base_url",
        help="endpoint for an OpenAI-compatible host; overrides the preset",
    )
    common.add_argument("--model")
    common.add_argument("--embedder", choices=["auto", "local", "api", "hash"])
    common.add_argument("--threshold", type=float, help="cosine score needed to reuse (0.90)")
    common.add_argument("--max-entries", type=int, dest="max_entries")
    common.add_argument("--ttl-seconds", type=float, dest="ttl_seconds")
    common.add_argument("--scope", choices=["global", "session"])
    common.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    common.add_argument("--home", help="cache directory (default ~/.semcache)")
    common.add_argument("--project", help="isolate this project's cache from others")
    common.add_argument(
        "--offline",
        action="store_true",
        default=None,
        help="use the built-in stub model: no key, no network, no cost",
    )
    common.add_argument(
        "--no-log-prompts",
        action="store_false",
        dest="log_prompts",
        default=None,
        help="record only hashes, not prompt text",
    )
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
    bench.add_argument(
        "--assert-targets",
        action="store_true",
        help="exit non-zero if a latency target or trap pair regresses",
    )
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
    "provider",
    "model",
    "embedder",
    "threshold",
    "max_entries",
    "ttl_seconds",
    "scope",
    "effort",
    "home",
    "project",
    "offline",
    "log_prompts",
    "debug",
)


SUBCOMMANDS = ("chat", "ask", "stats", "clear", "bench")
_PASSTHROUGH = ("-h", "--help", "--version")


def with_default_command(argv: list[str]) -> list[str]:
    """Make `chat` the default subcommand.

    Injecting it *before* parsing matters: the top-level parser does not know
    the shared flags, so `semcache --threshold 0.90` made argparse read "0.90"
    as the subcommand and fail outright. Reparsing afterwards was too late.
    """
    for token in argv:
        if token in _PASSTHROUGH:
            return argv
        if not token.startswith("-"):
            # First bare word decides: a known subcommand, or an argument that
            # belongs to an implicit `chat`.
            return argv if token in SUBCOMMANDS else ["chat", *argv]
    return ["chat", *argv]  # nothing but flags, or nothing at all


def main(argv: list[str] | None = None) -> int:
    _setup_console()
    parser = build_parser()
    args = parser.parse_args(with_default_command(list(argv if argv is not None else sys.argv[1:])))
    command = args.command or "chat"

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
