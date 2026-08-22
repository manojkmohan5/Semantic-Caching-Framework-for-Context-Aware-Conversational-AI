"""Using semcache from your own Python code.

Run it with no key at all (uses the built-in stub model, costs nothing):

    python examples/use_the_api.py --offline

Or against a real provider:

    set OPENROUTER_KEY=sk-or-v1-...            # Windows
    python examples/use_the_api.py --key %OPENROUTER_KEY% ^
        --provider openrouter --model "liquid/lfm-2.5-2.6b:free"
"""

from __future__ import annotations

import argparse

from semcache import SemCache

# Two ways of asking the same thing, then one that is genuinely different.
QUESTIONS = [
    "What causes memory fragmentation in long-running Python processes?",
    "Why do long-lived Python services suffer from heap fragmentation?",
    "How do I add an index in Postgres without locking the table?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", help="provider API key (else read from the env)")
    parser.add_argument("--provider", help="anthropic | openai | gemini | grok | ...")
    parser.add_argument("--model")
    parser.add_argument("--offline", action="store_true", help="stub model, no cost")
    parser.add_argument(
        "--threshold",
        type=float,
        # semcache's own default is 0.95. These two phrasings score ~0.90 with
        # bge-small, so 0.88 is used here to actually demonstrate a semantic
        # hit. Lower means more reuse and more risk -- tune it with `bench`.
        default=0.88,
        help="cosine score needed to reuse an answer (semcache default: 0.95)",
    )
    args = parser.parse_args()

    # One SemCache per worker. The context manager flushes the index on exit.
    with SemCache(
        args.key,
        provider=args.provider,
        model=args.model,
        offline=args.offline,
        project="api-example",
        threshold=args.threshold,
    ) as cache:
        # Optional: pay the model-load cost now instead of on the first question.
        cache.warm()

        for question in QUESTIONS:
            answer = cache.ask(question)
            where = "cache" if answer.from_cache else "model"
            match = f" {answer.similarity * 100:.0f}% match" if answer.similarity else ""
            print(f"\nQ: {question}")
            print(
                f"   [{where}] {answer.latency_ms:.0f}ms | "
                f"{answer.input_tokens} in / {answer.output_tokens} out{match}"
            )
            print(f"   {answer.text.strip()[:160]}...")

        # Ask the first one again, word for word: the cheapest possible path.
        repeat = cache.ask(QUESTIONS[0])
        print(f"\nExact repeat: [{repeat.source}] {repeat.latency_ms:.1f}ms")

        print(cache.report())


if __name__ == "__main__":
    main()
