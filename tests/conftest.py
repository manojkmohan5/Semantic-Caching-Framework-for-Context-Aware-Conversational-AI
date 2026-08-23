"""Test isolation from the ambient environment.

Config deliberately reads SEMCACHE_* env vars and provider keys, which is right
for the app and wrong for the suite: whatever is exported in the shell would
silently change what the tests exercise. CI set SEMCACHE_OFFLINE=1 workflow-wide
as a safety measure and every test that asked for a real provider got the stub
instead -- green locally, red on the runner.

Clearing them per test also means a developer's real API key can never be picked
up by a test run.
"""

from __future__ import annotations

import os

import pytest

#: Provider keys Config and the CLI look for on their own.
_KEY_VARS = (
    "SEMCACHE_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch, tmp_path):
    """Strip every ambient variable that could steer a test."""
    for name in list(os.environ):
        if name.startswith("SEMCACHE_") or name in _KEY_VARS:
            monkeypatch.delenv(name, raising=False)

    # Point the cache somewhere disposable so a test that forgets to pass
    # home= cannot touch the developer's real ~/.semcache.
    monkeypatch.setenv("SEMCACHE_HOME", str(tmp_path / "home"))
    # Keep the fastembed/HF download bars out of test output.
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
