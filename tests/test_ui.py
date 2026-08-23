"""Presentation rules that matter: never colour a pipe, always line up columns."""

from __future__ import annotations

from semcache.ui import (
    Style,
    _visible_len,
    columns,
    header,
    supports_256,
    user_row,
    wrap,
)


def test_colour_is_off_when_output_is_not_a_terminal(monkeypatch):
    """An escape code in a log file or a pipe is worse than no colour."""
    style = Style(enabled=False)
    assert style("text", "accent") == "text"
    assert style.hit("ok") == "ok"
    assert "\033" not in header(style, "1.0", "stub", "m", "/tmp", 3, 0.5)


def test_no_color_env_disables_colour(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    from semcache.ui import supports_colour

    assert supports_colour() is False


def test_basic_palette_falls_back_to_eight_colours(monkeypatch):
    monkeypatch.setenv("SEMCACHE_BASIC_COLOR", "1")
    assert supports_256() is False
    basic = Style(enabled=True, rich=False)
    # 8-colour codes only: no "38;5;" indexed sequences.
    assert "38;5;" not in basic.accent("x")
    assert "\033[35m" in basic.accent("x")

    rich = Style(enabled=True, rich=True)
    assert "38;5;141" in rich.accent("x")


def test_visible_length_ignores_escape_codes():
    """Padding must measure screen width, not string length -- escape codes take
    no columns but do count in len(), which misaligned every styled column."""
    style = Style(enabled=True, rich=True)
    painted = style.accent("hello")
    assert len(painted) > 5
    assert _visible_len(painted) == 5
    assert _visible_len("plain") == 5


def test_columns_line_up_even_when_styled():
    style = Style(enabled=True, rich=True)
    panels = [
        ("model", [style.strong("openrouter"), style.faint("glm-5.2")]),
        ("cache", [style.hit("9 answers"), style.faint("28 KB")]),
    ]
    rows = columns(style, panels)
    widths = {_visible_len(r) for r in rows}
    assert len(widths) == 1, f"rows must share one visible width, got {widths}"


def test_narrow_terminal_stacks_panels_instead_of_crushing_them(monkeypatch):
    monkeypatch.setattr("semcache.ui.width", lambda maximum=92: 40)
    style = Style(enabled=False)
    rows = columns(style, [("a", ["1"]), ("b", ["2"]), ("c", ["3"])])
    assert any("A" in r for r in rows) and any("C" in r for r in rows)


def test_user_row_stays_plain_without_colour():
    plain = user_row(Style(enabled=False), "a question")
    assert plain.strip() == "> a question"
    assert "\033" not in plain


def test_wrap_never_loses_a_word():
    text = "the quick brown fox jumps over the lazy dog"
    for limit in (10, 20, 80):
        assert " ".join(wrap(text, limit)).split() == text.split()
    assert wrap("", 10) == [""]
