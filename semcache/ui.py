"""Terminal presentation. No dependencies -- ANSI escapes and a capability check.

Everything renders inline in the terminal you launched from: no alternate screen,
no separate window, and scrollback keeps working. That is deliberate. Taking over
the screen would mean re-implementing wrapping, scrolling and line editing, and it
would fight the streaming answer, which is the best part of the product.

Colour is opt-out three ways, because an escape code in a log file or a pipe is
worse than no colour at all: NO_COLOR, SEMCACHE_NO_COLOR, or stdout not being a
terminal.
"""

from __future__ import annotations

import os
import shutil
import sys

RESET = "\033[0m"

#: Palette. The named roles map to 256-colour codes for a violet accent on dim
#: greys; basic 8-colour codes are the fallback for terminals that only speak
#: those (forced with SEMCACHE_BASIC_COLOR=1).
_CODES_256 = {
    "bold": "1",
    "dim": "2",
    "accent": "38;5;141",  # violet, the brand colour
    "accent_dim": "38;5;98",  # muted violet, for rules and bars
    "good": "38;5;114",  # soft green, a cache hit
    "warn": "38;5;180",  # sand, a model call
    "alert": "38;5;168",  # rose, a degraded answer
    "bad": "38;5;203",  # red, an error
    "grey": "38;5;245",  # secondary text
    "faint": "38;5;240",  # tertiary text
}
_CODES_BASIC = {
    "bold": "1",
    "dim": "2",
    "accent": "35",
    "accent_dim": "35",
    "good": "32",
    "warn": "33",
    "alert": "35",
    "bad": "31",
    "grey": "90",
    "faint": "90",
}

# Box drawing, kept to characters that render in a default Windows console.
BAR = "┃"  # heavy vertical: accent bar
BLOCK = "▌"  # left half block: the input marker
LINE = "─"
DOT = "·"

#: Three-row mark built from block characters. Deliberately tiny: it survives a
#: narrow terminal and any monospace font, unlike a large ASCII-art logo.
LOGO = ("█▀█▀█", "█▀▀▀█", "▀▀ ▀▀")


def _enable_windows_ansi() -> bool:
    """cmd.exe needs virtual-terminal processing switched on explicitly.

    Windows Terminal and PowerShell 7 handle ANSI natively; the classic console
    host does not, and would print the raw escapes instead.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # -11 = STD_OUTPUT_HANDLE, 0x4 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x4))
    except Exception:
        return False


def supports_colour() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("SEMCACHE_NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        if not sys.stdout.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    return _enable_windows_ansi()


def supports_256() -> bool:
    """Whether to use the 256-colour palette.

    Practically every terminal since the 2000s handles 256 colours, so this
    defaults to yes and offers an escape hatch rather than sniffing TERM strings.
    """
    if os.environ.get("SEMCACHE_BASIC_COLOR"):
        return False
    term = os.environ.get("TERM", "")
    return term not in {"dumb", "vt100", "ansi"}


class Style:
    """Wraps text in escapes, or returns it untouched when colour is off."""

    def __init__(self, enabled: bool | None = None, rich: bool | None = None):
        self.enabled = supports_colour() if enabled is None else enabled
        self.palette = _CODES_256 if (supports_256() if rich is None else rich) else _CODES_BASIC

    def __call__(self, text: str, *names: str) -> str:
        if not self.enabled or not names:
            return text
        codes = ";".join(self.palette[n] for n in names if n in self.palette)
        return f"\033[{codes}m{text}{RESET}" if codes else text

    # Named helpers, so call sites read as intent rather than as a colour.
    def hit(self, text: str) -> str:
        return self(text, "good")

    def miss(self, text: str) -> str:
        return self(text, "warn")

    def warn(self, text: str) -> str:
        return self(text, "alert")

    def error(self, text: str) -> str:
        return self(text, "bad")

    def faint(self, text: str) -> str:
        return self(text, "faint")

    def dim(self, text: str) -> str:
        return self(text, "grey")

    def strong(self, text: str) -> str:
        return self(text, "bold")

    def accent(self, text: str) -> str:
        return self(text, "accent")

    def brand(self, text: str) -> str:
        return self(text, "bold", "accent")


def width(maximum: int = 92) -> int:
    """Usable width, capped so long lines stay readable on a wide monitor."""
    try:
        cols = shutil.get_terminal_size((80, 24)).columns
    except Exception:
        cols = 80
    return max(40, min(cols - 4, maximum))


def _visible_len(text: str) -> int:
    """Length ignoring ANSI escapes, so padded rows actually line up. Escape
    codes take no screen width but do count in len()."""
    out, i = 0, 0
    while i < len(text):
        if text[i] == "\033":
            end = text.find("m", i)
            if end == -1:
                break
            i = end + 1
            continue
        out += 1
        i += 1
    return out


def header(
    style: Style,
    version: str,
    provider: str,
    model: str,
    home: str,
    entries: int,
    hit_rate: float,
) -> str:
    """Identity block: mark, name, what is loaded, and where it lives."""
    mark = [style(row, "accent") for row in LOGO]
    facts = [
        f"{style.brand('semcache')} {style.faint('v' + version)}",
        f"{style.strong(provider)} {style.faint(DOT)} {model}",
        style.faint(home),
    ]
    lines = [""]
    for i in range(3):
        lines.append(f"  {mark[i]}   {facts[i]}")
    lines.append("")

    if entries:
        noun = "answer" if entries == 1 else "answers"
        tip = (
            f"{style.hit(str(entries))} {noun} cached "
            f"{style.faint(DOT)} {style.hit(f'{hit_rate * 100:.0f}%')} of questions "
            f"answered without an API call so far."
        )
    else:
        tip = style.faint(
            "Nothing cached yet. Ask something, then ask it again in different words to see a hit."
        )
    lines.append(f"  {style(BAR, 'accent')} {tip}")
    lines.append(
        f"  {style.faint('/help for commands')}  {style.faint(DOT)}  "
        f"{style.faint('/dash for the dashboard')}"
    )
    lines.append("")
    return "\n".join(lines)


def columns(style: Style, panels: list, gutter: int = 3) -> list:
    """Lay labelled panels side by side, collapsing to one per row when the
    terminal is too narrow to hold them."""
    total = width()
    count = max(1, len(panels))
    each = (total - gutter * (count - 1)) // count
    if each < 22:  # too cramped to read side by side
        rows: list = []
        for title, values in panels:
            rows.append("  " + style.faint(title.upper()))
            rows += [f"  {v}" for v in values]
            rows.append("")
        return rows[:-1]

    depth = max(len(values) for _, values in panels)
    rows = ["  " + (" " * gutter).join(style.faint(t.upper().ljust(each)) for t, _ in panels)]
    for i in range(depth):
        cells = []
        for _, values in panels:
            cell = values[i] if i < len(values) else ""
            cells.append(cell + " " * max(0, each - _visible_len(cell)))
        rows.append("  " + (" " * gutter).join(cells))
    return rows


def dashboard(style: Style, panels: list, commands: list) -> str:
    """The fuller view, on demand via /dash: panels plus the command guide."""
    lines = [""]
    lines += columns(style, panels)
    lines += ["", f"  {style.faint('COMMANDS')}"]
    lines += [f"  {style.strong(cmd.ljust(8))} {style.faint(what)}" for cmd, what in commands]
    lines.append("")
    return "\n".join(lines)


def user_row(style: Style, text: str) -> str:
    """The question, on a full-width reversed bar so it stands out as the turn
    boundary when scrolling back."""
    if not style.enabled:
        return f"\n> {text}"
    total = width()
    lines = wrap(text, total - 4)
    rendered = []
    for line in lines:
        padded = f" > {line}".ljust(total)
        rendered.append("\033[7m" + padded + RESET)
    return "\n" + "\n".join(rendered)


def answer_marker(style: Style) -> str:
    return style("●", "accent") + " "


def prompt(style: Style) -> str:
    """Input marker. Short, so a pasted question has room on the line."""
    return style(BLOCK, "accent") + " "


def redraw_last_line() -> str:
    """Move up one line and clear it.

    The terminal already echoed what the user typed, so re-rendering the question
    as a framed block means erasing that echo first.
    """
    return "\033[1A\033[2K"


def _compact(n: float) -> str:
    """1234 -> 1.2k. Keeps the status bar one line on a narrow terminal."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:.0f}"


def status_bar(
    style: Style,
    model: str,
    context_tokens: int,
    context_window: int | None,
    sent_tokens: int,
    entries: int,
    hit_rate: float,
    saved_usd: float,
) -> str:
    """The strip that sits just above the input line.

    Nothing can stay *below* the cursor in an inline REPL, so this is printed
    immediately before the prompt -- visually attached to where you type.

    "context" is what actually rides along on the next call: the retained
    conversation turns. It is the number that costs money on every miss, which
    is why it leads.
    """
    ctx = f"ctx {_compact(context_tokens)}"
    if context_window:
        ctx += f"/{_compact(context_window)} ({context_tokens / context_window * 100:.1f}%)"
    else:
        ctx += " tok"

    left = f"{model}  {DOT}  {ctx}  {DOT}  sent {_compact(sent_tokens)}"
    right = f"{entries} cached  {DOT}  {hit_rate * 100:.0f}% hit"
    if saved_usd:
        right += f"  {DOT}  saved ${saved_usd:.4f}"

    total = width()
    gap = total - len(left) - len(right)
    if gap < 2:  # too narrow for one line; drop the least useful half
        return "  " + style.faint(left)
    return "  " + style.faint(left) + " " * gap + style.faint(right)


def wrap(text: str, limit: int) -> list:
    """Greedy word wrap. Enough for prompts; answers stream unwrapped."""
    lines: list = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > limit:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]
