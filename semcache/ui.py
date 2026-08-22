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
_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "grey": "90",
}

# Box drawing, kept to characters that render in a default Windows console.
BAR = "┃"  # heavy vertical: the accent bar beside a question
BLOCK = "▌"  # left half block: the input marker
LINE = "─"
DOT = "·"


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


class Style:
    """Wraps text in escapes, or returns it untouched when colour is off."""

    def __init__(self, enabled: bool | None = None):
        self.enabled = supports_colour() if enabled is None else enabled

    def __call__(self, text: str, *names: str) -> str:
        if not self.enabled or not names:
            return text
        codes = ";".join(_CODES[n] for n in names if n in _CODES)
        return f"\033[{codes}m{text}{RESET}" if codes else text

    # Named helpers, so call sites read as intent rather than as colour.
    def hit(self, text: str) -> str:
        return self(text, "green")

    def miss(self, text: str) -> str:
        return self(text, "yellow")

    def warn(self, text: str) -> str:
        return self(text, "magenta")

    def error(self, text: str) -> str:
        return self(text, "red")

    def faint(self, text: str) -> str:
        return self(text, "grey")

    def strong(self, text: str) -> str:
        return self(text, "bold")

    def brand(self, text: str) -> str:
        return self(text, "bold", "cyan")


def width(maximum: int = 92) -> int:
    """Usable width, capped so long lines stay readable on a wide monitor."""
    try:
        cols = shutil.get_terminal_size((80, 24)).columns
    except Exception:
        cols = 80
    return max(40, min(cols - 4, maximum))


def columns(style: Style, panels: list, gutter: int = 3) -> list:
    """Lay labelled panels side by side, wrapping to one panel per row if the
    terminal is too narrow to hold them."""
    total = width()
    count = max(1, len(panels))
    each = (total - gutter * (count - 1)) // count
    if each < 22:  # too cramped to be readable side by side
        rows = []
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
            # Pad on the visible length, not the styled length -- escape codes
            # have no width but do count in len().
            cells.append(cell + " " * max(0, each - _visible_len(cell)))
        rows.append("  " + (" " * gutter).join(cells))
    return rows


def _visible_len(text: str) -> int:
    """Length ignoring ANSI escapes, so padded columns actually line up."""
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


def dashboard(style: Style, version: str, panels: list, commands: list) -> str:
    """The screen you land on: what is loaded, what is cached, what it has saved,
    and the commands -- all inline, no alternate screen."""
    rule = style.faint(LINE * width())
    tagline = style.faint("semantic cache in front of your LLM")
    lines = [
        "",
        f"  {style.brand('semcache')} {style.faint('v' + version)}"
        f"   {style.faint(DOT)}   {tagline}",
        f"  {rule}",
        "",
    ]
    lines += columns(style, panels)
    lines += ["", f"  {style.faint('COMMANDS')}"]
    lines += [f"  {style.strong(cmd.ljust(8))} {style.faint(what)}" for cmd, what in commands]
    lines += ["", f"  {style.faint('Type a question and press Enter.')}", ""]
    return "\n".join(lines)


def prompt(style: Style) -> str:
    """Input marker. Short, so a pasted question has room on the line."""
    return style(BLOCK, "cyan") + " "


def redraw_last_line() -> str:
    """Move up one line and clear it.

    The terminal already echoed what the user typed, so re-rendering the question
    as a framed block means erasing that echo first.
    """
    return "\033[1A\033[2K"


def user_block(style: Style, text: str) -> str:
    """The question, re-rendered with a left accent bar."""
    bar = style(BAR, "cyan")
    lines = wrap(text, width() - 4)
    return "\n".join([""] + [f"  {bar} {style.strong(line)}" for line in lines])


def footer(style: Style, provider: str, model: str, entries: int, hit_rate: float) -> str:
    """Dim strip under each answer: what is loaded, and how it is doing."""
    left = f"{provider} {DOT} {model}"
    right = f"{entries} cached {DOT} {hit_rate * 100:.0f}% hit rate"
    gap = max(1, width() - len(left) - len(right))
    return "  " + style.faint(left + " " * gap + right)


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
