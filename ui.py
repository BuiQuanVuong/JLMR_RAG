"""Terminal UI helpers: Markdown rendering of streamed answers (Rich), a loading spinner, and a chat prompt
with tab completion, persistent history and a status bar (prompt_toolkit).

Everything falls back to plain text when output is piped/redirected or the libraries are missing, so
scripted use (`printf ... | python ask.py chat`, `> answer.txt`) keeps working unchanged.
"""
import contextlib
import sys

from common import DATA_DIR

HISTORY_FILE = DATA_DIR / "chat_history.txt"
SESSION = {"cost": 0.0}                     # API spend in this process (answers, condensing, comparisons)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    console = Console(highlight=False)
except ImportError:                          # plain mode
    console = None


def fancy():
    """Rich output only for an interactive terminal."""
    return console is not None and sys.stdout.isatty()


def add_cost(usd):
    SESSION["cost"] += usd or 0.0


def status(message):
    """Spinner while something slow runs (index/model loading)."""
    if fancy():
        return console.status(f"[cyan]{message}[/cyan]", spinner="dots")
    print(message, file=sys.stderr, flush=True)
    return contextlib.nullcontext()


def stream_answer(text_iter):
    """Show a streamed answer. In a terminal: a live Markdown preview of the latest lines while it streams,
    then the whole answer rendered once as Markdown (headings, tables, bold, code). Returns the text."""
    if not fancy():
        parts = []
        for t in text_iter:
            print(t, end="", flush=True)
            parts.append(t)
        print()
        return "".join(parts)
    buf = ""
    tail_lines = max(console.size.height - 4, 5)
    with Live(Markdown(""), console=console, refresh_per_second=8, transient=True,
              vertical_overflow="crop") as live:
        for t in text_iter:
            buf += t
            live.update(Markdown("\n".join(buf.splitlines()[-tail_lines:])))
    console.print(Markdown(buf))
    return buf


def note(text, style="dim"):
    """A secondary line (costs, retrieval info)."""
    if fancy():
        console.print(text, style=style, markup=False)
    else:
        print(text)


def panel(text, title):
    if fancy():
        console.print(Panel(text, title=title, border_style="cyan", expand=False))
    else:
        print(text)


class ChatPrompt:
    """prompt_toolkit input: Tab completes commands and paper keys, Up/Down browse history saved across
    sessions, and a bottom toolbar shows model, memory and session cost. Plain input() when piped."""

    def __init__(self, commands, paper_keys, toolbar):
        self.toolbar = toolbar
        self.session = None
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.history import FileHistory
        except ImportError:
            return
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.session = PromptSession(history=FileHistory(str(HISTORY_FILE)),
                                     completer=WordCompleter(list(commands) + list(paper_keys), WORD=True),
                                     complete_while_typing=False,
                                     bottom_toolbar=lambda: " " + self.toolbar())

    def ask(self, label):
        if self.session is None:
            return input(f"\n{label} > ")
        print()
        return self.session.prompt(f"{label} > ")
