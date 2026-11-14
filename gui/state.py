"""Shared state for the desktop app (single user, one process).

The index (embeddings, search model, reranker) is loaded ONCE in a background thread when the app starts;
pages check `ready()` and use `index()`. Long calls (retrieval, Claude) must run off the UI loop:
`await nicegui.run.io_bound(fn, ...)`.
"""
import threading

import ui as term_ui                 # the terminal UI module: its SESSION dict tracks API spend
from conversation import Conversation

STATE = {
    "index": None,          # ask.Index once loaded
    "error": None,          # load error message, if any
    "model": "haiku",       # "haiku" | "opus" for answers
    "no_rewrite": False,
    "conv": Conversation(), # the current chat conversation (saved to data/sessions after each answer)
}


def start_loading(no_rewrite=False):
    """Load the index in the background (about 25 s with the GPU)."""
    STATE["no_rewrite"] = no_rewrite

    def load():
        try:
            import ask
            STATE["index"] = ask.Index(rewrite="off") if no_rewrite else ask.Index()
        except BaseException as e:            # includes sys.exit from Index()
            STATE["error"] = f"{type(e).__name__}: {e}"

    threading.Thread(target=load, name="index-loader", daemon=True).start()


def ready():
    return STATE["index"] is not None


def index():
    return STATE["index"]


def spent():
    """Approximate API spend in this app session, in USD."""
    return term_ui.SESSION["cost"]


def status_text():
    if STATE["error"]:
        return f"index failed to load: {STATE['error']}"
    if not ready():
        return "loading the index (about 25 s)..."
    n = len(STATE["conv"])
    return f"{'Opus 5.5' if STATE['model'] == 'opus' else 'Haiku 5.5'} | memory {n} turn{'' if n == 1 else 's'} | session ~${spent():.4f}"
