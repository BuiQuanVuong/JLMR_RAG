"""The common page frame: header (title, status, model switch), navigation drawer, content area.

Every page is built as:
    with frame("Search", "search"):
        ...page content...
"""
from contextlib import contextmanager

from nicegui import ui

from gui import state

NAV = [  # (key, label, path, icon)
    ("chat", "Chat", "/chat", "chat"),
    ("search", "Search", "/search", "search"),
    ("topics", "Topic map", "/topics", "hub"),
    ("advances", "Recent advances", "/advances", "trending_up"),
    ("sessions", "Sessions", "/sessions", "history"),
]

PRIMARY = "#3b5bdb"


@contextmanager
def frame(title, active):
    ui.colors(primary=PRIMARY)
    ui.query("body").style("background-color: #f7f8fa")
    with ui.header(elevated=False).classes("items-center justify-between bg-white text-gray-900 border-b px-4"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("auto_stories", color="primary").classes("text-2xl")
            ui.label("JMLR research assistant").classes("text-lg font-semibold")
            ui.label(f"/ {title}").classes("text-gray-500")
        with ui.row().classes("items-center gap-3"):
            status = ui.label(state.status_text()).classes("text-sm text-gray-600")
            ui.timer(1.0, lambda: status.set_text(state.status_text()))
            model = ui.toggle({"haiku": "Haiku", "opus": "Opus"}, value=state.STATE["model"],
                              on_change=lambda e: state.STATE.update(model=e.value)).props("dense no-caps")
            model.tooltip("Model for answers: Haiku (about $0.002 each) or Opus (a few cents)")
    with ui.left_drawer(value=True, bordered=True).classes("bg-white").props("width=210"):
        for key, label, path, icon in NAV:
            selected = key == active
            with ui.link(target=path).classes("no-underline w-full"):
                with ui.row().classes(
                        "items-center gap-3 px-3 py-2 rounded-lg w-full "
                        + ("bg-indigo-50 text-indigo-800 font-medium" if selected else "text-gray-700 hover:bg-gray-100")):
                    ui.icon(icon)
                    ui.label(label)
        ui.separator().classes("my-3")
        ui.label("4,138 JMLR papers, 2000-2025").classes("text-xs text-gray-500 px-3")
    with ui.column().classes("w-full max-w-[1400px] mx-auto p-4 gap-4"):
        yield
