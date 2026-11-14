"""Sessions - the saved conversations (data/sessions, one JSON file per chat, saved after every answer).
Each card shows the title, date, number of turns and the questions asked; "Resume" makes it the current chat."""
from datetime import datetime

from nicegui import ui

import conversation
from gui import state


def _when(iso):
    try:
        d = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return iso or ""
    today = datetime.now().date()
    if d.date() == today:
        return f"Today {d:%H:%M}"
    if (today - d.date()).days == 1:
        return f"Yesterday {d:%H:%M}"
    return f"{d:%d %b %Y, %H:%M}"


def load_sessions():
    """Saved sessions, newest first, each with its questions (for the preview and the search)."""
    out = []
    for s in conversation.list_sessions(limit=10000):
        try:
            conv = conversation.Conversation.load(s["id"])
            questions = [t.question for t in conv.turns]
        except Exception:
            questions = []
        out.append({**s, "questions": questions})
    return out


def matches(s, needle):
    needle = needle.strip().lower()
    return not needle or needle in s["title"].lower() or any(needle in q.lower() for q in s["questions"])


def resume(session_id):
    try:
        state.STATE["conv"] = conversation.Conversation.load(session_id)
    except Exception as e:
        ui.notify(f"Could not open the session: {type(e).__name__}: {e}", type="negative")
        return
    ui.navigate.to("/chat")


def new_chat():
    state.STATE["conv"] = conversation.Conversation()
    ui.navigate.to("/chat")


def page():
    sessions = load_sessions()
    current = state.STATE["conv"].session_id

    with ui.column().classes("w-full max-w-4xl mx-auto gap-4"):
        with ui.row().classes("w-full items-end justify-between"):
            with ui.column().classes("gap-0"):
                ui.label("Saved conversations").classes("text-2xl font-semibold text-gray-800")
                ui.label(f"{len(sessions)} conversation{'' if len(sessions) == 1 else 's'} · saved automatically "
                         "after every answer").classes("text-sm text-gray-500")
            ui.button("New chat", icon="add_comment", on_click=new_chat).props("unelevated no-caps color=primary")
        search = ui.input(placeholder="Filter by title or question...").props(
            "outlined dense clearable rounded bg-color=white").classes("w-full")
        with search.add_slot("prepend"):
            ui.icon("search")

        @ui.refreshable
        def cards():
            shown = [s for s in sessions if matches(s, search.value or "")]
            if not sessions:
                with ui.card().classes("w-full items-center p-8 gap-2"):
                    ui.icon("forum", color="grey-5").classes("text-5xl")
                    ui.label("No saved conversations yet").classes("text-lg font-medium")
                    ui.label("Ask a question on the Chat page; each conversation is saved here.").classes(
                        "text-gray-500")
                    ui.button("Start a chat", icon="chat", on_click=new_chat).props("flat no-caps")
                return
            if not shown:
                ui.label(f"No conversation matches “{search.value}”.").classes("text-gray-500 py-4")
                return
            for s in shown:
                is_current = s["id"] == current
                with ui.card().classes("w-full p-4 gap-2 hover:shadow-md transition-shadow" +
                                       (" border-l-4 border-indigo-500" if is_current else "")):
                    with ui.row().classes("w-full no-wrap items-start justify-between gap-4"):
                        with ui.column().classes("gap-1 min-w-0 grow"):
                            ui.label(s["title"] or "(untitled)").classes(
                                "text-base font-medium text-gray-900 w-full truncate").tooltip(s["title"])
                            with ui.row().classes("items-center gap-3 text-xs text-gray-500"):
                                with ui.row().classes("items-center gap-1"):
                                    ui.icon("schedule").classes("text-sm")
                                    ui.label(_when(s["updated"]))
                                ui.badge(f"{s['turns']} turn{'' if s['turns'] == 1 else 's'}",
                                         color="indigo-1", text_color="indigo-9").props("rounded")
                                if is_current:
                                    ui.badge("current chat", color="green-1", text_color="green-9").props("rounded")
                        ui.button("Resume", icon="play_arrow", on_click=lambda sid=s["id"]: resume(sid)).props(
                            "unelevated no-caps color=primary")
                    if s["questions"]:
                        with ui.column().classes("w-full gap-0.5 pl-1 border-l-2 border-gray-100"):
                            for q in s["questions"][:4]:
                                ui.label(q).classes("text-sm text-gray-600 truncate w-full pl-2")
                            if len(s["questions"]) > 4:
                                ui.label(f"+ {len(s['questions']) - 4} more").classes(
                                    "text-xs text-gray-400 pl-2")

        cards()
        search.on_value_change(cards.refresh)
