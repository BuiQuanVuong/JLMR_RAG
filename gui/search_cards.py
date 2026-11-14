"""UI pieces shared by the Search and Paper pages: topic chips, BibTeX copy, paper result cards, page CSS."""
import webbrowser

from nicegui import ui

from gui import search_data as sd
from gui.pdfview import show_pdf

CSS = """
<style>
.sr-card { transition: box-shadow .15s ease, border-color .15s ease; }
.sr-card:hover { box-shadow: 0 4px 18px rgba(59,91,219,.10); border-color: #c5d0fa; }
.sr-title { color: #1e293b; text-decoration: none; }
.sr-title:hover { color: #3b5bdb; text-decoration: underline; }
.sr-snippet mark { background: #fff3bf; color: inherit; padding: 0 1px; border-radius: 2px; }
.sr-kbd { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 0.78rem;
          background: #eef1fd; color: #2f46b0; padding: 1px 6px; border-radius: 5px; white-space: nowrap; }
.sr-outline-item { cursor: pointer; border-radius: 6px; }
.sr-outline-item:hover { background: #eef1fd; color: #2f46b0; }
</style>
"""


def add_css():
    ui.add_head_html(CSS)


def copy_bibtex(key):
    entry = sd.bibtex(key)
    if not entry:
        ui.notify(f"No catalog entry for {key}", type="warning")
        return
    ui.clipboard.write(entry)
    ui.notify("BibTeX copied to the clipboard", type="positive", icon="content_copy", position="bottom-right")


def open_web(url):
    """Open a URL in the system browser (the app runs locally, so the server's machine is the user's)."""
    if url:
        webbrowser.open(url)
        ui.notify("Opened in your web browser", position="bottom-right")


def topic_chip(key, on_area=None, secondary=False):
    """Chip with the paper's primary (or secondary) subtopic; clicking it calls on_area(area id) if given."""
    t = sd.topics()
    tid = (t.secondary if secondary else t.primary).get(key)
    if not tid:
        return None
    area = t.parent.get(tid)
    chip = ui.chip(t.label(tid), icon="hub" if not secondary else "call_split",
                   color="indigo-1" if not secondary else "grey-2",
                   text_color="indigo-9" if not secondary else "grey-8").props("dense square").classes("text-xs m-0")
    tip = f"{'Secondary' if secondary else 'Primary'} subtopic, in area {area} {t.names.get(area, '')}"
    if on_area and area:
        chip.on("click", lambda: on_area(area))
        tip += " - click to filter by this area"
    chip.tooltip(tip)
    return chip


def result_card(rank, paper, hit, missing, query, on_area=None):
    key = paper["key"]
    with ui.card().classes("sr-card w-full p-4 gap-2 border border-gray-200 shadow-none rounded-xl"):
        with ui.row().classes("w-full items-start no-wrap gap-3"):
            ui.label(str(rank)).classes(
                "shrink-0 w-7 h-7 rounded-full bg-indigo-50 text-indigo-700 text-sm font-semibold "
                "flex items-center justify-center mt-0.5")
            with ui.column().classes("grow gap-1 min-w-0"):
                with ui.row().classes("w-full items-start justify-between no-wrap gap-3"):
                    ui.link(paper["title"], f"/paper/{key}").classes("sr-title text-[1.05rem] font-semibold leading-snug")
                    ui.badge(str(paper["year"] or ""), color="grey-2", text_color="grey-9").classes(
                        "shrink-0 text-xs px-2 py-1")
                ui.label(sd.short_authors(paper["authors"], 6)).classes("text-sm text-gray-600")
                with ui.row().classes("items-center gap-2 mt-1"):
                    topic_chip(key, on_area)
                    ui.label(key).classes("text-xs text-gray-400 font-mono")
                    if missing:
                        with ui.row().classes("items-center gap-1 text-xs text-amber-800 bg-amber-50 "
                                              "border border-amber-200 rounded-md px-2 py-0.5"):
                            ui.icon("rule", size="14px")
                            ui.label("missing: " + ", ".join(missing))
                if hit and (hit.get("text") or "").strip():
                    with ui.column().classes("w-full gap-1 mt-2 pl-3 border-l-[3px] border-indigo-200"):
                        ui.label(sd.where(hit)).classes("text-xs font-medium text-indigo-700")
                        ui.html(sd.snippet_html(hit["text"], query), sanitize=False).classes(
                            "sr-snippet text-sm text-gray-700 leading-relaxed")
                elif paper.get("abstract"):
                    with ui.column().classes("w-full gap-1 mt-2 pl-3 border-l-[3px] border-gray-200"):
                        ui.label("Abstract").classes("text-xs font-medium text-gray-500")
                        ui.html(sd.snippet_html(paper["abstract"], query, 260), sanitize=False).classes(
                            "sr-snippet text-sm text-gray-700 leading-relaxed")
                with ui.row().classes("items-center gap-1 mt-1 -ml-2"):
                    page = (hit or {}).get("page_start")
                    ui.button(f"Open PDF at p. {page}" if page else "Open PDF", icon="picture_as_pdf",
                              on_click=lambda k=key, pg=page: show_pdf(k, pg)).props("flat dense no-caps color=primary")
                    ui.button("BibTeX", icon="content_copy",
                              on_click=lambda k=key: copy_bibtex(k)).props("flat dense no-caps color=grey-8")
                    ui.button("Paper page", icon="article",
                              on_click=lambda k=key: ui.navigate.to(f"/paper/{k}")).props("flat dense no-caps color=grey-8")


def compact_card(paper, score=None):
    """Small card linking to a paper page (similar papers)."""
    key = paper["key"]
    with ui.link(target=f"/paper/{key}").classes("no-underline text-inherit block w-full"):
        with ui.card().classes("sr-card w-full h-full p-3 gap-1 border border-gray-200 shadow-none rounded-xl"):
            with ui.row().classes("w-full items-start justify-between no-wrap gap-2"):
                ui.label(paper["title"]).classes("text-sm font-semibold text-gray-900 leading-snug")
                if score is not None:
                    ui.label(f"{score:.2f}").classes("shrink-0 text-xs text-indigo-700 bg-indigo-50 rounded px-1.5 "
                                                     "py-0.5 font-mono").tooltip("Cosine similarity of the papers")
            ui.label(f"{sd.short_authors(paper['authors'], 3)} · {paper['year']}").classes("text-xs text-gray-600")
            t = sd.topics()
            tid = t.primary.get(key)
            if tid:
                ui.label(t.label(tid)).classes("text-xs text-indigo-800")
