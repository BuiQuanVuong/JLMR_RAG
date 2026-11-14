"""Paper page (/paper/<vNN>/<id>): header with actions, abstract, section outline (click -> PDF at that page),
BibTeX and similar papers.

Everything except "Similar papers" comes from the SQLite DB, so the page renders before the index has loaded;
the similar-papers panel shows placeholders until state.ready().
"""
from urllib.parse import quote

from nicegui import run, ui

from gui import search_data as sd
from gui import state
from gui.pdfview import open_external, show_pdf
from gui.search_cards import add_css, compact_card, copy_bibtex, open_web, topic_chip

N_SIMILAR = 8


def _similar(ix, key):
    try:
        return ix.similar(key, N_SIMILAR)
    except SystemExit:             # paper not in the embeddings
        return None


def _not_found(key):
    with ui.card().classes("w-full items-center text-center py-14 gap-3 rounded-2xl border border-gray-200 "
                           "shadow-none"):
        ui.icon("find_in_page", color="grey-5").classes("text-6xl")
        ui.label("Paper not found").classes("text-xl font-semibold text-gray-800")
        ui.label(f"There is no paper with the key '{key}' in the index. Keys look like v13/gretton12a or "
                 "v20/13-124.").classes("text-sm text-gray-500")
        pid = key.split("/")[-1]
        with ui.row().classes("gap-2 mt-2"):
            ui.button(f'Search for "{pid}"', icon="search",
                      on_click=lambda: ui.navigate.to(f"/search?q={quote(pid)}")).props("unelevated rounded no-caps")
            ui.button("Back", icon="arrow_back", on_click=ui.navigate.back).props("flat rounded no-caps")


def page(key):
    add_css()
    p = sd.paper_row(key)
    if p is None:
        _not_found(key)
        return
    t = sd.topics()

    ui.button("Back", icon="arrow_back", on_click=ui.navigate.back) \
        .props("flat dense no-caps color=grey-8").classes("-mb-2 self-start")

    # ------------------------------------------------------------ header
    with ui.card().classes("w-full p-6 gap-3 rounded-2xl border border-gray-200 shadow-none"):
        with ui.row().classes("items-center gap-2 text-xs text-gray-500"):
            ui.label("Journal of Machine Learning Research").classes("uppercase tracking-wider font-medium")
            ui.label("·")
            ui.label(f"Volume {p['volume']}" + (f" · {p['year']}" if p["year"] else ""))
            if p["n_pages"]:
                ui.label(f"· {p['n_pages']} pages")
        ui.label(p["title"]).classes("text-2xl font-bold text-gray-900 leading-tight")
        ui.label(p["authors"] or "").classes("text-base text-gray-700")
        with ui.row().classes("items-center gap-2"):
            if p["citation"]:
                ui.label(f"JMLR {p['volume']}{p['citation']}").classes("text-sm text-gray-500")
            ui.label(key).classes("text-xs font-mono text-gray-500 bg-gray-100 rounded px-2 py-0.5") \
                .tooltip("Paper key (use it with ask.py cite / similar)")
        if t.primary.get(key):
            with ui.row().classes("items-center gap-2"):
                ui.label("Topic").classes("text-xs uppercase tracking-wider text-gray-400 w-12")
                with ui.link(target="/topics").classes("no-underline"):
                    topic_chip(key)
                if t.secondary.get(key):
                    with ui.link(target="/topics").classes("no-underline"):
                        topic_chip(key, secondary=True)
                area = t.parent.get(t.primary[key])
                if area:
                    ui.label(f"area {area}: {t.names.get(area, '')}").classes("text-xs text-gray-500")
        with ui.row().classes("items-center gap-2 pt-2"):
            ui.button("Open PDF", icon="picture_as_pdf", on_click=lambda: show_pdf(key, 1)) \
                .props("unelevated rounded no-caps")
            ui.button("Open in PDF reader", icon="open_in_new", on_click=lambda: open_external(key)) \
                .props("outline rounded no-caps")
            ui.button("Copy BibTeX", icon="content_copy", on_click=lambda: copy_bibtex(key)) \
                .props("outline rounded no-caps")
            if p["abs_url"]:
                ui.button("JMLR page", icon="public", on_click=lambda: open_web(p["abs_url"])) \
                    .props("flat rounded no-caps").tooltip(p["abs_url"])

    # ------------------------------------------------------------ body
    with ui.element("div").classes("w-full grid grid-cols-1 lg:grid-cols-3 gap-4 items-start"):
        with ui.column().classes("lg:col-span-2 gap-4 min-w-0"):
            with ui.card().classes("w-full p-6 gap-2 rounded-2xl border border-gray-200 shadow-none"):
                _heading("notes", "Abstract")
                ui.label(p["abstract"] or "No abstract in the catalog.") \
                    .classes("text-[0.95rem] text-gray-800 leading-relaxed whitespace-pre-line")
            with ui.card().classes("w-full p-6 gap-3 rounded-2xl border border-gray-200 shadow-none"):
                _heading("auto_awesome_motion", "Similar papers")
                similar_box = ui.element("div").classes("w-full grid grid-cols-1 md:grid-cols-2 gap-3")
            with ui.card().classes("w-full p-0 gap-0 rounded-2xl border border-gray-200 shadow-none"):
                with ui.expansion("BibTeX", icon="format_quote").classes("w-full"):
                    ui.code(sd.bibtex(p), language="bibtex").classes("w-full text-xs")
        with ui.column().classes("gap-4 min-w-0 lg:sticky lg:top-20"):
            _outline_card(key)

    # ------------------------------------------------------------ similar papers (needs the index)
    def placeholders(note):
        similar_box.clear()
        with similar_box:
            for _ in range(4):
                with ui.card().classes("w-full p-3 gap-2 border border-gray-200 shadow-none rounded-xl"):
                    ui.skeleton("text").classes("w-5/6")
                    ui.skeleton("text").classes("w-1/2 text-xs")
            ui.label(note).classes("text-xs text-gray-500 md:col-span-2")

    loaded = {"started": False}

    async def load_similar():
        if state.STATE["error"]:
            timer.cancel()
            similar_box.clear()
            with similar_box:
                ui.label(f"The index failed to load: {state.STATE['error']}").classes("text-sm text-red-700")
            return
        if not state.ready() or loaded["started"]:
            return
        loaded["started"] = True
        timer.cancel()
        rows = await run.io_bound(_similar, state.index(), key)
        similar_box.clear()
        with similar_box:
            if not rows:
                ui.label("This paper has no embedding, so no similar papers can be computed.") \
                    .classes("text-sm text-gray-500")
            for other, score in rows or []:
                compact_card(other, score)

    placeholders("Loading the search index (about 25 s at app start)..." if not state.ready() else "Finding...")
    timer = ui.timer(0.5, load_similar)
    ui.timer(0.05, load_similar, once=True)


def _heading(icon, text, extra=None):
    with ui.row().classes("items-center gap-2"):
        ui.icon(icon, color="primary").classes("text-xl")
        ui.label(text).classes("text-base font-semibold text-gray-900")
        if extra:
            ui.label(extra).classes("text-xs text-gray-500")


def _outline_card(key):
    items = sd.outline(key)
    with ui.card().classes("w-full p-5 gap-2 rounded-2xl border border-gray-200 shadow-none"):
        _heading("toc", "Sections", "click to open the PDF at that page")
        if not items:
            ui.label("No section outline for this paper.").classes("text-sm text-gray-500")
            return
        with ui.scroll_area().classes("w-full h-[62vh] -mx-1"), ui.column().classes("w-full gap-0"):
            for depth, label, page, is_ref in items:
                with ui.row().classes("sr-outline-item w-full items-baseline justify-between no-wrap gap-2 py-1 pr-2 "
                                      + ("text-gray-500 " if is_ref else "text-gray-800 ")
                                      + ("font-medium" if depth == 0 else "text-[0.82rem]")) \
                        .style(f"padding-left: {6 + 14 * depth}px") as row:
                    ui.label(label).classes("text-sm leading-snug min-w-0" if depth == 0 else "leading-snug min-w-0")
                    ui.label(str(page)).classes("text-xs text-gray-400 font-mono shrink-0")
                row.on("click", lambda pg=page: show_pdf(key, pg))
