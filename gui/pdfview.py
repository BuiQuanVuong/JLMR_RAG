"""Shared PDF viewer: open a paper's PDF inside the app, at a given page.

PDFs are served by the app from the JMLR folder at /pdfs/<vNN>/<id>.pdf (see gui/app.py); the browser's built-in
PDF viewer (Edge WebView2 in the desktop window) honours #page=N.

    from gui.pdfview import show_pdf, pdf_url
    show_pdf("v13/gretton12a", page=10)
"""
import os

from nicegui import ui

from common import JMLR_DIR, connect, open_path


def pdf_url(key, page=None):
    """URL of a paper's PDF as served by the app ('v13/gretton12a' -> '/pdfs/v13/gretton12a.pdf#page=10')."""
    return f"/pdfs/{key}.pdf" + (f"#page={page}" if page else "")


def paper_title(key):
    r = connect().execute("SELECT title, year FROM papers WHERE key=?", (key,)).fetchone()
    return f"{r['title']} ({r['year']})" if r else key


def open_external(key):
    path = JMLR_DIR / f"{key}.pdf"
    if path.exists():
        open_path(path)
    else:
        ui.notify(f"No PDF for {key}", type="warning")


def show_pdf(key, page=None):
    """A large dialog with the PDF at `page`, plus 'open in PDF reader' and BibTeX buttons."""
    with ui.dialog().props("maximized") as dialog, ui.card().classes("w-full h-full p-0 gap-0"):
        with ui.row().classes("w-full items-center justify-between px-4 py-2 border-b bg-white"):
            ui.label(paper_title(key) + (f"  - page {page}" if page else "")).classes("font-medium")
            with ui.row().classes("gap-2"):
                ui.button("Paper page", icon="article",
                          on_click=lambda: (dialog.close(), ui.navigate.to(f"/paper/{key}"))).props("flat dense")
                ui.button("Open in PDF reader", icon="open_in_new",
                          on_click=lambda: open_external(key)).props("flat dense")
                ui.button(icon="close", on_click=dialog.close).props("flat round dense")
        ui.element("iframe").props(f'src="{pdf_url(key, page)}"').classes("w-full grow border-0").style(
            "height: calc(100vh - 60px)")
    dialog.open()
    return dialog
