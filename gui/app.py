"""Desktop app (NiceGUI): `jmlr-rag gui` opens it in its own window; `--browser` serves it in the browser.

Routes                      module (one file per page; each exposes page(...))
  /, /chat                  gui/chat.py        conversation with memory, sources, PDF at the cited page
  /search                   gui/search.py      ranked papers, Scholar-style queries, filters
  /paper/{vNN}/{id}         gui/paper.py       one paper: abstract, sections, similar papers, BibTeX, PDF
  /topics                   gui/topicmap.py    interactive topic map
  /advances                 gui/advances.py    recent advances per subtopic, momentum charts
  /sessions                 gui/sessions.py    saved conversations
Shared: gui/frame.py (header + navigation), gui/state.py (index loaded once, model, conversation),
gui/pdfview.py (PDF dialog). PDFs are served from the JMLR folder at /pdfs/.
"""
from nicegui import app, ui

from common import JMLR_DIR
from gui import advances, chat, paper, search, sessions, state, topicmap
from gui.frame import frame

app.add_static_files("/pdfs", str(JMLR_DIR))


@ui.page("/")
@ui.page("/chat")
def _chat():
    with frame("Chat", "chat"):
        chat.page()


@ui.page("/search")
def _search(q: str = ""):
    with frame("Search", "search"):
        search.page(q)


@ui.page("/paper/{volume}/{pid}")
def _paper(volume: str, pid: str):
    with frame("Paper", "search"):
        paper.page(f"{volume}/{pid}")


@ui.page("/topics")
def _topics():
    with frame("Topic map", "topics"):
        topicmap.page()


@ui.page("/advances")
def _advances():
    with frame("Recent advances", "advances"):
        advances.page()


@ui.page("/sessions")
def _sessions():
    with frame("Sessions", "sessions"):
        sessions.page()


def run(native=True, port=8765, no_rewrite=False, load_index=True):
    if load_index:
        state.start_loading(no_rewrite)
    ui.run(title="JMLR research assistant", native=native, window_size=(1480, 940) if native else None,
           port=port, reload=False, show=not native, favicon="📚", show_welcome_message=False)
