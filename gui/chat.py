"""Chat - a conversation with memory over the JMLR papers: streamed Markdown answers, clickable [n] citations
that open the PDF at the cited page, a sources list under every answer, /compare and /reading.

The blocking work (condensing a follow-up, retrieval, the Claude call) runs in a worker thread
(nicegui.run.io_bound); the worker only appends streamed text to a buffer, and the event handler, which stays
on the UI loop, re-renders the answer from that buffer a few times per second.
"""
import argparse
import asyncio
import contextlib
import io
import itertools
import threading

from nicegui import context, run, ui

import ask
import ui as term_ui
from conversation import Conversation
from gui import state
from gui.chat_render import (linkify_citations, oneshot_markdown, source_from_hit, source_from_label,
                             source_tooltip)
from gui.pdfview import show_pdf

REFRESH = 0.15                       # seconds between re-renders of a streaming answer
BUSY = threading.Lock()              # one answer at a time (the conversation is shared by all windows)
_IDS = itertools.count(1)
MODEL_NAMES = {"haiku": "Claude Haiku 5.5", "opus": "Claude Opus 5.5"}

EXAMPLES = [
    ("help_outline", "How do kernel two-sample tests choose the kernel bandwidth?"),
    ("compare_arrows", "/compare How do these papers handle label noise?"),
    ("menu_book", "/reading Gaussian processes for machine learning"),
]

CSS = """
.chat-md { font-size: 0.95rem; line-height: 1.6; color: #1f2937; overflow-wrap: anywhere; }
.chat-md h1, .chat-md h2 { font-size: 1.2rem; font-weight: 600; margin: 1rem 0 .4rem; }
.chat-md h3, .chat-md h4 { font-size: 1.05rem; font-weight: 600; margin: .9rem 0 .3rem; }
.chat-md p { margin: .45rem 0; }
.chat-md ul, .chat-md ol { margin: .4rem 0 .4rem 1.4rem; }
.chat-md ul { list-style: disc; } .chat-md ol { list-style: decimal; }
.chat-md li { margin: .15rem 0; }
.chat-md table { border-collapse: collapse; margin: .6rem 0; font-size: .88rem; display: block; overflow-x: auto; }
.chat-md th, .chat-md td { border: 1px solid #e5e7eb; padding: .35rem .6rem; text-align: left; vertical-align: top; }
.chat-md th { background: #f3f4f8; font-weight: 600; }
.chat-md code { background: #f3f4f6; padding: .05rem .3rem; border-radius: 4px; font-size: .85em; }
.chat-md pre { background: #f6f8fa; padding: .7rem; border-radius: 8px; overflow-x: auto; }
.chat-md pre code { background: none; padding: 0; }
.chat-md blockquote { border-left: 3px solid #c5cdf5; padding-left: .8rem; color: #4b5563; }
.chat-md a { color: #3b5bdb; }
.chat-md a.cite { display: inline-block; min-width: 1.15rem; padding: 0 .3rem; margin: 0 1px; border-radius: 6px;
  background: #e8ecfd; color: #3b5bdb; font-size: .75rem; font-weight: 600; line-height: 1.15rem;
  text-align: center; text-decoration: none; vertical-align: 1px; cursor: pointer; }
.chat-md a.cite:hover { background: #3b5bdb; color: white; }
.chat-md .cite-group { white-space: nowrap; }
.user-bubble { background: #3b5bdb; color: white; border-radius: 18px 18px 4px 18px; white-space: pre-wrap;
  overflow-wrap: anywhere; }
.src-row:hover { background: #f5f7ff; }
.chat-input .q-field__control { border-radius: 14px; }
"""

# event delegation: a click on any a.cite (also in answers added later) is sent to Python as a 'cite' event
CITE_JS = """<script>
document.addEventListener('click', (e) => {
  const a = e.target.closest && e.target.closest('a.cite');
  if (!a) return;
  e.preventDefault();
  emitEvent('cite', {a: a.dataset.a, n: Number(a.dataset.n)});
});
</script>"""

SCROLL_JS = ("const el = document.scrollingElement;"
             "if ({force} || el.scrollHeight - el.scrollTop - window.innerHeight < 350)"
             " window.scrollTo({{top: el.scrollHeight, behavior: '{behavior}'}});")


def scroll_down(force=False, smooth=True):
    ui.run_javascript(SCROLL_JS.format(force="true" if force else "false", behavior="smooth" if smooth else "auto"))


class Answer:
    """One assistant message: status line, Markdown body, footnote, sources."""

    def __init__(self, view):
        self.view = view
        self.id = f"a{next(_IDS)}"
        self.sources = []
        view.answers[self.id] = self
        with ui.row().classes("w-full no-wrap items-start gap-3"):
            ui.avatar("auto_stories", color="indigo-1", text_color="primary", size="34px").classes("mt-1")
            with ui.column().classes("grow min-w-0 gap-1"):
                self.status = ui.row().classes("items-center gap-2 text-gray-500 text-sm")
                with self.status:
                    self.spinner = ui.spinner("dots", size="1.4em", color="primary")
                    self.status_label = ui.label("")
                self.status.visible = False
                self.query_label = ui.label("").classes("text-xs text-gray-500 italic")
                self.query_label.visible = False
                self.md = ui.markdown("").classes("chat-md w-full")
                self.foot = ui.label("").classes("text-xs text-gray-400")
                self.foot.visible = False
                self.extra = ui.column().classes("w-full gap-1")

    def set_status(self, text):
        self.status_label.set_text(text)
        self.status.visible = bool(text)

    def set_text(self, text, linkify=True):
        self.md.set_content(linkify_citations(text, self.id, self.sources) if linkify else text)

    def footnote(self, text):
        self.foot.set_text(text)
        self.foot.visible = bool(text)

    def error(self, text):
        with self.extra:
            with ui.row().classes("items-start gap-2 bg-red-50 text-red-800 rounded-lg px-3 py-2 text-sm w-full"):
                ui.icon("error_outline").classes("text-lg")
                ui.label(text).classes("grow")

    def warn(self, text):
        with self.extra:
            with ui.row().classes("items-center gap-2 bg-amber-50 text-amber-800 rounded-lg px-3 py-1 text-sm"):
                ui.icon("warning_amber")
                ui.label(text)

    def show_sources(self, sources, open_=True):
        self.sources = sources
        if not sources:
            return
        n_papers = len({s["key"] for s in sources})
        with self.extra:
            with ui.expansion(f"Sources - {len(sources)} passages from {n_papers} papers", icon="library_books",
                              value=open_).classes("w-full border border-gray-200 rounded-lg bg-white text-sm").props(
                    "dense header-class='text-gray-600'"):
                for i, s in enumerate(sources, 1):
                    self._source_row(i, s)

    def _source_row(self, i, s):
        with ui.row().classes("src-row w-full no-wrap items-center gap-2 px-1 py-0.5 rounded"):
            ui.label(str(i)).classes("text-xs font-semibold text-primary bg-indigo-50 rounded px-1.5 min-w-[1.6rem] "
                                     "text-center")
            with ui.column().classes("grow min-w-0 gap-0"):
                ui.label(s["title"]).classes("text-sm text-gray-800 truncate w-full").tooltip(s["title"])
                meta = [m for m in (s.get("who"), str(s.get("year") or "")) if m]
                where = []
                if s.get("page"):
                    where.append(f"p.{s['page']}" + (f"-{s['page_end']}" if s.get("page_end") and
                                                      s["page_end"] != s["page"] else ""))
                if s.get("section"):
                    where.append(s["section"])
                ui.label(" · ".join([", ".join(meta)] + where if meta else where)).classes(
                    "text-xs text-gray-500 truncate w-full")
            if s.get("key"):
                ui.button(icon="picture_as_pdf", on_click=lambda s=s: show_pdf(s["key"], s.get("page"))).props(
                    "flat round dense size=sm color=primary").tooltip(
                    f"Open the PDF at page {s['page']}" if s.get("page") else "Open the PDF")
                ui.button(icon="article", on_click=lambda s=s: ui.navigate.to(f"/paper/{s['key']}")).props(
                    "flat round dense size=sm color=grey-7").tooltip("Paper page")


class ChatView:
    def __init__(self):
        self.answers = {}
        ui.add_css(CSS)
        ui.add_body_html(CITE_JS)
        ui.on("cite", self.on_cite)

        with ui.column().classes("w-full max-w-4xl mx-auto gap-5 pb-48"):
            self.loading = ui.card().classes("w-full items-center p-6 gap-2")
            with self.loading:
                self.loading_spinner = ui.spinner("dots", size="2.5em", color="primary")
                self.loading_title = ui.label("Loading the search index").classes("text-lg font-medium")
                self.loading_text = ui.label("Embeddings, search model and reranker for 4,138 papers - this takes "
                                             "about half a minute. The chat opens as soon as it is ready.").classes(
                    "text-sm text-gray-500 text-center")
            self.welcome = ui.column().classes("w-full items-center gap-3 pt-6")
            with self.welcome:
                ui.icon("forum", color="primary").classes("text-5xl")
                ui.label("Ask the JMLR papers").classes("text-2xl font-semibold text-gray-800")
                ui.label("Answers cite the passages they use; click a [n] to open the PDF at that page. "
                         "Follow-up questions remember the conversation.").classes(
                    "text-gray-500 text-center max-w-xl")
                with ui.row().classes("gap-2 justify-center pt-2"):
                    for icon, text in EXAMPLES:
                        ui.button(text, icon=icon, on_click=lambda t=text: self.use_example(t)).props(
                            "outline no-caps rounded color=primary").classes("bg-white text-sm")
            self.messages = ui.column().classes("w-full gap-5")

        with ui.page_sticky("bottom", y_offset=0, expand=True).classes("z-10"):
            with ui.column().classes("w-full items-center px-4 pt-6 pb-4").style(
                    "background: linear-gradient(to bottom, rgba(247,248,250,0), #f7f8fa 35%)"):
                with ui.card().classes("w-full max-w-4xl p-2 gap-1 shadow-md rounded-2xl"):
                    with ui.row().classes("w-full no-wrap items-end gap-2"):
                        self.input = ui.textarea(placeholder="Ask about the JMLR papers...").props(
                            "autogrow borderless dense input-class='max-h-48'").classes(
                            "chat-input grow px-2 text-base")
                        self.input.on("keydown.enter.exact.prevent", self.send)
                        self.send_btn = ui.button(icon="arrow_upward", on_click=self.send).props(
                            "round unelevated color=primary").mark("send").tooltip("Send (Enter)")
                    with ui.row().classes("w-full items-center justify-between px-2"):
                        self.hint = ui.label("").classes("text-xs text-gray-400")
                        self.new_btn = ui.button("New chat", icon="add_comment", on_click=self.new_chat).props(
                            "flat dense no-caps size=sm color=grey-8")

        self.render_conversation(state.STATE["conv"])
        self.was_ready = None
        self.refresh_ready()
        self.ready_timer = ui.timer(0.5, self.refresh_ready)

    # ------------------------------------------------------------ state

    def refresh_ready(self):
        ready = state.ready()
        if state.STATE["error"]:
            self.loading_spinner.visible = False
            self.loading_title.set_text("The index failed to load")
            self.loading_text.set_text(state.STATE["error"])
            self.loading.visible = True
        else:
            self.loading.visible = not ready
        busy = BUSY.locked()
        can_send = ready and not busy
        self.send_btn.set_enabled(can_send)
        if self.input.enabled != ready:
            self.input.set_enabled(ready)
            self.input.props(f'placeholder="{"Ask about the JMLR papers..." if ready else "Loading the index..."}"')
        self.new_btn.set_enabled(not busy)
        if not ready:
            self.hint.set_text("Waiting for the index to load...")
        elif busy:
            self.hint.set_text("Working on the answer...")
        else:
            self.hint.set_text("Enter to send · Shift+Enter for a new line · /compare <question> · /reading <topic>")
        if ready and not self.was_ready and not busy:
            self.input.run_method("focus")
        self.was_ready = ready

    def render_conversation(self, conv):
        """Show the conversation's saved turns (a resumed session) or the welcome text."""
        self.messages.clear()
        self.answers.clear()
        self.welcome.visible = not conv.turns
        with self.messages:
            if conv.turns:
                when = conv.created.replace("T", " ")[:16]
                ui.label(f"Conversation started {when} · saved under Sessions").classes(
                    "text-xs text-gray-400 self-center")
            for t in conv.turns:
                self.user_bubble(t.question)
                a = Answer(self)
                if t.query and t.query != t.question:
                    a.query_label.set_text(f"Searched for: {t.query}")
                    a.query_label.visible = True
                a.show_sources([source_from_label(s) for s in t.sources], open_=False)
                a.set_text(t.answer)
        if conv.turns:
            scroll_down(force=True, smooth=False)

    def user_bubble(self, text):
        with ui.row().classes("w-full justify-end"):
            ui.label(text).classes("user-bubble px-4 py-2 max-w-[80%] text-[0.95rem]")

    def use_example(self, text):
        self.input.set_value(text)
        self.input.run_method("focus")

    def new_chat(self):
        if BUSY.locked():
            ui.notify("Wait for the current answer to finish.", type="warning")
            return
        state.STATE["conv"] = Conversation()
        self.render_conversation(state.STATE["conv"])
        ui.notify("New conversation - the previous one stays saved under Sessions.", type="info")

    def on_cite(self, e):
        args = e.args if isinstance(e.args, dict) else (e.args[0] if e.args else {})
        a = self.answers.get(args.get("a"))
        n = args.get("n")
        if not a or not isinstance(n, int) or not 1 <= n <= len(a.sources):
            return
        s = a.sources[n - 1]
        if s.get("key"):
            show_pdf(s["key"], s.get("page"))
        else:
            ui.notify(source_tooltip(s))

    # ------------------------------------------------------------ sending

    async def send(self):
        text = (self.input.value or "").strip()
        if not text:
            return
        if not state.ready():
            ui.notify("The index is still loading - one moment.", type="warning")
            return
        if not BUSY.acquire(blocking=False):
            ui.notify("Wait for the current answer to finish.", type="warning")
            return
        answer = None
        try:
            self.input.set_value("")
            self.refresh_ready()
            self.welcome.visible = False
            with self.messages:
                self.user_bubble(text)
                answer = Answer(self)
            scroll_down(force=True)
            cmd = text.split(" ", 1)[0].lower()
            if cmd in ("/compare", "/reading"):
                await self.run_oneshot(cmd, text.partition(" ")[2].strip(), answer)
            elif text.startswith("/"):
                answer.error("Commands here: /compare <question> and /reading <topic>. "
                             "Use the New chat button to start over and the Sessions page to resume.")
            else:
                await self.run_turn(text, answer)
        finally:
            BUSY.release()
            if answer:
                answer.set_status("")
            self.refresh_ready()

    async def run_turn(self, question, answer):
        ix, conv, model = state.index(), state.STATE["conv"], state.STATE["model"]
        answer.set_status("Condensing the follow-up and searching..." if len(conv) else
                          "Searching 4,138 papers...")
        try:
            turn = await run.io_bound(ask.prepare_chat_turn, ix, conv, question)
        except Exception as e:
            answer.error(f"Search failed: {type(e).__name__}: {e}")
            return
        term_ui.add_cost(turn["condense_cost"])
        if turn["condense_error"]:
            answer.query_label.set_text(f"Could not condense the follow-up ({turn['condense_error']}); "
                                        "searched for the question as typed.")
            answer.query_label.visible = True
        elif turn["query"] != question:
            answer.query_label.set_text(f"Searching for: {turn['query']}")
            answer.query_label.visible = True
        hits = turn["hits"]
        if not hits:
            answer.set_status("")
            answer.error("No matching passages found. Try other words or a broader question.")
            return
        answer.sources = [source_from_hit(ix.papers, h) for h in hits]
        n_papers = len({h["paper_key"] for h in hits})
        mem = f", with {len(turn['history']) // 2} earlier turn(s) as context" if turn["history"] else ""
        answer.set_status(f"Writing the answer with {MODEL_NAMES[model]} from {len(hits)} passages in {n_papers} papers{mem}...")

        buf = []

        def render(chunks):
            for t in chunks:
                buf.append(t)

        task = asyncio.ensure_future(run.io_bound(ask.stream_claude, model, turn["system"], turn["user"],
                                                  turn["history"], render))
        shown = 0
        while not task.done():
            await asyncio.wait([task], timeout=REFRESH)
            if len(buf) != shown:
                shown = len(buf)
                answer.set_text("".join(buf) + " ▍")
                scroll_down()
        try:
            msg, cost = task.result()
        except ask.ClaudeError as e:
            answer.set_text("".join(buf))
            answer.set_status("")
            answer.error(str(e))
            return
        except Exception as e:
            answer.set_text("".join(buf))
            answer.set_status("")
            answer.error(f"{type(e).__name__}: {e}")
            return
        text = ask.answer_text(msg) or "".join(buf)
        answer.set_status("")
        answer.set_text(text or "_(no answer)_")
        try:
            await run.io_bound(ask.record_chat_turn, ix, conv, question, turn, msg)
        except Exception as e:
            answer.warn(f"The turn could not be saved: {type(e).__name__}: {e}")
        if msg.stop_reason == "refusal":
            answer.warn("The model declined to answer this request (not added to the conversation memory).")
        elif msg.stop_reason == "max_tokens":
            answer.warn("The answer was cut off at the length limit.")
        u = msg.usage
        extra = f" + condensing ~${turn['condense_cost']:.4f}" if turn["condense_cost"] else ""
        answer.footnote(f"{msg.model} · {u.input_tokens:,} in / {u.output_tokens:,} out tokens · ~${cost:.4f}{extra}")
        answer.show_sources(answer.sources)
        answer.set_text(text or "_(no answer)_")     # re-link citations now that the sources are final
        scroll_down()

    async def run_oneshot(self, cmd, query, answer):
        """/compare and /reading: run the terminal command in a thread, capture its printed output and show
        it as Markdown while it streams. They neither use nor change the conversation memory."""
        if not query:
            answer.error(f"Usage: {cmd} <{'question' if cmd == '/compare' else 'topic'}>")
            return
        ix, model = state.index(), state.STATE["model"]
        a = argparse.Namespace(query=query, model=model, k=None, since=None, until=None, papers=None, filter=None)
        fn = ask.cmd_compare if cmd == "/compare" else ask.cmd_reading
        answer.set_status("Comparing papers side by side (about a minute)..." if cmd == "/compare"
                          else "Building a reading path...")
        out = io.StringIO()
        before = state.spent()

        def work():
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                try:
                    fn(ix, a)
                except SystemExit as e:
                    if e.code not in (None, 0):
                        print(f"\n**{e.code}**")

        task = asyncio.ensure_future(run.io_bound(work))
        shown = ""
        while not task.done():
            await asyncio.wait([task], timeout=REFRESH * 2)
            text = out.getvalue()
            if text != shown:
                shown = text
                answer.set_text(oneshot_markdown(text) + " ▍", linkify=False)
                scroll_down()
        answer.set_status("")
        try:
            task.result()
        except Exception as e:
            answer.error(f"{type(e).__name__}: {e}")
        answer.set_text(oneshot_markdown(out.getvalue()) or "_(no output)_", linkify=False)
        answer.footnote(f"{cmd} with {ask.MODELS[model]} · ~${state.spent() - before:.4f} · "
                        "not added to the conversation memory")
        scroll_down()


def page():
    ChatView()
