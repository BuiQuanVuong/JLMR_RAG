"""Full-screen terminal app for the JMLR research assistant (Textual).

  jmlr-rag            (no arguments, in a terminal) opens it;  jmlr-rag app  also does
  jmlr-rag chat       keeps the classic line-based chat

Tabs (F1-F4):
  Chat      conversation with memory; answers stream as Markdown; the sources pane lists the cited passages,
            Enter on one opens its PDF. Type /compare <question> or /reading <topic> for those modes.
  Search    ranked papers; the highlighted paper's abstract is shown; Enter opens the PDF
  Topics    the topic map: areas -> subtopics -> papers, with trends, top authors and abstracts
  Sessions  saved conversations; Enter resumes one in the Chat tab

Keys: Ctrl+N new conversation, Ctrl+O switch Haiku/Opus, Ctrl+Q quit.
Retrieval and Claude calls run in background threads, so the screen stays responsive.
"""
import contextlib
import io
import os
import time
from collections import Counter

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown, OptionList, Static, TabbedContent, TabPane, Tree
from textual.widgets.option_list import Option

import ask
import topics
import ui
from common import JMLR_DIR, REWRITE, connect
from conversation import Conversation, list_sessions

WELCOME = """# JMLR research assistant

Ask a question about the 4,138 JMLR papers (2000-2025). Follow-ups build on the conversation, which is saved
automatically. **F2** search, **F3** topic map, **F4** saved sessions, **Ctrl+N** new conversation,
**Ctrl+O** switch between Claude Haiku 5.5 and Opus 5.5.

In the chat box, `/compare <question>` compares papers side by side and `/reading <topic>` builds a reading path.
"""


def load_index(no_rewrite=False):
    """Separate function so tests can substitute a small fake index."""
    return ask.Index(rewrite="off") if no_rewrite else ask.Index()


@contextlib.contextmanager
def quiet():
    """Keep stray prints from background work off the full-screen display."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


def open_pdf(key):
    con = connect()
    p = con.execute("SELECT pdf_path FROM papers WHERE key=?", (key,)).fetchone()
    if p:
        os.startfile(str(JMLR_DIR / p["pdf_path"]))
    return p is not None


class ResearchApp(App):
    TITLE = "JMLR research assistant"
    CSS = """
    #chat-body { height: 1fr; }
    #log { width: 2fr; padding: 0 1; }
    #side { width: 1fr; border-left: solid $primary-darken-2; padding: 0 1; }
    #sources { height: 1fr; }
    .you { color: $accent; margin-top: 1; }
    .answer { margin-bottom: 1; }
    #search-body, #topics-body, #sessions-body { height: 1fr; }
    #search-results, #topic-tree, #session-list { width: 1fr; }
    #search-detail, #topic-detail, #session-preview { width: 1fr; padding: 0 1; border-left: solid $primary-darken-2; }
    .pane-title { text-style: bold; color: $accent; }
    """
    BINDINGS = [
        Binding("f1", "tab('chat')", "Chat"),
        Binding("f2", "tab('search')", "Search"),
        Binding("f3", "tab('topics')", "Topics"),
        Binding("f4", "tab('sessions')", "Sessions"),
        Binding("ctrl+n", "new_chat", "New chat"),
        Binding("ctrl+o", "toggle_model", "Haiku/Opus"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, model="haiku", no_rewrite=False, resume=None):
        super().__init__()
        self.model_name = model
        self.no_rewrite = no_rewrite
        self.ix = None
        self.conv = Conversation()
        self.resume_ref = resume
        self.busy = False
        self.sources = []          # (label, key) for the current answer
        self.search_hits = []
        self.session_rows = []

    # ------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="chat"):
            with TabPane("Chat", id="chat"):
                with Horizontal(id="chat-body"):
                    with VerticalScroll(id="log"):
                        yield Markdown(WELCOME)
                    with Vertical(id="side"):
                        yield Static("Sources (Enter opens the PDF)", classes="pane-title")
                        yield OptionList(id="sources")
                yield Input(placeholder="Loading the index (about 30 s)...", id="question", disabled=True)
            with TabPane("Search", id="search"):
                yield Input(placeholder="Search papers (free, local) - Enter to search", id="search-q", disabled=True)
                with Horizontal(id="search-body"):
                    yield OptionList(id="search-results")
                    yield Markdown("", id="search-detail")
            with TabPane("Topics", id="topics"):
                with Horizontal(id="topics-body"):
                    yield Tree(Text("JMLR topic map"), id="topic-tree")
                    yield Markdown("Select an area, subtopic or paper.", id="topic-detail")
            with TabPane("Sessions", id="sessions"):
                with Horizontal(id="sessions-body"):
                    yield OptionList(id="session-list")
                    yield Markdown("Saved conversations. Enter resumes one.", id="session-preview")
        yield Footer()

    def on_mount(self):
        self.update_status()
        self.build_topic_tree()
        self.refresh_sessions()
        if self.resume_ref:
            from conversation import resolve_session
            sid = resolve_session(self.resume_ref)
            if sid:
                self.show_conversation(Conversation.load(sid))
        self.load_index_worker()

    def update_status(self):
        n = len(self.conv)
        state = "loading index..." if self.ix is None else ("working..." if self.busy else "ready")
        self.sub_title = (f"{ask.MODELS[self.model_name]} | memory: {n} turn{'' if n == 1 else 's'} | "
                          f"session ~${ui.SESSION['cost']:.4f} | {state}")

    # ------------------------------------------------------------ index loading

    @work(thread=True, exclusive=True, group="index")
    def load_index_worker(self):
        try:
            with quiet():
                ix = load_index(self.no_rewrite)
        except BaseException as e:                     # includes sys.exit from Index()
            self.call_from_thread(self.notify, f"Could not load the index: {e}", severity="error", timeout=30)
            return
        self.call_from_thread(self.index_ready, ix)

    def index_ready(self, ix):
        self.ix = ix
        for wid, text in (("#question", "Ask a question (Enter). Follow-ups use the conversation."),
                          ("#search-q", "Search papers (free, local) - Enter to search")):
            w = self.query_one(wid, Input)
            w.disabled, w.placeholder = False, text
        self.query_one("#question", Input).focus()
        self.update_status()
        self.notify("Index loaded.", timeout=3)

    # ------------------------------------------------------------ actions

    def action_tab(self, tab):
        self.query_one(TabbedContent).active = tab
        target = {"chat": "#question", "search": "#search-q", "topics": "#topic-tree",
                  "sessions": "#session-list"}[tab]
        self.query_one(target).focus()
        if tab == "sessions":
            self.refresh_sessions()

    def action_new_chat(self):
        if self.busy:
            return self.notify("Wait for the current answer to finish.", severity="warning")
        self.show_conversation(Conversation())
        self.notify("New conversation (the previous one stays saved).", timeout=3)

    def action_toggle_model(self):
        self.model_name = "opus" if self.model_name == "haiku" else "haiku"
        self.update_status()
        self.notify(f"Answers now use {ask.MODELS[self.model_name]}.", timeout=3)

    # ------------------------------------------------------------ chat

    def show_conversation(self, conv):
        """Replace the chat log with conv's turns (used for new and resumed conversations)."""
        self.conv = conv
        log = self.query_one("#log", VerticalScroll)
        log.remove_children()
        if not conv.turns:
            log.mount(Markdown(WELCOME))
        for t in conv.turns:
            log.mount(Static(Text(f"You: {t.question}"), classes="you"))
            log.mount(Markdown(t.answer, classes="answer"))
        self.set_sources([(s, s.rsplit("[", 1)[-1].rstrip("]")) for s in (conv.turns[-1].sources if conv.turns else [])])
        log.scroll_end(animate=False)
        self.update_status()

    def set_sources(self, items):
        self.sources = items
        ol = self.query_one("#sources", OptionList)
        ol.clear_options()
        ol.add_options([Option(Text(f"[{i}] {label}")) for i, (label, _) in enumerate(items, 1)])

    def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip()
        if not text:
            return
        if event.input.id == "question":
            if self.busy:
                return self.notify("Wait for the current answer to finish.", severity="warning")
            event.input.value = ""
            self.start_turn(text)
        elif event.input.id == "search-q":
            self.search_worker(text)

    def start_turn(self, text):
        log = self.query_one("#log", VerticalScroll)
        if len(log.children) == 1 and not self.conv.turns:
            log.remove_children()                      # drop the welcome text
        log.mount(Static(Text(f"You: {text}"), classes="you"))
        answer = Markdown("_Searching the papers..._", classes="answer")
        log.mount(answer)
        log.scroll_end(animate=False)
        self.busy = True
        self.update_status()
        if text.startswith("/compare ") or text.startswith("/reading "):
            self.oneshot_worker(text, answer)
        else:
            self.chat_worker(text, answer)

    def _streamer(self, answer, log):
        """render() for ask.stream_claude: update the answer widget a few times per second."""
        def render(chunks):
            buf, last = "", 0.0
            for t in chunks:
                buf += t
                if time.monotonic() - last > 0.15:
                    self.call_from_thread(answer.update, buf)
                    self.call_from_thread(log.scroll_end, animate=False)
                    last = time.monotonic()
            self.call_from_thread(answer.update, buf or "_(no answer)_")
            self.call_from_thread(log.scroll_end, animate=False)
        return render

    @work(thread=True, exclusive=True, group="chat")
    def chat_worker(self, question, answer):
        log = self.query_one("#log", VerticalScroll)
        try:
            with quiet():
                turn = ask.prepare_chat_turn(self.ix, self.conv, question)
            ui.add_cost(turn["condense_cost"])
            if not turn["hits"]:
                self.call_from_thread(answer.update, "_No matching passages found._")
                return
            labels = [ask.source_label(self.ix, h) for h in turn["hits"]]
            self.call_from_thread(self.set_sources, [(l, h["paper_key"]) for l, h in zip(labels, turn["hits"])])
            if len(self.conv) and turn["query"] != question:
                self.call_from_thread(self.notify, f"Searching for: {turn['query']}", timeout=6)
            self.call_from_thread(answer.update, "_Writing the answer..._")
            with quiet():
                msg, cost = ask.stream_claude(self.model_name, turn["system"], turn["user"], turn["history"],
                                              render=self._streamer(answer, log))
                ask.record_chat_turn(self.ix, self.conv, question, turn, msg)
            if msg.stop_reason == "refusal":
                self.call_from_thread(self.notify, "The model declined to answer.", severity="warning")
            self.call_from_thread(self.notify, f"{msg.model}: {msg.usage.input_tokens} in / "
                                               f"{msg.usage.output_tokens} out, ~${cost:.4f}", timeout=5)
        except ask.ClaudeError as e:
            self.call_from_thread(answer.update, f"**Error:** {e}")
        except Exception as e:
            self.call_from_thread(answer.update, f"**Error:** {type(e).__name__}: {e}")
        finally:
            self.busy = False
            self.call_from_thread(self.update_status)
            self.call_from_thread(self.refresh_sessions)

    @work(thread=True, exclusive=True, group="chat")
    def oneshot_worker(self, text, answer):
        """/compare and /reading: run the existing command, capture its output, show it as Markdown."""
        import argparse
        cmd, _, rest = text.partition(" ")
        a = argparse.Namespace(query=rest.strip(), model=self.model_name, k=None, since=None, until=None,
                               papers=None)
        self.call_from_thread(answer.update, f"_Running {cmd} (this takes a little while)..._")
        try:
            with quiet() as buf:
                try:
                    (ask.cmd_compare if cmd == "/compare" else ask.cmd_reading)(self.ix, a)
                except SystemExit as e:
                    print(f"\n**{e}**")
            self.call_from_thread(answer.update, buf.getvalue().replace("\n ", "\n\n "))
        finally:
            self.busy = False
            self.call_from_thread(self.update_status)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        lid, i = event.option_list.id, event.option_index
        if lid == "sources" and i < len(self.sources):
            self._open(self.sources[i][1])
        elif lid == "search-results" and i < len(self.search_hits):
            self._open(self.search_hits[i][0]["key"])
        elif lid == "session-list" and i < len(self.session_rows):
            if self.busy:
                return self.notify("Wait for the current answer to finish.", severity="warning")
            self.show_conversation(Conversation.load(self.session_rows[i]["id"]))
            self.action_tab("chat")
            self.notify(f"Resumed: {self.conv.title}", timeout=4)

    def _open(self, key):
        try:
            ok = open_pdf(key)
            self.notify(f"Opening {key}" if ok else f"No PDF for {key}", timeout=3)
        except OSError as e:
            self.notify(f"Could not open the PDF: {e}", severity="error")

    # ------------------------------------------------------------ search

    @work(thread=True, exclusive=True, group="search")
    def search_worker(self, q):
        with quiet():
            ranked = self.ix.rank_papers(q, k=25)
            topic_of = topics.paper_topic_names(connect())
        self.call_from_thread(self.show_search, ranked, topic_of)

    def show_search(self, ranked, topic_of):
        self.search_hits = ranked
        ol = self.query_one("#search-results", OptionList)
        ol.clear_options()
        ol.add_options([Option(Text(f"{p['title']} ({p['year']})")) for p, _ in ranked])
        self._topic_of = topic_of
        if ranked:
            ol.highlighted = 0
            ol.focus()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted):
        lid, i = event.option_list.id, event.option_index
        if lid == "search-results" and i is not None and i < len(self.search_hits):
            p, hit = self.search_hits[i]
            where = ""
            if hit:
                sec = f", {hit['section']}" if hit.get("section") else ""
                where = f"\n\n**Best match** (p.{hit['page_start']}{sec}): {hit['text'][:600]}..."
            topic = getattr(self, "_topic_of", {}).get(p["key"], "")
            self.query_one("#search-detail", Markdown).update(
                f"### {p['title']}\n\n{p['authors']} ({p['year']})  `{p['key']}`"
                + (f"\n\nTopic: {topic}" if topic else "") + f"\n\n{p['abstract'] or ''}{where}\n\n_Enter opens the PDF._")
        elif lid == "session-list" and i is not None and i < len(self.session_rows):
            conv = Conversation.load(self.session_rows[i]["id"])
            qs = "\n".join(f"{n}. {t.question}" for n, t in enumerate(conv.turns, 1))
            self.query_one("#session-preview", Markdown).update(
                f"### {conv.title}\n\nStarted {conv.created.replace('T', ' ')}\n\n{qs}\n\n_Enter resumes it._")

    # ------------------------------------------------------------ sessions

    def refresh_sessions(self):
        self.session_rows = list_sessions(limit=100)
        ol = self.query_one("#session-list", OptionList)
        ol.clear_options()
        ol.add_options([Option(Text(f"{s['updated'][:16].replace('T', ' ')}  ({s['turns']})  {s['title']}"))
                        for s in self.session_rows])

    # ------------------------------------------------------------ topic map

    def build_topic_tree(self):
        tree = self.query_one("#topic-tree", Tree)
        con = connect()
        if not topics.has_topics(con):
            tree.root.add_leaf(Text("No topic map yet - run: python topics.py build"))
            return
        rows = topics._topic_rows(con)
        tree.root.expand()
        for aid in sorted((t for t in rows if rows[t]["level"] == 1), key=int):
            area = tree.root.add(Text(f"{aid}. {rows[aid]['name']}  ({rows[aid]['size']})"), data=("topic", aid))
            for sid in sorted((t for t in rows if rows[t]["parent"] == aid), key=lambda t: int(t.split(".")[1])):
                area.add(Text(f"{sid}  {rows[sid]['name']}  ({rows[sid]['size']})"), data=("topic", sid))

    def on_tree_node_expanded(self, event: Tree.NodeExpanded):
        node = event.node
        if node.data and node.data[0] == "topic" and "." in node.data[1] and not node.children:
            con = connect()
            for m in sorted(topics._members(con, node.data[1]), key=lambda m: -m["score"]):
                node.add_leaf(Text(f"{m['year']}  {m['title']}"), data=("paper", m["paper_key"]))

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        if not event.node.data:
            return
        kind, ident = event.node.data
        detail = self.query_one("#topic-detail", Markdown)
        con = connect()
        if kind == "paper":
            p = con.execute("SELECT * FROM papers WHERE key=?", (ident,)).fetchone()
            detail.update(f"### {p['title']}\n\n{p['authors']} ({p['year']})  `{ident}`\n\n{p['abstract'] or ''}"
                          "\n\n_Press Enter again (or click) on the paper to open the PDF._")
            if getattr(self, "_last_selected", None) == ident:
                self._open(ident)
            self._last_selected = ident
            return
        self._last_selected = None
        rows, total = topics._topic_rows(con), topics._period_counts(con)
        r = rows[ident]
        members = topics._members(con, ident)
        spark, _ = topics._trend(members, total)
        counts = Counter(topics._period(m["year"]) for m in members)
        periods = " · ".join(f"{a}-{b % 100:02d}: {counts[i]}" for i, (a, b) in enumerate(topics.PERIODS))
        central = "\n".join(f"- {m['year']} {m['title']}" for m in sorted(members, key=lambda m: -m["score"])[:8])
        detail.update(f"### {ident}. {r['name']}\n\n{r['description']}\n\n**{r['size']} papers** - per period: "
                      f"{periods}  `{spark}`\n\n**Top authors:** {topics._authors(members)}\n\n"
                      f"**Most central papers:**\n\n{central}")


def run(a):
    ResearchApp(model=a.model, no_rewrite=a.no_rewrite, resume=getattr(a, "resume", None)).run()
