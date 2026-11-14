r"""Tests for the Chat and Sessions pages: no GPU, no API calls, sessions in a temp folder.

  .venv\Scripts\python -m pytest gui\chat_test.py -q

A fake index returns fixed passages and a fake Claude stream returns a fixed Markdown answer; everything else
(prompt building, chat memory, session saving/resuming, the pages and their event handlers) is the real code,
driven by NiceGUI's simulated user.
"""
import asyncio
import pathlib
import sys
import tempfile
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ask  # noqa: E402
import conversation  # noqa: E402
from common import connect  # noqa: E402
from gui.chat_render import fix_lists, linkify_citations, oneshot_markdown, source_from_label  # noqa: E402

KEY = "v13/gretton12a"
ANSWER = ("## Answer\n\nThe bandwidth is set by the **median heuristic** [1], see also [1, 2] and [9].\n\n"
          "| Paper | Rule |\n|---|---|\n| P1 | median [2] |\n\n`code [1]`")


class FakeIndex:
    def __init__(self):
        self.con = connect()
        self.papers = {r["key"]: dict(r) for r in self.con.execute("SELECT * FROM papers")}

    def passages(self, q, k=12, **kw):
        return [dict(id=1, paper_key=KEY, seq=3, page_start=10, page_end=10, section="2 The MMD", is_ref=0,
                     text="The median heuristic sets the kernel bandwidth to the median distance."),
                dict(id=2, paper_key=KEY, seq=9, page_start=24, page_end=25, section="7 Experiments", is_ref=0,
                     text="We compare bandwidth choices.")]

    def rank_papers(self, q, k=25, **kw):
        return [(self.papers[KEY], self.passages(q)[0])]


CALLS = []


def fake_stream(model_name, system, user_text, history=None, render=None):
    CALLS.append((model_name, len(history or [])))
    render = render or ask.ui.stream_answer
    render(iter([ANSWER[i:i + 9] for i in range(0, len(ANSWER), 9)]))
    msg = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=ANSWER)], stop_reason="end_turn",
                                model=ask.MODELS[model_name],
                                usage=types.SimpleNamespace(input_tokens=1000, output_tokens=200))
    return msg, 0.0002


def failing_stream(*a, **kw):
    raise ask.ClaudeError("No Anthropic API key found.")


MAIN = """
import importlib
import gui.app
importlib.reload(gui.app)            # re-register the pages after NiceGUI's global reset
from nicegui import ui
ui.run(storage_secret="test")
"""


# ------------------------------------------------------------ pure helpers

def test_linkify_citations():
    sources = [dict(key=KEY, title="Kernel two-sample test", year=2012, page=10, section="2"),
               dict(key=KEY, title="Kernel two-sample test", year=2012, page=24, section="7")]
    html = linkify_citations(ANSWER, "a7", sources)
    assert html.count('class="cite"') == 4                     # [1], [1, 2] (two) and the table's [2]
    assert 'data-a="a7" data-n="2"' in html
    assert "[9]" in html                                       # out of range: left alone
    assert "`code [1]`" in html                                # inline code: left alone
    assert linkify_citations("see [link](http://x) [1]", "a", sources).startswith("see [link](http://x)")


def test_fix_lists():
    assert fix_lists("**Head**\n- a\n- b") == "**Head**\n\n- a\n- b"
    assert fix_lists("text\n\n1. a\n2. b") == "text\n\n1. a\n2. b"
    assert fix_lists("```\nx\n- not a list\n```") == "```\nx\n- not a list\n```"


def test_source_from_label():
    ix = FakeIndex()
    label = ask.source_label(ix, ix.passages("q")[1])
    s = source_from_label(label)
    assert (s["key"], s["page"], s["section"], str(s["year"])) == (KEY, 24, "7 Experiments", "2012"), s
    assert s["title"] == ix.papers[KEY]["title"]


def test_oneshot_markdown():
    text = ("Found 2 candidate papers. Asking claude-haiku-5-5 to build a reading path...\n\n## Path\n\n1. [1] A\n"
            "\n(claude-haiku-5-5: 10 in / 5 out tokens, ~$0.0001)\n\nCandidates:\n [1] Kernel tests (2012)  [v13/gretton12a]\n"
            " [2] B (2020)  [v21/19-1]\n\nOpen a paper with:  python ask.py open <key>\n")
    md = oneshot_markdown(text)
    assert "- **[1]** Kernel tests (2012)  [[v13/gretton12a]](/paper/v13/gretton12a)" in md
    assert "*Found 2 candidate papers." in md and "Open a paper" not in md and "## Path" in md


# ------------------------------------------------------------ pages

@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(conversation, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(ask, "stream_claude", fake_stream)
    monkeypatch.setattr(conversation.Conversation, "condense",
                        lambda self, q, client=None: (f"{q} (standalone)", 0.0, None))
    from gui import state
    state.STATE.update(index=FakeIndex(), error=None, model="haiku", conv=conversation.Conversation())
    main = tmp_path / "main.py"
    main.write_text(MAIN, encoding="utf-8")
    CALLS.clear()
    return state, main


async def idle(timeout=10):
    from gui import chat
    await asyncio.sleep(0.05)
    for _ in range(int(timeout / 0.05)):
        if not chat.BUSY.locked():
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.1)


def markdowns(user):
    from nicegui import ui
    # find() returns a *set*: sort by element id (creation order) so [-1] is really the latest answer
    return [m.content for m in sorted(user.find(ui.markdown).elements, key=lambda m: m.id)]


@pytest.mark.asyncio
async def test_chat_flow(setup):
    from nicegui import ui
    from nicegui.testing.user_simulation import user_simulation
    state, main = setup
    async with user_simulation(main_file=main) as user:
        await user.open("/chat")
        await user.should_see("Ask the JMLR papers")
        box = user.find(ui.textarea)

        # first question, sent with Enter
        box.type("How do kernel two-sample tests choose the bandwidth?").trigger("keydown.enter.exact.prevent")
        await idle()
        conv = state.STATE["conv"]
        assert len(conv) == 1 and CALLS == [("haiku", 0)], (len(conv), CALLS)
        assert conversation.list_sessions()[0]["turns"] == 1
        md = markdowns(user)[-1]
        assert md.count('class="cite"') == 4 and "▍" not in md, md
        await user.should_see("Sources - 2 passages from 1 papers")
        await user.should_see("claude-haiku-5-5 · 1,000 in / 200 out tokens")
        assert box.elements.pop().value == ""

        # click on citation [1] -> PDF dialog at page 10
        layout = user.client.layout
        from nicegui import events
        for listener in layout._event_listeners.values():
            if listener.type == "cite":
                with user.client:
                    events.handle_event(listener.handler, events.GenericEventArguments(
                        sender=layout, client=user.client, args={"a": md.split('data-a="')[1].split('"')[0], "n": 1}))
        await asyncio.sleep(0.1)
        frames = [e for e in user.client.elements.values() if e.tag == "iframe"]
        assert frames and frames[-1].props["src"] == f"/pdfs/{KEY}.pdf#page=10", [f.props for f in frames]

        # follow-up: condensed query shown, history sent
        user.find(ui.textarea).type("Which of those handles functional data?")
        user.find(marker="send").click()                   # the send button works like Enter
        await idle()
        assert len(state.STATE["conv"]) == 2 and CALLS[-1] == ("haiku", 2), CALLS
        await user.should_see("Searching for: Which of those handles functional data? (standalone)")

        # /reading: captured output rendered as Markdown, memory untouched
        user.find(ui.textarea).type("/reading kernel tests").trigger("keydown.enter.exact.prevent")
        await idle()
        assert len(state.STATE["conv"]) == 2
        assert "(/paper/v13/gretton12a)" in markdowns(user)[-1], markdowns(user)[-1]
        await user.should_see("not added to the conversation memory")

        # an API error is shown inline
        ask.stream_claude = failing_stream
        user.find(ui.textarea).type("Another question").trigger("keydown.enter.exact.prevent")
        await idle()
        await user.should_see("No Anthropic API key found.")
        assert len(state.STATE["conv"]) == 2

        # New chat: fresh conversation, welcome text back
        user.find("New chat").click()
        await asyncio.sleep(0.1)
        assert len(state.STATE["conv"]) == 0
        await user.should_see("Ask the JMLR papers")


@pytest.mark.asyncio
async def test_loading_then_ready(setup):
    from nicegui import ui
    from nicegui.testing.user_simulation import user_simulation
    state, main = setup
    ix = state.STATE["index"]
    state.STATE["index"] = None
    async with user_simulation(main_file=main) as user:
        await user.open("/chat")
        await user.should_see("Loading the search index")
        assert not user.find(ui.textarea).elements.pop().enabled
        state.STATE["index"] = ix
        await asyncio.sleep(0.8)
        assert user.find(ui.textarea).elements.pop().enabled
        await user.should_not_see("Loading the search index")


@pytest.mark.asyncio
async def test_sessions_resume(setup):
    from nicegui import ui
    from nicegui.testing.user_simulation import user_simulation
    state, main = setup
    ix = FakeIndex()
    for qs in (["What is the lasso?"], ["How do kernel tests pick a bandwidth?", "And for functional data?"]):
        c = conversation.Conversation()
        for q in qs:
            c.add(q, q, ANSWER, [ask.source_label(ix, h) for h in ix.passages(q)])
        await asyncio.sleep(1.05)                     # session ids have one-second resolution
    async with user_simulation(main_file=main) as user:
        await user.open("/sessions")
        await user.should_see("2 conversations")
        await user.should_see("And for functional data?")
        user.find(ui.input).type("lasso")
        await asyncio.sleep(0.1)
        await user.should_not_see("And for functional data?")
        user.find(ui.input).elements.pop().set_value("kernel")
        await asyncio.sleep(0.1)
        user.find("Resume").click()
        await asyncio.sleep(0.3)
        assert [t.question for t in state.STATE["conv"].turns] == ["How do kernel tests pick a bandwidth?",
                                                                   "And for functional data?"]
        await user.should_see("Conversation started")
        mds = markdowns(user)
        assert len(mds) == 2 and 'class="cite"' in mds[0], mds
        await user.should_see("Sources - 2 passages from 1 papers")
