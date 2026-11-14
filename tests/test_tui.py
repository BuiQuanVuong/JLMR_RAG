r"""Headless test of the full-screen app: no GPU, no API calls, sessions in a temp folder.

  .venv\Scripts\python tests\test_tui.py

A fake index returns one fixed passage and a fake Claude stream returns a fixed Markdown answer; everything
else (prompt building, chat memory, session saving/resuming, the topic tree from the real database, search
display, key bindings) is the real code.
"""
import asyncio
import pathlib
import sys
import tempfile
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tests.fixture import use_fixture  # noqa: E402

use_fixture()                          # JMLR_TEST_FIXTURE=1: synthetic index (must precede importing common)

import ask  # noqa: E402
import conversation  # noqa: E402
import tui  # noqa: E402
from common import connect  # noqa: E402

conversation.SESSIONS_DIR = pathlib.Path(tempfile.mkdtemp())
KEY = "v13/gretton12a"


class FakeIndex:
    def __init__(self):
        self.con = connect()
        self.papers = {r["key"]: dict(r) for r in self.con.execute("SELECT * FROM papers")}

    def passages(self, q, k=12, **kw):
        return [dict(id=1, paper_key=KEY, seq=3, page_start=10, page_end=10, section="2 The MMD", is_ref=0,
                     text="The median heuristic sets the kernel bandwidth to the median distance.")]

    def rank_papers(self, q, k=25, **kw):
        return [(self.papers[KEY], self.passages(q)[0])]


ANSWER = "## Answer\n\nThe bandwidth is set by the **median heuristic** [1].\n\n| Paper | Rule |\n|---|---|\n| P1 | median |"
CALLS = []


def fake_stream(model_name, system, user_text, history=None, render=None):
    CALLS.append((model_name, len(history or [])))
    render(iter([ANSWER[i:i + 9] for i in range(0, len(ANSWER), 9)]))
    msg = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=ANSWER)], stop_reason="end_turn",
                                model=ask.MODELS[model_name],
                                usage=types.SimpleNamespace(input_tokens=1000, output_tokens=200))
    return msg, 0.0002


tui.load_index = lambda no_rewrite=False: FakeIndex()
ask.stream_claude = fake_stream
conversation.Conversation.condense = lambda self, q, client=None: (f"{q} (standalone)", 0.0, None)


async def settle(app, pilot):
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


async def main():
    app = tui.ResearchApp()
    async with app.run_test(size=(150, 45)) as pilot:
        for _ in range(50):
            if app.ix is not None:
                break
            await pilot.pause(0.1)
        assert app.ix is not None, "index never loaded"
        assert not app.query_one("#question").disabled

        # two chat turns: the second is a follow-up that gets the first as history
        for q in ("How do kernel two-sample tests choose the bandwidth?", "Which of those handles functional data?"):
            app.query_one("#question").value = q
            await pilot.press("enter")
            await settle(app, pilot)
        assert len(app.conv) == 2, len(app.conv)
        assert CALLS == [("haiku", 0), ("haiku", 2)], CALLS
        assert app.conv.turns[1].query.endswith("(standalone)")
        assert app.query_one("#sources").option_count == 1
        assert "median heuristic" in app.conv.turns[0].answer

        # saved automatically; Ctrl+N starts fresh; the session can be resumed from the Sessions tab
        saved = conversation.list_sessions()
        assert len(saved) == 1 and saved[0]["turns"] == 2, saved
        await pilot.press("ctrl+o")
        assert app.model_name == "opus"
        await pilot.press("ctrl+n")
        assert len(app.conv) == 0
        await pilot.press("f4")
        await pilot.pause()
        sessions = app.query_one("#session-list")
        assert sessions.option_count == 1
        sessions.highlighted = 0
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.conv) == 2 and app.query_one(tui.TabbedContent).active == "chat"
        app.save_screenshot(str(pathlib.Path(tempfile.gettempdir()) / "jmlr_tui_chat.svg"))

        # topic tree from the real database
        tree = app.query_one("#topic-tree")
        assert len(tree.root.children) == 18, len(tree.root.children)

        # search tab
        await pilot.press("f2")
        app.query_one("#search-q").value = "kernel two-sample test"
        app.query_one("#search-q").focus()
        await pilot.press("enter")
        await settle(app, pilot)
        assert app.query_one("#search-results").option_count == 1
        app.save_screenshot(str(pathlib.Path(tempfile.gettempdir()) / "jmlr_tui_search.svg"))
    print("All TUI checks passed: chat with memory, sources, autosave, new/resume session, model toggle, "
          "topic tree (18 areas), search.")


asyncio.run(main())
