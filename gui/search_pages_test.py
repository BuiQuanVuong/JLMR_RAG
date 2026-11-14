"""Tests for the Search and Paper pages (NiceGUI user simulation, fake index: no models, no GPU, no API).

    python -m pytest gui/search_pages_test.py -q -p no:cacheprovider -p nicegui.testing.user_plugin -o asyncio_mode=auto -o main_file=
"""
import asyncio

import pytest
from nicegui import ui
from nicegui.testing.user_simulation import user_simulation

from gui import search_data as sd
from gui import state

KEY = "v13/gretton12a"
MAIN = """
import importlib
import gui.app
importlib.reload(gui.app)      # re-register the @ui.page routes inside the simulation's fresh globals
gui.app.ui.run(reload=False)
"""


class FakeIndex:
    """Ranks papers by how many query words their title + abstract contain."""

    def __init__(self):
        self.con = sd.db()
        self.papers = {r["key"]: dict(r) for r in self.con.execute("SELECT * FROM papers")}
        self.calls = []

    def rank_papers(self, q, k=10, since=None, until=None, allowed=None):
        self.calls.append(dict(q=q, k=k, since=since, until=until, allowed=allowed))
        words = [w for w in q.lower().split() if len(w) > 3]
        keys = [key for key, p in self.papers.items() if (allowed is None or key in allowed)
                and (since is None or (p["year"] or 0) >= since) and (until is None or (p["year"] or 0) <= until)]
        keys.sort(key=lambda key: -sum(w in (self.papers[key]["title"] + " " + (self.papers[key]["abstract"] or ""))
                                       .lower() for w in words))
        return [(self.papers[key], dict(paper_key=key, page_start=10, section="2 The MMD / 2.1 Definition",
                                        text="The maximum mean discrepancy is a kernel two-sample test statistic."))
                for key in keys[:k]]

    def similar(self, key, k=10):
        return [(p, 0.5) for p in list(self.papers.values())[:k] if p["key"] != key][:k]


@pytest.fixture
def fake():
    ix = FakeIndex()
    old = state.STATE["index"]
    state.STATE["index"] = ix
    yield ix
    state.STATE["index"] = old


@pytest.fixture
async def user(tmp_path):
    main = tmp_path / "main.py"
    main.write_text(MAIN, encoding="utf-8")
    async with user_simulation(main_file=main) as u:
        yield u


# ---------------------------------------------------------------- pure helpers

def test_outline_has_sections_in_order_with_first_pages():
    items = sd.outline(KEY)
    labels = [label for _, label, _, _ in items]
    assert labels[:2] == ["Abstract", "1 Introduction"]
    assert (1, "2.1 Deﬁnition of the Maximum Mean Discrepancy", 3, False) in items
    assert items[-1][1] == "References" and items[-1][3] is True
    assert len(labels) == len(set((d, lbl) for d, lbl, _, _ in items))


def test_topics_and_areas():
    t = sd.topics()
    assert len(t.areas()) == 18
    assert t.primary[KEY] == "2.6" and t.area_of(KEY) == "2"
    assert KEY in t.papers_in_area("2")


def test_snippet_escapes_and_marks():
    out = sd.snippet_html("a <b> kernel test & more", "kernel test")
    assert "&lt;b&gt;" in out and "<mark>kernel</mark>" in out and "&amp;" in out


def test_boolean_search_with_area_filter(fake):
    res = sd.search(fake, 'intitle:"two-sample" kernel author:Gretton', k=10)
    assert res.boolean and res.n_strict >= 1 and any(p["key"] == KEY for p, _, _ in res.rows)
    assert [c[0] for c in res.counts] == ['intitle:"two sample"', "kernel", "author:Gretton"]
    res = sd.search(fake, 'intitle:"two-sample" kernel author:Gretton', k=10, area="16")   # bandits: none
    assert not res.rows and "none within the selected filters" in res.note
    res = sd.search(fake, "intitle:gretton kernel", k=5)                                    # no strict match
    assert res.rows and all(miss == ["intitle:gretton"] for _, _, miss in res.rows)


def test_plain_search_passes_filters(fake):
    sd.search(fake, "kernel test", k=7, since=2000, until=2010, area="12")
    call = fake.calls[-1]
    assert call["k"] == 7 and call["since"] is None and call["until"] == 2010
    assert call["allowed"] == sd.topics().papers_in_area("12")


# ---------------------------------------------------------------- pages

async def test_search_page_from_url_and_enter(user, fake):
    await user.open("/search?q=kernel%20two-sample%20test")
    await user.should_see("A Kernel Two-Sample Test", retries=50)
    await user.should_see("Open PDF at p. 10")
    user.find(ui.input).clear().type("maximum mean discrepancy").trigger("keydown.enter")
    for _ in range(50):
        if fake.calls and fake.calls[-1]["q"] == "maximum mean discrepancy":
            break
        await asyncio.sleep(0.05)
    assert fake.calls[-1]["q"] == "maximum mean discrepancy"


async def test_search_page_boolean_summary(user, fake):
    await user.open("/search?q=" + 'intitle:"two-sample" kernel author:Gretton'.replace(" ", "%20"))
    await user.should_see("Scholar-style query", retries=50)
    await user.should_see("match every term")
    await user.should_see("A Kernel Two-Sample Test")


async def test_search_page_empty_and_loading(user):
    old = state.STATE["index"]
    state.STATE["index"] = None
    try:
        await user.open("/search")
        await user.should_see("Search the JMLR archive", retries=20)
        await user.open("/search?q=kernel")
        await user.should_see("Loading the search index", retries=20)
    finally:
        state.STATE["index"] = old


async def test_paper_page(user, fake):
    await user.open(f"/paper/{KEY}")
    await user.should_see("A Kernel Two-Sample Test")
    await user.should_see("2.6 Kernel-based Two-Sample and Independence Testing")
    await user.should_see("Sections")
    await user.should_see("1 Introduction")
    await user.should_see("Similar papers")
    await user.should_see(fake.similar(KEY, 8)[0][0]["title"], retries=50)


async def test_paper_not_found(user):
    await user.open("/paper/v99/nope")
    await user.should_see("Paper not found")
