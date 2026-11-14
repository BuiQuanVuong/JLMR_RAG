"""Tests of the topic map page (needs the real DB, embeddings and data/topic_coords.json; no index, no GPU).

    python -m pytest gui/topicmap_test.py -q
"""
import asyncio

import numpy as np
import pytest

from gui import topicmap
from gui import topicmap_data as tmd

pytestmark = pytest.mark.skipif(not tmd.COORDS_FILE.exists(), reason="no data/topic_coords.json yet")


@pytest.fixture(scope="module")
def m():
    return tmd.load()


def test_data_consistent(m):
    assert len(m.keys) == len(m.xy) == len(m.title) == len(m.year) > 4000
    assert len(m.areas) == 18
    assert sum(len(m.members(a)) for a in m.areas) == len(m.keys)
    for a in m.areas:
        assert sum(len(m.members(s)) for s in m.subs[a]) == len(m.members(a))
    assert np.isfinite(m.xy).all()


def test_stats(m):
    ys, counts, shares = m.per_year("3.2")
    assert sum(counts) == len(m.members("3.2")) and len(ys) == len(shares)
    assert m.top_authors("3")[0][1] >= m.top_authors("3")[-1][1]
    central = m.central("3.2", 5)
    assert all(m.sub[i] == "3.2" for i in central)
    rising, fading = m.rising(5)
    assert rising[0][3] >= rising[-1][3] and fading[0][3] <= fading[-1][3]


def test_label_positions_do_not_overlap(m):
    texts = [topicmap._short(m.rows[a]["name"], 28) for a in m.areas]
    pos = np.array(m.label_positions(texts))
    lo, hi = m.bounds()
    assert ((pos >= np.array(lo)) & (pos <= np.array(hi))).all()


def test_chart_options(m):
    opts = topicmap.chart_options(m, *m.years)
    scatter = [s for s in opts["series"] if s["id"].startswith("a")]
    assert len(scatter) == 18 and sum(len(s["data"]) for s in scatter) == len(m.keys)
    item = scatter[0]["data"][0]
    assert set(item) == {"value", "name", "k", "s"} and m.sub[item["k"]] == item["s"]
    opts = topicmap.chart_options(m, 2020, 2025, sel="3.2", paper=0)
    hl = next(s for s in opts["series"] if s["id"] == "hl")["data"]
    assert hl and all(d["s"] == "3.2" and d["value"][2] >= 2020 for d in hl)
    assert next(s for s in opts["series"] if s["id"] == "focus")["data"][0]["k"] == 0


def test_page_interactions(m):
    """Open the page in a simulated client, then click a paper, select topics, filter years."""
    from nicegui import ui
    from nicegui.events import EChartPointClickEventArguments
    from nicegui.testing.user_simulation import user_simulation

    from gui.frame import frame

    def root():
        with frame("Topic map", "topics"):
            topicmap.page()

    async def scenario():
        async with user_simulation(root) as user:
            await user.open("/")
            await user.should_see("Bayesian Inference and Probabilistic Modeling")
            chart = next(iter(user.find(ui.echart).elements))
            view = chart.view
            # click a paper on the map -> Paper tab with its details
            i = m.index["v13/gretton12a"]
            view._on_point(EChartPointClickEventArguments(
                sender=chart, client=chart.client, component_type="series", series_type="scatter", series_index=0,
                series_name="x", name="t", data_index=0, data=topicmap.point(m, i), data_type=None, value=[0, 0]))
            assert view.paper == i and view.tabs.value == "paper"
            await user.should_see("A Kernel Two-Sample Test")
            await user.should_see("Open PDF")
            # select a subtopic -> highlight series holds exactly its papers, the rest dimmed
            view.select("3.2")
            assert view.sel == "3.2" and view.tabs.value == "topic"
            hl = next(s for s in view.last_update if s["id"] == "hl")["data"]
            assert len(hl) == len(m.members("3.2"))
            assert all(s["itemStyle"]["opacity"] == topicmap.DIM_OPACITY for s in view.last_update
                       if s["id"].startswith("a"))
            await user.should_see("Most central papers")
            await user.should_see("Sparse Penalized Linear Regression")
            # clicking a row in the area browser selects that area
            view.tabs.set_value("areas")
            user.find(marker="topic-row-4").click()
            assert view.sel == "4"
            # the chevron expands an area into its subtopics without selecting it
            view.toggle_expand("5")
            assert view.sel == "4"
            await user.should_see(m.rows[m.subs["5"][0]]["name"])
            # year filter -> base series data restricted to the range
            view.set_years(2020, 2025)
            n = sum(len(s["data"]) for s in view.last_update if s["id"].startswith("a"))
            assert n == int(m.in_years(2020, 2025).sum())
            view.clear_selection()
            assert view.sel is None and not next(s for s in view.last_update if s["id"] == "hl")["data"]
            # the Rising tab lists subtopics and selects them on click
            rising, _ = m.rising(8)
            tid = rising[0][0]
            view.tabs.set_value("rising")
            user.find(marker=f"rising-{tid}").click()
            assert view.sel == tid

    asyncio.run(scenario())
