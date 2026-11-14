"""Interactive topic map: every JMLR paper on a 2D map (t-SNE of its embedding), coloured by research area.

    map (ECharts scatter)               side panel (tabs)
    hover: title, year, subtopic        Areas  - the 18 areas with size + trend, expandable to subtopics
    wheel / drag: zoom, pan             Topic  - the selected area/subtopic: description, papers per year,
    legend: toggle areas                         top authors, most central papers
    click: paper -> Paper tab           Paper  - title, authors, abstract, Paper page / Open PDF
                                        Rising - subtopics that grew / shrank most (2020-25 vs 2005-19)
Selecting a topic highlights its papers and dims the rest; the year range hides papers outside it.
Deep links: /topics?topic=3.2, /topics?paper=v13/gretton12a, /topics?tab=rising,
/topics?years=2020-2025 (combinable).

The 2D coordinates are computed once (about a minute, CPU) and cached in data/topic_coords.json
(see gui/topicmap_data.py). The point data is sent to the browser once; later changes (selection, years)
are small partial `setOption` updates merged by series id.
"""
import html
import json

from nicegui import run, ui

from common import collection_info
from gui import topicmap_data as tmd
from gui.frame import PRIMARY
from gui.pdfview import show_pdf

DIM_OPACITY = 0.10
BASE_OPACITY = 0.78
LABEL_DIM = 0.35         # area labels while a topic is selected (they would hide its points)
PANEL_W = "w-[400px] min-w-[400px]"
BOX_W = "388px"          # panel content width (fixed: the scroll area would otherwise grow sideways)
SCROLL_PROPS = 'content-style="padding: 0; gap: 0"'
CSS = ".tm-panel .q-scrollarea__bar--h, .tm-panel .q-scrollarea__thumb--h { display: none; }"
MAP_H = "calc(100vh - 196px)"


def page():
    ui.add_css(CSS)
    params = _query_params()
    root = ui.column().classes("w-full gap-3")
    m = tmd.cached()
    if m is not None:
        with root:
            TopicMapView(m, params)
        return
    with root:
        with ui.card().classes("w-full items-center justify-center gap-3 py-16").style(f"height: {MAP_H}"):
            ui.spinner("dots", size="3em", color="primary")
            if not tmd.COORDS_FILE.exists():
                ui.label("Computing the map (one-time, about a minute)").classes("text-lg font-medium")
                ui.label("t-SNE of all paper embeddings on the CPU; later visits open instantly.").classes(
                    "text-sm text-gray-500")
            else:
                ui.label("Loading the map").classes("text-lg font-medium")
            status = ui.label().classes("text-xs text-gray-400")
            ticker = ui.timer(0.5, lambda: status.set_text(tmd.status()))

    async def load():
        try:
            data = await run.io_bound(tmd.load)
        except Exception as e:  # noqa: BLE001 - shown to the user
            ticker.cancel()
            root.clear()
            with root, ui.card().classes("w-full p-6"):
                ui.label("The topic map could not be built").classes("text-lg font-medium text-red-700")
                ui.label(f"{type(e).__name__}: {e}").classes("text-sm text-gray-600")
                ui.label("Build the topics first with: python topics.py build").classes("text-sm text-gray-500")
            return
        ticker.cancel()
        root.clear()
        with root:
            TopicMapView(data, params)

    ui.timer(0.1, load, once=True)


def _query_params():
    try:
        return dict(ui.context.client.request.query_params)
    except Exception:
        return {}


# ---------------------------------------------------------------- helpers

def _short(name, n=30):
    return name if len(name) <= n else name[:n - 1].rstrip() + "…"


def sparkline_svg(values, color, w=84, h=22):
    """A small area sparkline as inline SVG (values: any non-negative numbers)."""
    if not values:
        return ""
    top = max(values) or 1
    step = w / max(len(values) - 1, 1)
    pts = [(i * step, h - 2 - (v / top) * (h - 4)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{h} {line} {w:.1f},{h}"
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="display:block">'
            f'<polygon points="{area}" fill="{color}" fill-opacity="0.18"/>'
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.6" '
            f'stroke-linejoin="round" stroke-linecap="round"/></svg>')


def smooth(values, k=3):
    """Centered moving average (k odd), for readable per-year sparklines."""
    h = k // 2
    return [sum(values[max(0, i - h):i + h + 1]) / len(values[max(0, i - h):i + h + 1]) for i in range(len(values))]


def point(m, i):
    """One scatter data item: value [x, y, year]; name = title (HTML-escaped for the tooltip); k = paper index."""
    return {"value": [float(m.xy[i, 0]), float(m.xy[i, 1]), int(m.year[i])],
            "name": html.escape(m.title[i]), "k": int(i), "s": m.sub[i]}


def area_series_data(m, area, lo, hi):
    idx = m.members(area)
    keep = (m.year[idx] >= lo) & (m.year[idx] <= hi)
    return [point(m, i) for i in idx[keep]]


def highlight_data(m, tid, lo, hi):
    if not tid:
        return []
    idx = m.members(tid)
    keep = (m.year[idx] >= lo) & (m.year[idx] <= hi)
    return [point(m, i) for i in idx[keep]]


def tooltip_js(m):
    subs = {t: _short(r["name"], 60) for t, r in m.rows.items() if r["level"] == 2}
    return ("function(p){if(p.seriesType!=='scatter'&&p.seriesType!=='effectScatter')return '';"
            f"const S={json.dumps(subs)};const d=p.data||{{}};"
            "return '<div style=\"max-width:340px;white-space:normal\"><b>'+p.name+'</b><br/>'"
            "+'<span style=\"color:#6b7280\">'+p.value[2]+' &middot; '+d.s+' '+(S[d.s]||'')+'</span></div>';}")


def chart_options(m, lo, hi, sel=None, paper=None):
    series = []
    texts = [_short(m.rows[a]["name"], 28) for a in m.areas]
    if getattr(m, "_label_pos", None) is None:
        m._label_pos = m.label_positions(texts)
    lo_xy, hi_xy = m.bounds()
    for a, text, (cx, cy) in zip(m.areas, texts, m._label_pos):
        color = tmd.area_color(a)
        series.append({
            "id": f"a{a}", "name": m.label(a), "type": "scatter", "data": area_series_data(m, a, lo, hi),
            "symbolSize": 6, "itemStyle": {"color": color, "opacity": DIM_OPACITY if sel else BASE_OPACITY},
            "emphasis": {"scale": 2.2, "itemStyle": {"opacity": 1, "borderColor": "#111827", "borderWidth": 1}},
            "markPoint": {
                "silent": True, "symbol": "circle", "symbolSize": 0, "animation": False,
                "data": [{"coord": [cx, cy], "name": a}],
                "label": {"show": True, "formatter": text, "opacity": LABEL_DIM if sel else 1, "color": "#111827",
                          "fontSize": 11, "fontWeight": 600, "backgroundColor": "rgba(255,255,255,0.86)",
                          "borderColor": color, "borderWidth": 1.5, "borderRadius": 6, "padding": [3, 6]},
            },
        })
    series.append({"id": "hl", "name": "selection", "type": "scatter", "data": highlight_data(m, sel, lo, hi),
                   "symbolSize": 7, "z": 5,
                   "itemStyle": {"color": tmd.area_color(sel.split(".")[0]) if sel else PRIMARY, "opacity": 0.95, "borderColor": "#111827", "borderWidth": 0.6}})
    series.append({"id": "focus", "name": "paper", "type": "effectScatter",
                   "data": [point(m, paper)] if paper is not None else [], "symbolSize": 13, "z": 6,
                   "rippleEffect": {"scale": 2.6, "brushType": "stroke"},
                   "itemStyle": {"color": "#111827", "borderColor": "#ffffff", "borderWidth": 2}})
    return {
        "animation": False,
        "grid": {"left": 8, "right": 8, "top": 8, "bottom": 46},
        "xAxis": {"type": "value", "show": False, "min": lo_xy[0], "max": hi_xy[0]},
        "yAxis": {"type": "value", "show": False, "min": lo_xy[1], "max": hi_xy[1]},
        "dataZoom": [{"type": "inside", "xAxisIndex": 0, "filterMode": "none"},
                     {"type": "inside", "yAxisIndex": 0, "filterMode": "none"}],
        "tooltip": {"trigger": "item", "confine": True, "backgroundColor": "rgba(255,255,255,0.97)",
                    "borderColor": "#e5e7eb", "textStyle": {"color": "#111827", "fontSize": 12},
                    ":formatter": tooltip_js(m)},
        "toolbox": {"right": 10, "top": 6, "itemSize": 14,
                    "feature": {"restore": {"title": "Reset zoom"}, "saveAsImage": {"title": "Save image",
                                                                                    "name": "jmlr_topic_map"}}},
        "legend": {"type": "scroll", "bottom": 6, "left": 10, "right": 10, "itemWidth": 10, "itemHeight": 10,
                   "icon": "circle", "data": [m.label(a) for a in m.areas], "itemStyle": {"opacity": 1},
                   "textStyle": {"fontSize": 11, "color": "#374151", "width": 170, "overflow": "truncate"},
                   "pageIconColor": PRIMARY},
        "series": series,
    }


# ---------------------------------------------------------------- the page

class TopicMapView:
    def __init__(self, m, params=None):
        self.m = m
        self.lo, self.hi = m.years
        self.sel = None          # selected topic id ('3' or '3.2')
        self.paper = None        # selected paper index
        self.expanded = set()    # areas expanded in the browser
        params = params or {}
        try:                                       # /topics?years=2020-2025
            lo, hi = (int(y) for y in params["years"].split("-"))
            self.lo, self.hi = max(min(lo, hi), m.years[0]), min(max(lo, hi), m.years[1])
        except (KeyError, ValueError):
            pass
        if params.get("topic") in m.rows:          # deep links: /topics?topic=3.2  /topics?paper=v13/gretton12a
            self.sel = params["topic"]
            self.expanded.add(self.sel.split(".")[0])
        if params.get("paper") in m.index:
            self.paper = m.index[params["paper"]]
        self._build()
        if self.sel:
            self._show_chip()
        tab = params.get("tab")
        self.tabs.set_value(tab if tab in ("areas", "topic", "paper", "rising") else
                            "paper" if self.paper is not None else "topic" if self.sel else "areas")

    # -- layout
    def _build(self):
        m = self.m
        with ui.row().classes("w-full items-center justify-between gap-4 no-wrap"):
            with ui.column().classes("gap-0 min-w-0 shrink"):
                ui.label("Topic map").classes("text-xl font-semibold text-gray-900")
                ui.label(f"{len(m.keys):,} papers in {len(m.areas)} research areas and "
                         f"{sum(len(s) for s in m.subs.values())} subtopics. Scroll to zoom, drag to pan, "
                         "click a paper.").classes("text-sm text-gray-500 truncate w-full")
            with ui.row().classes("items-center gap-3 no-wrap shrink-0"):
                with ui.row().classes("items-center gap-1.5 pl-3 pr-1 py-0.5 rounded-full bg-indigo-50 no-wrap "
                                      "max-w-[320px]") as self.sel_chip:
                    self.sel_dot = ui.element("span").classes("inline-block w-2.5 h-2.5 rounded-full shrink-0")
                    self.sel_label = ui.label().classes("text-sm text-indigo-900 truncate")
                    ui.button(icon="close", on_click=self.clear_selection).props(
                        "flat round dense size=xs color=indigo-9").tooltip("Clear selection")
                self.sel_chip.set_visibility(False)
                ui.label("Years").classes("text-sm text-gray-600")
                self.range = ui.range(min=m.years[0], max=m.years[1], step=1,
                                      value={"min": self.lo, "max": self.hi}).props(
                    "label snap markers dense").classes("w-56")
                self.range.on("change", self._on_years)
                self.count = ui.label().classes("text-sm text-gray-600 w-44 whitespace-nowrap")
        with ui.row().classes("w-full gap-3 no-wrap items-stretch"):
            with ui.card().classes("grow p-0 overflow-hidden").style(f"height: {MAP_H}"):
                self.chart = ui.echart(chart_options(m, self.lo, self.hi, self.sel, self.paper), on_point_click=self._on_point).classes(
                    "w-full h-full")
                self.chart.view = self      # lets tests reach the view from the element
                # the canvas can be created before the flex layout has settled: re-measure once it has
                ui.timer(0.8, lambda: self.chart.run_chart_method("resize"), once=True)
            with ui.card().classes(f"{PANEL_W} p-0 gap-0").style(f"height: {MAP_H}"):
                with ui.tabs().props("dense no-caps align=justify active-color=primary indicator-color=primary") \
                        .classes("w-full text-gray-600 border-b") as self.tabs:
                    self.t_areas = ui.tab("areas", "Areas", icon="category")
                    self.t_topic = ui.tab("topic", "Topic", icon="bubble_chart")
                    self.t_paper = ui.tab("paper", "Paper", icon="article")
                    self.t_rising = ui.tab("rising", "Rising", icon="trending_up")
                with ui.tab_panels(self.tabs, value="areas", animated=False).classes("w-full grow min-h-0"):
                    with ui.tab_panel("areas").classes("p-0"):
                        with ui.scroll_area().classes("w-full h-full tm-panel").props(SCROLL_PROPS):
                            self.areas_box = ui.column().classes("gap-0 py-1 px-1").style(f"width: {BOX_W}")
                    with ui.tab_panel("topic").classes("p-0"):
                        with ui.scroll_area().classes("w-full h-full tm-panel").props(SCROLL_PROPS):
                            self.topic_box = ui.column().classes("gap-3 p-4").style(f"width: {BOX_W}")
                    with ui.tab_panel("paper").classes("p-0"):
                        with ui.scroll_area().classes("w-full h-full tm-panel").props(SCROLL_PROPS):
                            self.paper_box = ui.column().classes("gap-3 p-4").style(f"width: {BOX_W}")
                    with ui.tab_panel("rising").classes("p-0"):
                        with ui.scroll_area().classes("w-full h-full tm-panel").props(SCROLL_PROPS):
                            self.rising_box = ui.column().classes("gap-1 p-3").style(f"width: {BOX_W}")
        self._render_areas()
        self._render_topic()
        self._render_paper()
        self._render_rising()
        self._update_count()

    # -- area browser
    def _render_areas(self):
        m = self.m
        self.areas_box.clear()
        with self.areas_box:
            for a in m.areas:
                color = tmd.area_color(a)
                _, counts, shares = m.per_year(a)
                self._topic_row(a, m.rows[a]["name"], m.size_primary[a], smooth(shares), color, level=1)
                if a in self.expanded:
                    for s in m.subs[a]:
                        _, _, sh = m.per_year(s)
                        self._topic_row(s, m.rows[s]["name"], m.size_primary[s], smooth(sh), color, level=2)

    def _topic_row(self, tid, name, size, spark, color, level):
        selected = tid == self.sel
        bg = "bg-indigo-50 " if selected else "hover:bg-gray-50 "
        with ui.row().classes("w-full items-center gap-0 no-wrap rounded-lg " + bg + ("pl-1" if level == 1 else "pl-9")):
            if level == 1:                       # the chevron sits outside the clickable part of the row
                ui.button(icon="expand_more" if tid in self.expanded else "chevron_right",
                          on_click=lambda t=tid: self.toggle_expand(t)).props(
                    "flat round dense size=sm color=grey-7").tooltip("Subtopics")
            row = ui.row().classes("grow min-w-0 items-center gap-2 no-wrap cursor-pointer pr-2 "
                                   + ("py-2 pl-1" if level == 1 else "py-1"))
            row.mark(f"topic-row-{tid}").on("click", lambda t=tid: self.select(t))
        with row:
            if level == 1:
                ui.element("span").classes("inline-block w-2.5 h-2.5 rounded-full shrink-0").style(
                    f"background:{color}")
            else:
                ui.label(tid).classes("text-xs text-gray-400 w-8 shrink-0")
            with ui.column().classes("gap-0 grow min-w-0"):
                ui.label(name).classes(
                    ("text-sm font-medium text-gray-900" if level == 1 else "text-[13px] text-gray-700")
                    + " truncate w-full").tooltip(name)
            ui.label(f"{size}").classes("text-xs text-gray-500 w-8 text-right shrink-0")
            ui.html(sparkline_svg(spark, color, w=64 if level == 2 else 76, h=18 if level == 2 else 22),
                    sanitize=False).classes("shrink-0").tooltip("share of each year's papers, {}-{}".format(*collection_info()[1:]))

    def toggle_expand(self, area):
        self.expanded ^= {area}
        self._render_areas()

    # -- selection
    def select(self, tid):
        m = self.m
        if tid == self.sel:
            self.tabs.set_value("topic")
            return
        self.sel = tid
        area = tid.split(".")[0]
        self.expanded.add(area)
        self._apply()
        self._render_areas()
        self._render_topic()
        self._show_chip()
        self.tabs.set_value("topic")

    def _show_chip(self):
        self.sel_label.set_text(self.m.label(self.sel))
        self.sel_label._props["title"] = self.m.label(self.sel)      # native hover text for the truncated name
        self.sel_dot.style(f"background:{tmd.area_color(self.sel.split('.')[0])}")
        self.sel_chip.set_visibility(True)

    def clear_selection(self):
        self.sel = None
        self._apply()
        self._render_areas()
        self._render_topic()
        self.sel_chip.set_visibility(False)
        self.tabs.set_value("areas")

    def _apply(self, data_changed=False):
        """Push the selection / year filter to the chart as a partial update (merged by series id)."""
        m = self.m
        series = []
        for a in m.areas:
            s = {"id": f"a{a}", "itemStyle": {"opacity": DIM_OPACITY if self.sel else BASE_OPACITY},
                 "markPoint": {"label": {"opacity": LABEL_DIM if self.sel else 1}}}
            if data_changed:
                s["data"] = area_series_data(m, a, self.lo, self.hi)
            series.append(s)
        color = tmd.area_color(self.sel.split(".")[0]) if self.sel else PRIMARY
        series.append({"id": "hl", "data": highlight_data(m, self.sel, self.lo, self.hi),
                       "itemStyle": {"color": color}})
        series.append({"id": "focus", "data": [point(m, self.paper)] if self.paper is not None else []})
        self.chart.run_chart_method("setOption", {"series": series})
        self.last_update = series     # kept for tests

    def _on_years(self, _=None):
        v = self.range.value or {}
        self.set_years(int(v.get("min", self.m.years[0])), int(v.get("max", self.m.years[1])))

    def set_years(self, lo, hi):
        self.lo, self.hi = min(lo, hi), max(lo, hi)
        self._apply(data_changed=True)
        self._update_count()

    def _update_count(self):
        n = int(self.m.in_years(self.lo, self.hi).sum())
        self.count.set_text(f"{self.lo}-{self.hi}: {n:,} papers")

    # -- paper
    def _on_point(self, e):
        data = e.data if isinstance(e.data, dict) else {}
        if "k" in data:
            self.show_paper(int(data["k"]))

    def show_paper(self, i):
        self.paper = int(i)
        self._apply()
        self._render_paper()
        self.tabs.set_value("paper")

    def _render_paper(self):
        m, box = self.m, self.paper_box
        box.clear()
        with box:
            if self.paper is None:
                _empty("touch_app", "Click a paper on the map", "Its title, authors and abstract appear here.")
                return
            i = self.paper
            key, sub = m.keys[i], m.sub[i]
            area = sub.split(".")[0]
            ui.label(m.title[i]).classes("text-base font-semibold text-gray-900 leading-snug")
            ui.label(f"{int(m.year[i])}  ·  {m.authors[i]}").classes("text-sm text-gray-600")
            with ui.row().classes("gap-1"):
                _topic_chip(self, sub, tmd.area_color(area))
                sec = m.secondary.get(key)
                if sec and sec in m.rows:
                    _topic_chip(self, sec, tmd.area_color(sec.split(".")[0]), secondary=True)
            text = tmd.abstract(key).strip()
            if text:
                ui.label(text if len(text) <= 900 else text[:900].rsplit(" ", 1)[0] + " …").classes(
                    "text-sm text-gray-700 leading-relaxed")
            with ui.row().classes("gap-2 pt-1"):
                ui.button("Paper page", icon="article", on_click=lambda: ui.navigate.to(f"/paper/{key}")).props(
                    "unelevated no-caps")
                ui.button("Open PDF", icon="picture_as_pdf", on_click=lambda: show_pdf(key)).props(
                    "outline no-caps")
            ui.label(key).classes("text-xs text-gray-400")

    # -- topic detail
    def _render_topic(self):
        m, box = self.m, self.topic_box
        box.clear()
        with box:
            if not self.sel:
                _empty("bubble_chart", "Select an area or subtopic",
                       "Pick one in the Areas or Rising tab to highlight its papers on the map.")
                return
            tid = self.sel
            r = m.rows[tid]
            area = tid.split(".")[0]
            color = tmd.area_color(area)
            with ui.row().classes("items-center gap-2 no-wrap w-full"):
                ui.element("span").classes("inline-block w-3 h-3 rounded-full shrink-0").style(f"background:{color}")
                if "." in tid:
                    ui.link(m.label(area), "#").classes("text-xs text-gray-500 no-underline hover:underline").on(
                        "click.prevent", lambda: self.select(area))
                else:
                    ui.label(f"Research area {tid}").classes("text-xs text-gray-500")
                ui.space()
                ui.button(icon="close", on_click=self.clear_selection).props("flat round dense size=sm").tooltip(
                    "Clear selection")
            ui.label(f"{tid}. {r['name']}").classes("text-lg font-semibold text-gray-900 leading-snug -mt-2")
            members = m.members(tid)
            recent = int(m.in_years(*tmd.RECENT)[members].sum())
            with ui.row().classes("gap-2"):
                _stat(f"{len(members)}", "papers")
                _stat(f"{recent}", "since 2020")
                yrs = m.year[members]
                _stat(f"{int(yrs.min())}-{int(yrs.max())}" if len(yrs) else "-", "years")
            if r["description"]:
                ui.label(r["description"]).classes("text-sm text-gray-700 leading-relaxed")
            try:
                terms = json.loads(r["terms"] or "[]")[:10]
            except ValueError:
                terms = []
            if terms:
                with ui.row().classes("gap-1"):
                    for t in terms:
                        ui.label(t).classes("text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-700")
            ys, counts, shares = m.per_year(tid)
            _section("Papers per year")
            ui.echart({
                "animation": False,
                "grid": {"left": 30, "right": 8, "top": 10, "bottom": 22},
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"},
                            ":formatter": "function(p){p=p[0];return p.name+': <b>'+p.value+'</b> papers';}"},
                "xAxis": {"type": "category", "data": [str(y) for y in ys], "axisTick": {"show": False},
                          "axisLine": {"lineStyle": {"color": "#d1d5db"}},
                          "axisLabel": {"color": "#6b7280", "fontSize": 10, "interval": 4}},
                "yAxis": {"type": "value", "minInterval": 1, "splitNumber": 3,
                          "splitLine": {"lineStyle": {"color": "#f3f4f6"}},
                          "axisLabel": {"color": "#9ca3af", "fontSize": 10}},
                "series": [{"type": "bar", "data": counts, "barWidth": "62%",
                            "itemStyle": {"color": color, "borderRadius": [3, 3, 0, 0]}}],
            }).classes("w-full h-36")
            if "." not in tid:
                _section("Subtopics")
                with ui.column().classes("w-full gap-0"):
                    for s in m.subs[tid]:
                        with ui.row().classes("w-full items-center gap-2 no-wrap cursor-pointer rounded px-1 py-1 "
                                              "hover:bg-gray-50").on("click", lambda t=s: self.select(t)):
                            ui.label(s).classes("text-xs text-gray-400 w-8 shrink-0")
                            ui.label(m.rows[s]["name"]).classes("text-[13px] text-gray-800 grow truncate")
                            ui.label(str(m.size_primary[s])).classes("text-xs text-gray-500 shrink-0")
            _section("Top authors")
            with ui.column().classes("w-full gap-0.5"):
                authors = m.top_authors(tid, 8)
                top = authors[0][1] if authors else 1
                for name, k in authors:
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        ui.label(name).classes("text-[13px] text-gray-800 w-40 truncate shrink-0")
                        with ui.element("div").classes("grow h-1.5 rounded-full bg-gray-100"):
                            ui.element("div").classes("h-1.5 rounded-full").style(
                                f"width:{100 * k / top:.0f}%;background:{color}")
                        ui.label(str(k)).classes("text-xs text-gray-500 w-5 text-right shrink-0")
            _section("Most central papers")
            with ui.column().classes("w-full gap-0"):
                for i in m.central(tid, 10):
                    with ui.row().classes("w-full items-start gap-2 no-wrap cursor-pointer rounded px-1 py-1.5 "
                                          "hover:bg-indigo-50").on("click", lambda i=i: self.show_paper(i)):
                        ui.label(str(int(m.year[i]))).classes("text-xs text-gray-400 w-9 shrink-0 pt-0.5")
                        ui.label(m.title[i]).classes("text-[13px] text-gray-800 leading-snug")

    # -- rising / fading
    def _render_rising(self):
        m, box = self.m, self.rising_box
        box.clear()
        rising, fading = m.rising(8)
        with box:
            ui.label("Share of each period's papers: 2005-19 vs 2020-25 (subtopics with at least 20 papers)."
                     ).classes("text-xs text-gray-500 px-1 pb-1")
            for title, icon, items, tone in (("Rising", "trending_up", rising, "text-emerald-700"),
                                             ("Fading", "trending_down", fading, "text-rose-700")):
                with ui.row().classes(f"items-center gap-1 px-1 pt-2 {tone}"):
                    ui.icon(icon).classes("text-lg")
                    ui.label(title).classes("text-sm font-semibold")
                for tid, before, recent, ratio in items:
                    color = tmd.area_color(tid.split(".")[0])
                    with ui.row().classes("w-full items-center gap-2 no-wrap cursor-pointer rounded-lg px-2 py-1.5 "
                                          "hover:bg-gray-50").mark(f"rising-{tid}").on(
                            "click", lambda t=tid: self.select(t)):
                        ui.element("span").classes("inline-block w-2 h-2 rounded-full shrink-0").style(
                            f"background:{color}")
                        with ui.column().classes("gap-0 grow min-w-0"):
                            ui.label(f"{tid} {m.rows[tid]['name']}").classes("text-[13px] text-gray-800 truncate w-full")
                            ui.label(f"{before:.1%} → {recent:.1%}").classes("text-xs text-gray-500")
                        ui.label(f"×{ratio:.1f}").classes(f"text-sm font-medium shrink-0 {tone}")


def _empty(icon, title, text):
    with ui.column().classes("w-full items-center text-center gap-1 py-16"):
        ui.icon(icon).classes("text-4xl text-gray-300")
        ui.label(title).classes("text-sm font-medium text-gray-700")
        ui.label(text).classes("text-xs text-gray-500")


def _section(title):
    ui.label(title).classes("text-xs font-semibold uppercase tracking-wide text-gray-500 pt-2")


def _stat(value, label):
    with ui.column().classes("gap-0 px-3 py-1.5 rounded-lg bg-gray-50 border border-gray-100"):
        ui.label(value).classes("text-base font-semibold text-gray-900 leading-tight")
        ui.label(label).classes("text-[11px] text-gray-500 leading-tight")


def _topic_chip(view, tid, color, secondary=False):
    label = f"{tid} {_short(view.m.rows[tid]['name'], 38)}" + ("  (2nd)" if secondary else "")
    with ui.row().classes("items-center gap-1.5 px-2 py-0.5 rounded-full border border-gray-200 cursor-pointer hover:bg-gray-50 "
                          "no-wrap").on("click", lambda: view.select(tid)):
        ui.element("span").classes("inline-block w-2 h-2 rounded-full").style(f"background:{color}")
        ui.label(label).classes("text-xs text-gray-700")
