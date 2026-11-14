"""Recent advances dashboard (/advances): KPIs, momentum and growth charts, and one card per subtopic with a
Claude-written headline and cited "advances" bullets. Reads data/advances.json, made by `python advances.py`.
"""
import json
import math

from nicegui import ui

from common import DATA_DIR, connect

DATA_FILE = DATA_DIR / "advances.json"
PRIMARY = "#3b5bdb"
UP, DOWN, NEUTRAL = "#3b5bdb", "#e8590c", "#adb5bd"
AREA_COLORS = ["#3b5bdb", "#0ca678", "#f59f00", "#e64980", "#7048e8", "#1098ad", "#d9480f", "#868e96"]

_cache = {"mtime": None, "data": None}


def load():
    """data/advances.json (cached until the file changes), or None."""
    try:
        mtime = DATA_FILE.stat().st_mtime
    except OSError:
        return None
    if _cache["mtime"] != mtime:
        try:
            _cache["data"] = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            _cache["mtime"] = mtime
        except Exception:
            return None
    return _cache["data"]


def _paper_meta(keys):
    if not keys:
        return {}
    keys = list(keys)
    con = connect()
    out = {}
    for i in range(0, len(keys), 500):
        part = keys[i:i + 500]
        q = f"SELECT key, title, year FROM papers WHERE key IN ({','.join('?' * len(part))})"
        out.update({r["key"]: (r["title"], r["year"]) for r in con.execute(q, part)})
    return out


def _fmt_m(m):
    return f"{m:.1f}x" if m < 10 else f"{m:.0f}x"


def _short(name, n=34):
    return name if len(name) <= n else name[:n - 1].rstrip() + "…"


# ---------------------------------------------------------------- small building blocks

def _kpi(icon, label, value, sub="", color="indigo"):
    with ui.card().props("flat bordered").classes("rounded-xl p-4 gap-1 grow basis-56 bg-white"):
        with ui.row().classes("items-center gap-2 no-wrap"):
            with ui.element("div").classes(f"rounded-lg bg-{color}-50 p-1.5 flex"):
                ui.icon(icon).classes(f"text-{color}-700 text-xl")
            ui.label(label).classes("text-xs uppercase tracking-wide text-gray-500 font-medium")
        ui.label(value).classes("text-2xl font-semibold text-gray-900 leading-tight mt-1")
        if sub:
            ui.label(sub).classes("text-xs text-gray-500")


def _momentum_badge(s):
    m = s["momentum"]
    if s.get("momentum_low_base") and s["recent"] > 0:
        cls, text, tip = "bg-indigo-50 text-indigo-700", f"new · {_fmt_m(m)}", \
            "Few papers before 2023, so the ratio is large but uncertain"
    elif m >= 1.15:
        cls, text, tip = "bg-indigo-50 text-indigo-700", f"▲ {_fmt_m(m)}", "Growing share of JMLR papers"
    elif m <= 0.87:
        cls, text, tip = "bg-orange-50 text-orange-700", f"▼ {_fmt_m(m)}", "Shrinking share of JMLR papers"
    else:
        cls, text, tip = "bg-gray-100 text-gray-600", f"● {_fmt_m(m)}", "Roughly stable share"
    lab = ui.label(text).classes(f"text-xs font-semibold px-2 py-0.5 rounded-full whitespace-nowrap {cls}")
    lab.tooltip(f"Momentum {m:.2f}: share of papers 2023-25 vs 2010-22. {tip}.")


def _sparkline(per_year, years, recent_from=2023, w=220, h=34):
    """Inline SVG bars of papers per year; recent years in the primary colour."""
    n, top = len(per_year), max(max(per_year), 1)
    bw = w / n
    bars = []
    for i, (v, y) in enumerate(zip(per_year, years)):
        bh = 0 if v == 0 else max(2.0, (h - 2) * v / top)
        col = PRIMARY if y >= recent_from else "#c5cbe3"
        bars.append(f'<rect x="{i * bw + 0.6:.1f}" y="{h - bh:.1f}" width="{bw - 1.2:.1f}" height="{bh:.1f}" '
                    f'rx="1" fill="{col}"><title>{y}: {v}</title></rect>')
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="0" y1="{h - 0.5}" x2="{w}" y2="{h - 0.5}" stroke="#e9ecef"/>' + "".join(bars) + "</svg>")


# ---------------------------------------------------------------- charts

def _momentum_options(items, title_names):
    """Horizontal diverging bars of log2(momentum) so that x0.5 and x2 are equally long."""
    items = sorted(items, key=lambda s: s["momentum"])
    names = [title_names(s) for s in items]
    data = []
    for s in items:
        v = math.log2(max(s["momentum"], 0.05))
        data.append({"value": round(v, 3), "itemStyle": {"color": UP if s["momentum"] >= 1 else DOWN,
                                                         "borderRadius": [0, 4, 4, 0] if v >= 0 else [4, 0, 0, 4]},
                     "id": s["id"], "m": s["momentum"], "recent": s["recent"], "full": s["name"]})
    return {
        "grid": {"left": 8, "right": 24, "top": 8, "bottom": 24, "containLabel": True},
        "tooltip": {"trigger": "item", ":formatter":
                    "p => `<b>${p.data.full}</b><br/>momentum ×${p.data.m.toFixed(2)}<br/>${p.data.recent} papers 2023-25`"},
        "xAxis": {"type": "value", "splitLine": {"lineStyle": {"color": "#f1f3f5"}},
                  "axisLabel": {"color": "#868e96", ":formatter": "v => '×' + +Math.pow(2, v).toFixed(2)"}},
        "yAxis": {"type": "category", "data": names, "axisTick": {"show": False},
                  "axisLine": {"lineStyle": {"color": "#dee2e6"}},
                  "axisLabel": {"color": "#495057", "fontSize": 11}},
        "series": [{"type": "bar", "data": data, "barMaxWidth": 14,
                    "markLine": {"silent": True, "symbol": "none", "data": [{"xAxis": 0}],
                                 "lineStyle": {"color": "#495057", "type": "solid", "width": 1},
                                 "label": {"show": False}}}],
    }


def _per_year_options(data, top_n=6):
    years, areas = data["years"], data["areas"]
    start = years.index(2005) if 2005 in years else 0
    ranked = sorted(areas.values(), key=lambda a: -a["recent"])
    top, rest = ranked[:top_n], ranked[top_n:]
    series = []
    for i, a in enumerate(top):
        series.append({"name": _short(a["name"], 26), "type": "line", "stack": "all", "smooth": 0.3,
                       "symbol": "none", "lineStyle": {"width": 1}, "areaStyle": {"opacity": 0.85},
                       "color": AREA_COLORS[i], "emphasis": {"focus": "series"},
                       "data": a["per_year"][start:]})
    other = [sum(a["per_year"][j] for a in rest) for j in range(len(years))]
    series.append({"name": "Other areas", "type": "line", "stack": "all", "smooth": 0.3, "symbol": "none",
                   "lineStyle": {"width": 1}, "areaStyle": {"opacity": 0.6}, "color": "#ced4da",
                   "emphasis": {"focus": "series"}, "data": other[start:]})
    return {
        "grid": {"left": 8, "right": 16, "top": 80, "bottom": 8, "containLabel": True},
        "legend": {"type": "plain", "top": 0, "left": 0, "right": 0, "icon": "roundRect", "itemWidth": 10,
                   "itemHeight": 10, "textStyle": {"fontSize": 11, "color": "#495057"}},
        "tooltip": {"trigger": "axis", "order": "seriesDesc"},
        "xAxis": {"type": "category", "boundaryGap": False, "data": [str(y) for y in years[start:]],
                  "axisLabel": {"color": "#868e96"}, "axisLine": {"lineStyle": {"color": "#dee2e6"}}},
        "yAxis": {"type": "value", "splitLine": {"lineStyle": {"color": "#f1f3f5"}}, "axisLabel": {"color": "#868e96"}},
        "series": series,
    }


# ---------------------------------------------------------------- page

def _missing():
    with ui.card().props("flat bordered").classes("rounded-xl p-8 w-full items-center text-center gap-3 bg-white"):
        ui.icon("insights", color="primary").classes("text-5xl")
        ui.label("No recent-advances data yet").classes("text-xl font-semibold")
        ui.label("Generate it once from the project folder (about a minute, ~$0.05-0.10 of Claude Haiku):") \
            .classes("text-gray-600")
        ui.code("python advances.py", language="bash").classes("text-sm")
        ui.label("or `python advances.py --dry-run` for the statistics only, without API calls. "
                 "Then reload this page.").classes("text-xs text-gray-500")
        ui.button("Reload", icon="refresh", on_click=lambda: ui.navigate.reload()).props("unelevated no-caps")


def page():
    data = load()
    if not data:
        _missing()
        return
    areas, subs, summaries = data["areas"], data["subtopics"], data.get("summaries", {})
    gen = data.get("generation") or {}
    years = data["years"]
    recent_a, recent_b = data["windows"]["recent"]
    cited = {k for s in summaries.values() for adv in s.get("advances", []) for k in adv["keys"]}
    meta = _paper_meta(cited)

    # ---- KPI row
    eligible = [s for s in subs.values() if s["base"] >= 3 and s["recent"] >= 5]
    fastest = max(eligible or subs.values(), key=lambda s: s["momentum"])
    total_recent = sum(a["recent"] for a in areas.values())
    with_recent = sum(1 for s in subs.values() if s["recent"] > 0)
    rising = sum(1 for s in subs.values() if s["momentum"] >= 1)
    with ui.row().classes("w-full gap-4 items-stretch"):
        _kpi("article", f"Recent papers {recent_a}-{recent_b % 100}", f"{total_recent:,}",
             f"{total_recent / max(sum(data['totals_per_year']), 1):.0%} of all JMLR papers since {years[0]}")
        _kpi("hub", "Subtopics covered", f"{len(summaries)} / {len(subs)}",
             f"{with_recent} with {recent_a}-{recent_b % 100} papers · {rising} growing")
        _kpi("rocket_launch", "Fastest-rising subtopic", _short(fastest["name"], 30),
             f"momentum ×{fastest['momentum']:.1f} · {fastest['recent']} recent papers", color="emerald")
        if gen:
            _kpi("auto_awesome", "Summaries generated", gen.get("date", "?"),
                 f"{gen.get('model', '')} · {gen.get('calls', 0)} calls · ${gen.get('cost_usd', 0):.3f}",
                 color="amber")
        else:
            _kpi("auto_awesome", "Summaries generated", "not yet", "run: python advances.py", color="amber")

    state = {"area": None, "sort": "momentum", "q": "", "level": "areas"}

    # ---- charts
    with ui.row().classes("w-full gap-4 items-stretch no-wrap max-lg:flex-wrap"):
        with ui.card().props("flat bordered").classes("rounded-xl p-4 gap-2 bg-white basis-1/2 grow min-w-[380px]"):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.column().classes("gap-0"):
                    ui.label("Momentum").classes("text-base font-semibold")
                    ui.label(f"Share of papers {recent_a}-{recent_b % 100} vs 2010-22 (log scale)") \
                        .classes("text-xs text-gray-500")
                ui.toggle({"areas": "Areas", "subtopics": "Subtopics"}, value="areas",
                          on_change=lambda e: (state.update(level=e.value), momentum_chart.refresh())) \
                    .props("dense no-caps unelevated toggle-color=primary").classes("text-xs")

            @ui.refreshable
            def momentum_chart():
                if state["level"] == "areas":
                    items, h = list(areas.values()), 430
                    names = lambda s: _short(s["name"], 30)
                else:
                    ranked = sorted((s for s in subs.values() if s["recent"] + s["base"] >= 8),
                                    key=lambda s: -s["momentum"])
                    items, h = ranked[:10] + ranked[-8:], 430
                    names = lambda s: f"{s['id']} {_short(s['name'], 26)}"
                chart = ui.echart(_momentum_options(items, names)).classes("w-full").style(f"height:{h}px")

                def click(e):
                    tid = (e.data or {}).get("id") if isinstance(e.data, dict) else None
                    if not tid:
                        return
                    area = tid.split(".")[0]
                    area_select.value = area
                    if "." in tid:
                        search.value = subs[tid]["name"]
                    ui.run_javascript("document.getElementById('advances-cards')?.scrollIntoView({behavior:'smooth'})")
                chart.on_point_click(click)
            momentum_chart()
            ui.label("Click a bar to filter the cards below. Subtopics view: the 10 fastest-rising and 8 fading "
                     "subtopics with at least 8 papers.").classes("text-[11px] text-gray-400")

        with ui.card().props("flat bordered").classes("rounded-xl p-4 gap-2 bg-white basis-1/2 grow min-w-[380px]"):
            ui.label("Papers per year by area").classes("text-base font-semibold")
            ui.label("Papers per year, stacked: the six areas with most papers in 2023-25, the rest grouped as other") \
                .classes("text-xs text-gray-500")
            ui.echart(_per_year_options(data)).classes("w-full").style("height:448px")

    # ---- filter bar
    area_opts = {None: "All areas"} | {tid: f"{tid}. {a['name']}" for tid, a in
                                       sorted(areas.items(), key=lambda kv: int(kv[0]))}
    with ui.card().props("flat bordered").classes("rounded-xl px-4 py-3 w-full bg-white") \
            .props("id=advances-cards"):
        with ui.row().classes("w-full items-center gap-3 no-wrap max-md:flex-wrap"):
            ui.icon("filter_list").classes("text-gray-500 text-xl")
            area_select = ui.select(area_opts, value=None, label="Area",
                                    on_change=lambda e: (state.update(area=e.value), cards.refresh())) \
                .props("dense outlined options-dense").classes("w-80")
            ui.select({"momentum": "Momentum", "recent": "Recent papers", "name": "Name"}, value="momentum",
                      label="Sort by", on_change=lambda e: (state.update(sort=e.value), cards.refresh())) \
                .props("dense outlined options-dense").classes("w-44")
            search = ui.input(placeholder="Filter by name, author, claim...",
                              on_change=lambda e: (state.update(q=e.value or ""), cards.refresh())) \
                .props("dense outlined clearable debounce=250").classes("grow min-w-[220px]")
            with search.add_slot("prepend"):
                ui.icon("search").classes("text-gray-400")
            count_label = ui.label().classes("text-sm text-gray-500 whitespace-nowrap")

    def matches(s, q):
        if not q:
            return True
        summ = summaries.get(s["id"], {})
        hay = " ".join([s["id"], s["name"], s["description"], s["area_name"], summ.get("headline", ""),
                        " ".join(a["claim"] for a in summ.get("advances", [])),
                        " ".join(a["name"] for a in s["top_authors"])]).lower()
        return all(w in hay for w in q.lower().split())

    @ui.refreshable
    def cards():
        items = [s for s in subs.values()
                 if (state["area"] is None or s["area"] == state["area"]) and matches(s, state["q"])]
        key = {"momentum": lambda s: -s["momentum"], "recent": lambda s: (-s["recent"], -s["momentum"]),
               "name": lambda s: s["name"].lower()}[state["sort"]]
        items.sort(key=key)
        count_label.set_text(f"{len(items)} of {len(subs)} subtopics")
        if not items:
            with ui.column().classes("w-full items-center py-10 text-gray-500"):
                ui.icon("search_off").classes("text-4xl")
                ui.label("No subtopic matches these filters.")
            return
        with ui.element("div").classes("w-full grid gap-4 grid-cols-1 md:grid-cols-2 2xl:grid-cols-3"):
            for s in items:
                _card(s, summaries.get(s["id"]), meta, years, recent_a)

    cards()


def _card(s, summ, meta, years, recent_from):
    with ui.card().props("flat bordered").classes("rounded-xl p-4 gap-2 bg-white hover:shadow-md transition-shadow"):
        with ui.row().classes("w-full items-start justify-between no-wrap gap-2"):
            with ui.column().classes("gap-0 min-w-0"):
                ui.label(f"{s['area']}. {s['area_name']}").classes(
                    "text-[11px] uppercase tracking-wide text-gray-400 font-medium truncate max-w-full")
                ui.label(f"{s['id']}  {s['name']}").classes("text-[15px] font-semibold text-gray-900 leading-snug")
            _momentum_badge(s)
        with ui.row().classes("w-full items-end justify-between no-wrap gap-3"):
            with ui.column().classes("gap-0"):
                with ui.row().classes("items-baseline gap-1"):
                    ui.label(str(s["recent"])).classes("text-xl font-semibold text-indigo-700 leading-none")
                    ui.label(f"papers {recent_from}-25").classes("text-xs text-gray-500")
                ui.label(f"{s['size']} in total").classes("text-[11px] text-gray-400")
            ui.html(_sparkline(s["per_year"], years, recent_from), sanitize=False) \
                .tooltip(f"Papers per year {years[0]}-{years[-1]} ({recent_from}+ highlighted)")
        ui.separator().classes("my-1")
        if summ and summ.get("headline"):
            if summ.get("window") and summ["window"][0] != recent_from:
                ui.label(f"Few papers in {recent_from}-25: summary covers {summ['window'][0]}-"
                         f"{summ['window'][1] % 100}").classes(
                    "text-[11px] text-amber-700 bg-amber-50 rounded px-2 py-0.5 self-start")
            ui.label(summ["headline"]).classes("text-sm text-gray-800 font-medium leading-snug")
            with ui.column().classes("gap-1.5 w-full"):
                for adv in summ.get("advances", []):
                    with ui.row().classes("w-full no-wrap items-start gap-2"):
                        ui.element("div").classes("w-1.5 h-1.5 rounded-full bg-indigo-400 mt-[7px] shrink-0")
                        with ui.element("div").classes("text-[13px] text-gray-700 leading-snug"):
                            ui.label(adv["claim"]).classes("inline mr-1")
                            for k in adv["keys"]:
                                title, year = meta.get(k, (k, None))
                                ui.link(f"{year or k}", f"/paper/{k}").classes(
                                    "inline-block align-middle text-[11px] leading-4 px-1.5 mx-0.5 rounded-md "
                                    "no-underline bg-indigo-50 text-indigo-700 hover:bg-indigo-100 "
                                    "border border-indigo-100").tooltip(f"{title} [{k}]")
        else:
            with ui.row().classes("items-center gap-2 text-gray-500"):
                ui.icon("info_outline").classes("text-base")
                ui.label(f"No summary: only {s['window_count']} paper{'s' if s['window_count'] != 1 else ''} "
                         f"in {s['window'][0]}-{s['window'][1] % 100}." if s["window_count"] < 2
                         else "No summary yet - run python advances.py.").classes("text-sm")
            if s["central"]:
                for p in s["central"][:2]:
                    ui.link(f"{p['year']} · {_short(p['title'], 70)}", f"/paper/{p['key']}") \
                        .classes("text-xs text-indigo-700 no-underline hover:underline")
        if s["top_authors"]:
            with ui.row().classes("w-full items-start gap-1 no-wrap mt-auto pt-1"):
                ui.icon("group").classes("text-gray-400 text-sm mt-[1px]")
                ui.label(", ".join(f"{a['name']} ({a['papers']})" for a in s["top_authors"][:4])) \
                    .classes("text-[11px] text-gray-500 leading-snug")
