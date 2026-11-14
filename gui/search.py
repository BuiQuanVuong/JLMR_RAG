"""Search page: ranked papers for a plain-language query, or Google Scholar-style Boolean queries, with filters.

No Claude calls: plain queries use ix.rank_papers (dense + BM25 + passages, rewrite off); Scholar queries use boolq
matching + the same ranking (gui/search_data.search). Results show the best-matching passage with its page.
"""
from urllib.parse import quote

from nicegui import run, ui

from gui import search_data as sd
from gui import state
from gui.search_cards import add_css, result_card

EXAMPLES = [  # (query, what it does)
    ('"two-sample test" kernel', 'an exact phrase plus a word anywhere in the paper'),
    ("intitle:bandit regret author:Lattimore", "a word in the title, another anywhere, an author"),
    ("federated prototypes OR rules", "federated AND (prototypes OR rules)"),
    ("gaussian process -classification", "exclude papers that mention a word"),
    ('intitle:"variational inference" -"deep"', "a title phrase, excluding a phrase"),
]

K_OPTIONS = {10: "10 results", 20: "20 results", 30: "30 results", 50: "50 results"}


def _kbd(text):
    return f'<span class="sr-kbd">{text}</span>'


def page(q=""):
    add_css()
    st = {"q": (q or "").strip(), "k": 10, "years": {"min": sd.YEAR_MIN, "max": sd.YEAR_MAX}, "area": None,
          "token": 0}
    areas = {a: f"{a}. {name}" for a, name in sd.topics().areas()}

    # ------------------------------------------------------------ search box
    with ui.card().classes("w-full p-5 gap-3 rounded-2xl border border-gray-200 shadow-none"):
        with ui.row().classes("w-full items-center no-wrap gap-3"):
            box = ui.input(placeholder="Search 4,138 JMLR papers - a topic, a question, or a Scholar query",
                           value=st["q"]).props('outlined rounded clearable autofocus input-class="text-base"') \
                .classes("grow")
            with box.add_slot("prepend"):
                ui.icon("search", color="primary")
            go = ui.button("Search", icon="arrow_forward").props("unelevated rounded no-caps size=md").classes("px-5")
        ui.html("Scholar syntax: " + " ".join(_kbd(s) for s in ('"exact phrase"', "OR", "-exclude",
                                                                 "intitle:word", "author:Name"))
                + ' <span class="text-gray-400">- or just describe what you are looking for.</span>',
                sanitize=False).classes("text-sm text-gray-600 -mt-1")
        with ui.expansion("Query syntax and examples", icon="help_outline").classes(
                "w-full text-sm text-gray-700 bg-gray-50 rounded-lg").props("dense"):
            ui.markdown(
                "Plain text is ranked by meaning (title, abstract and full text). As soon as a query uses any "
                "Scholar syntax, papers must **match every term** (anywhere in the paper; bibliographies excluded), "
                "and the matches are ranked by relevance. Words are stemmed; hyphens count as spaces. If no paper "
                "matches every term, the closest matches are shown with what each is missing."
            ).classes("text-sm")
            with ui.column().classes("gap-1 w-full"):
                for ex, what in EXAMPLES:
                    with ui.row().classes("items-center gap-3 no-wrap"):
                        ui.button(ex, on_click=lambda e=ex: _search_for(e)) \
                            .props("flat dense no-caps color=primary").classes("font-mono text-xs")
                        ui.label(what).classes("text-xs text-gray-500")

        # -------------------------------------------------------- filters
        with ui.row().classes("w-full items-center gap-x-8 gap-y-2 pt-1"):
            with ui.row().classes("items-center gap-3 no-wrap min-w-[300px] grow-[2]"):
                ui.icon("date_range", color="grey-7")
                years_lbl = ui.label().classes("text-sm text-gray-700 w-[92px] shrink-0")
                rng = ui.range(min=sd.YEAR_MIN, max=sd.YEAR_MAX, step=1, value=dict(st["years"])) \
                    .props("label snap color=primary").classes("grow")
            area_sel = ui.select({None: "All research areas", **areas}, value=None, label="Research area") \
                .props("outlined dense options-dense").classes("min-w-[280px] grow")
            k_sel = ui.select(K_OPTIONS, value=10).props("outlined dense options-dense").classes("w-[130px]")

    def show_years():
        v = rng.value or st["years"]
        years_lbl.set_text(f"{int(v['min'])} - {int(v['max'])}")
    show_years()

    status = ui.row().classes("w-full items-center justify-between px-1 min-h-[28px]")
    results = ui.column().classes("w-full gap-3")

    # ------------------------------------------------------------ actions
    def set_area(area):
        area_sel.set_value(area)          # triggers a search via on_value_change

    def _signature():
        y = rng.value or st["years"]
        return (st["q"], k_sel.value, int(y["min"]), int(y["max"]), area_sel.value)

    async def _start():
        st["q"] = (box.value or "").strip()
        st["sig"] = _signature()
        ui.navigate.history.replace(f"/search?q={quote(st['q'])}" if st["q"] else "/search")
        await _run()

    async def _search_for(text):
        box.set_value(text)
        await _start()

    async def _filters_changed():
        show_years()
        if st["q"] and _signature() != st.get("sig"):     # skip echoes of programmatic changes
            await _start()

    async def _range_released(e):
        if isinstance(e.args, dict) and {"min", "max"} <= set(e.args):
            rng.set_value({"min": e.args["min"], "max": e.args["max"]})
        await _filters_changed()

    async def _run():
        st["token"] += 1
        token = st["token"]
        status.clear()
        results.clear()
        q = st["q"]
        if not q:
            _empty_intro()
            return
        if not state.ready():
            _waiting_for_index(token)
            return
        with status:
            with ui.row().classes("items-center gap-2 text-sm text-gray-600"):
                ui.spinner(size="18px")
                ui.label("Searching...")
        with results:
            for _ in range(3):
                with ui.card().classes("w-full p-4 gap-2 border border-gray-200 shadow-none rounded-xl"):
                    ui.skeleton("text").classes("w-2/3 text-lg")
                    ui.skeleton("text").classes("w-1/3")
                    ui.skeleton("rect", height="54px").classes("w-full")
        years = rng.value or st["years"]
        try:
            res = await run.io_bound(sd.search, state.index(), q, int(k_sel.value or 10),
                                     int(years["min"]), int(years["max"]), area_sel.value)
        except Exception as e:  # noqa: BLE001 - show any backend failure on the page
            if token != st["token"]:
                return
            status.clear()
            results.clear()
            with results:
                _message("error_outline", "The search failed", f"{type(e).__name__}: {e}", "red")
            return
        if token != st["token"]:      # a newer search started meanwhile
            return
        _render(res, q)

    def _render(res, q):
        status.clear()
        results.clear()
        if res.boolean:
            with results:
                _boolean_summary(res)
        n = len(res.rows)
        with status:
            filt = []
            y = rng.value or st["years"]
            if (y["min"], y["max"]) != (sd.YEAR_MIN, sd.YEAR_MAX):
                filt.append(f"{int(y['min'])}-{int(y['max'])}")
            if area_sel.value:
                filt.append(areas.get(area_sel.value, ""))
            ui.label(f"{n} paper{'s' if n != 1 else ''}" + (f"  ·  {' · '.join(filt)}" if filt else "")) \
                .classes("text-sm text-gray-600")
            if filt:
                ui.button("Clear filters", icon="filter_alt_off", on_click=_clear_filters) \
                    .props("flat dense no-caps size=sm color=grey-8")
        with results:
            if not res.rows:
                hint = res.note or ("Try fewer or broader terms, or widen the year range / research area."
                                    if filt else "Try fewer or broader terms.")
                _message("search_off", "No papers found", hint, "grey")
                return
            for i, (p, hit, missing) in enumerate(res.rows, 1):
                result_card(i, p, hit, missing, q, on_area=set_area)

    def _boolean_summary(res):
        with ui.card().classes("w-full px-4 py-3 gap-2 rounded-xl border border-indigo-100 bg-indigo-50/60 "
                               "shadow-none"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("rule", color="primary")
                ui.label("Scholar-style query").classes("text-sm font-semibold text-indigo-900")
                if res.n_strict:
                    ui.label(f"{res.n_strict} paper{'s' if res.n_strict != 1 else ''} match every term"
                             + (f"; showing the top {len(res.rows)}" if len(res.rows) < res.n_strict else "")) \
                        .classes("text-sm text-indigo-900")
                elif res.rows:
                    ui.label(f"No paper matches every term - closest matches, most of the {res.n_positive} "
                             f"terms first").classes("text-sm text-amber-800")
            with ui.row().classes("items-center gap-2"):
                for label, n in res.counts:
                    neg = label.startswith("-")
                    with ui.row().classes(
                            "items-center gap-2 rounded-lg px-2.5 py-1 text-xs border "
                            + ("bg-red-50 border-red-200 text-red-800" if neg else
                               "bg-white border-indigo-200 text-gray-800" if n else
                               "bg-amber-50 border-amber-300 text-amber-900")):
                        ui.label(label).classes("font-mono")
                        ui.label(f"{n:,} papers" + (" excluded" if neg else "")).classes("text-gray-500")

    async def _clear_filters():
        rng.set_value({"min": sd.YEAR_MIN, "max": sd.YEAR_MAX})
        area_sel.set_value(None)
        show_years()
        await _start()

    def _waiting_for_index(token):
        with results:
            card = _message("hourglass_top", "Loading the search index",
                            "It is loaded once when the app starts (about 25 s). Your search will run as soon as "
                            "it is ready.", "indigo", spinner=True)

        def poll():
            if token != st["token"]:
                timer.cancel()
            elif state.STATE["error"]:
                timer.cancel()
                results.clear()
                with results:
                    _message("error_outline", "The index failed to load", state.STATE["error"], "red")
            elif state.ready():
                timer.cancel()
                ui.timer(0.01, _run, once=True)
        timer = ui.timer(0.5, poll)
        return card

    def _empty_intro():
        with results:
            with ui.column().classes("w-full items-center py-10 gap-2 text-center"):
                ui.icon("travel_explore", color="indigo-3").classes("text-6xl")
                ui.label("Search the JMLR archive").classes("text-lg font-semibold text-gray-800")
                ui.label("Every result links to its paper page and opens the PDF at the best-matching page.") \
                    .classes("text-sm text-gray-500")
                with ui.row().classes("gap-2 mt-2 justify-center"):
                    for ex in ("kernel two-sample test", "Thompson sampling regret bounds",
                               'intitle:"two-sample" kernel author:Gretton', "causal discovery with latent variables"):
                        ui.chip(ex, icon="north_east", color="white", text_color="indigo-9",
                                on_click=lambda e=ex: _search_for(e)) \
                            .props("outline clickable").classes("text-sm")

    # ------------------------------------------------------------ wiring
    box.on("keydown.enter", _start)
    go.on_click(_start)
    rng.on("change", _range_released)              # on release, not while dragging
    rng.on_value_change(lambda e: show_years())
    area_sel.on_value_change(_filters_changed)
    k_sel.on_value_change(_filters_changed)
    st["sig"] = _signature()
    ui.timer(0.1, _run, once=True)


def _message(icon, title, text, color="grey", spinner=False):
    with ui.card().classes("w-full items-center text-center py-10 gap-2 rounded-xl border border-gray-200 "
                           "shadow-none") as card:
        if spinner:
            ui.spinner("dots", size="42px", color="primary")
        else:
            ui.icon(icon, color=f"{color}-5" if color != "indigo" else "primary").classes("text-5xl")
        ui.label(title).classes("text-base font-semibold text-gray-800")
        ui.label(text).classes("text-sm text-gray-500 max-w-[560px]")
    return card
