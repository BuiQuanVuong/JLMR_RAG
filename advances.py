"""Recent advances per research area and subtopic: local statistics plus a short Claude-written summary.

For each of the topic map's subtopics (see topics.py) this computes, locally:
  - papers per year 2000-2025, the recent count (2023-2025) and momentum = share of all JMLR papers in
    2023-25 that belong to the subtopic / its share in 2010-22 (>1 = growing),
  - the top recent authors and the most central recent papers (closest to the subtopic centre),
and the same statistics per research area. Then one Claude Haiku call per subtopic reads its recent papers
(title, year, key, trimmed abstract) and writes a one-sentence headline plus 3-5 "advances" bullets, each
citing the paper keys it draws on. Cited keys are checked against the subtopic's recent papers.

A subtopic with fewer than 3 papers in 2023-25 is widened to 2021-25 (the window used is recorded); with
fewer than 2 papers even then it gets no summary.

Output: data/advances.json (read by the desktop app's /advances page, gui/advances.py).

Usage:  python advances.py             recompute statistics and regenerate all summaries (~$0.05-0.10)
        python advances.py --dry-run   statistics only, no API calls (keeps existing summaries if any)
"""
import argparse
import datetime as dt
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import DATA_DIR, connect, utf8_console
import topics

OUT = DATA_DIR / "advances.json"
MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6       # USD per token
BUDGET = 0.30                                       # hard stop, USD
WORKERS = 6

YEARS = list(range(2000, 2026))
RECENT = (2023, 2025)
WIDE = (2021, 2025)
BASE = (2010, 2022)
MIN_RECENT = 3              # below this, widen the window to WIDE
MIN_SUMMARY = 2             # below this (even widened), no summary
MAX_PAPERS = 22             # papers sent per call (most central first)
ABSTRACT_CHARS = 1200
MAX_INPUT_CHARS = 32000     # cap on the paper list in one prompt
MIN_BASE_PAPERS = 3         # momentum is flagged as low-confidence below this many base-period papers
SMOOTH = 0.002              # added to both shares (as in topics.show_rising)

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "advances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"claim": {"type": "string"},
                               "keys": {"type": "array", "items": {"type": "string"}}},
                "required": ["claim", "keys"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["headline", "advances"],
    "additionalProperties": False,
}

PROMPT = """You are summarising recent research in one subtopic of the Journal of Machine Learning Research (JMLR).

Subtopic: {name}
Description: {description}
Research area: {area}

Below are the subtopic's JMLR papers from {a}-{b} ({n} shown, most central first), each with its key, year, \
title and (trimmed) abstract.

Write:
- "headline": ONE sentence (at most 30 words) on where this subtopic is heading, based only on these papers.
- "advances": 3 to 5 bullets (fewer if the papers do not support more). Each "claim" is a short, concrete \
statement of a recent advance or direction (at most 30 words), and "keys" lists the keys (exactly as given, \
e.g. "v25/23-0123") of the 1-3 papers below that it draws on. Every claim must be supported by the cited \
abstracts; do not invent results, numbers or papers. Prefer claims that synthesise a theme across papers.

Plain text only, no markdown. Do not mention "JMLR" or "these papers" in the claims.

Papers:
{papers}"""


def _share_ratio(n_recent, tot_recent, n_base, tot_base):
    r, b = n_recent / max(tot_recent, 1), n_base / max(tot_base, 1)
    return round((r + SMOOTH) / (b + SMOOTH), 3), r, b


def _in(y, window):
    return window[0] <= (y or 0) <= window[1]


def _top_authors(members, n=6):
    c = Counter(a.strip() for m in members for a in (m["authors"] or "").split(",") if a.strip())
    return [{"name": a, "papers": k} for a, k in c.most_common(n)]


def topic_stats(members, totals_by_year, n_central=8):
    """Statistics of one topic from its (primary) members."""
    per_year = Counter(m["year"] for m in members)
    tot = lambda w: sum(totals_by_year[y] for y in range(w[0], w[1] + 1))
    cnt = lambda w: sum(per_year[y] for y in range(w[0], w[1] + 1))
    momentum, share_recent, share_base = _share_ratio(cnt(RECENT), tot(RECENT), cnt(BASE), tot(BASE))
    recent = [m for m in members if _in(m["year"], RECENT)]
    window = RECENT if len(recent) >= MIN_RECENT else WIDE
    in_window = sorted((m for m in members if _in(m["year"], window)), key=lambda m: -m["score"])
    return {
        "size": len(members),
        "per_year": [per_year[y] for y in YEARS],
        "recent": cnt(RECENT),
        "base": cnt(BASE),
        "share_recent": round(share_recent, 5),
        "share_base": round(share_base, 5),
        "momentum": momentum,
        "momentum_low_base": cnt(BASE) < MIN_BASE_PAPERS,
        "window": list(window),
        "window_count": len(in_window),
        "top_authors": _top_authors(in_window),
        "central": [{"key": m["paper_key"], "title": m["title"], "year": m["year"]} for m in in_window[:n_central]],
    }


def compute_stats(con):
    rows = topics._topic_rows(con)
    totals_by_year = Counter(y for (y,) in con.execute("SELECT year FROM papers WHERE status='done'"))
    order = lambda t: [int(p) for p in t.split(".")]
    areas, subs = {}, {}
    for tid in sorted(rows, key=order):
        r = rows[tid]
        members = topics._members(con, tid)
        s = topic_stats(members, totals_by_year, n_central=8 if r["level"] == 2 else 5)
        s.update(id=tid, name=r["name"], description=r["description"] or "")
        if r["level"] == 1:
            areas[tid] = s
        else:
            s["area"] = r["parent"]
            s["area_name"] = rows[r["parent"]]["name"]
            subs[tid] = s
            s["_members"] = members
    return areas, subs, totals_by_year


def paper_list(con, sub):
    """The papers (most central first) of a subtopic's window, as prompt text, and their keys."""
    w = sub["window"]
    members = sorted((m for m in sub["_members"] if _in(m["year"], w)), key=lambda m: -m["score"])[:MAX_PAPERS]
    parts, keys, used = [], [], 0
    for m in members:
        row = con.execute("SELECT abstract FROM papers WHERE key=?", (m["paper_key"],)).fetchone()
        abstract = " ".join((row["abstract"] or "").split()) if row else ""
        if len(abstract) > ABSTRACT_CHARS:
            abstract = abstract[:ABSTRACT_CHARS].rsplit(" ", 1)[0] + " ..."
        part = f"[{m['paper_key']}] ({m['year']}) {m['title']}\n{abstract}\n"
        if used + len(part) > MAX_INPUT_CHARS and parts:
            break
        parts.append(part)
        keys.append(m["paper_key"])
        used += len(part)
    return "\n".join(parts), keys


def summarise(client, sub, papers_text, keys):
    prompt = PROMPT.format(name=sub["name"], description=sub["description"], area=sub["area_name"],
                           a=sub["window"][0], b=sub["window"][1], n=len(keys), papers=papers_text)
    msg = client.messages.create(
        model=MODEL, max_tokens=4000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}])
    cost = msg.usage.input_tokens * PRICE_IN + msg.usage.output_tokens * PRICE_OUT
    if msg.stop_reason != "end_turn":
        return None, cost, f"stop_reason={msg.stop_reason}"
    data = json.loads(next(b.text for b in msg.content if b.type == "text"))
    valid = set(keys)
    bullets, dropped = [], 0
    for adv in data.get("advances", [])[:5]:
        cited = []
        for k in adv.get("keys", []):
            k = k.strip().strip("[]")
            if k in valid and k not in cited:
                cited.append(k)
            else:
                dropped += 1
        claim = " ".join(adv.get("claim", "").split())
        if cited and claim:
            bullets.append({"claim": claim, "keys": cited})
    headline = " ".join(data.get("headline", "").split())
    return {"headline": headline, "advances": bullets, "dropped_keys": dropped,
            "input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}, cost, None


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="statistics only, no API calls")
    ap.add_argument("--only", nargs="*", help="regenerate only these subtopic ids (others kept from the old file)")
    a = ap.parse_args()

    con = connect()
    if not topics.has_topics(con):
        sys.exit("No topic map yet - run: python topics.py build")
    areas, subs, totals = compute_stats(con)
    old = {}
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    old_summaries = old.get("summaries", {})

    widened = [t for t, s in subs.items() if s["window"] != list(RECENT)]
    skipped = [t for t, s in subs.items() if s["window_count"] < MIN_SUMMARY]
    todo = [t for t, s in subs.items() if s["window_count"] >= MIN_SUMMARY and (not a.only or t in a.only)]
    print(f"{len(areas)} areas, {len(subs)} subtopics; {sum(s['recent'] for s in subs.values())} recent "
          f"({RECENT[0]}-{RECENT[1]}) papers. Widened to {WIDE[0]}-{WIDE[1]}: {', '.join(widened) or 'none'}. "
          f"Too few papers for a summary: {', '.join(skipped) or 'none'}.")
    rising = sorted(subs.values(), key=lambda s: -s["momentum"])
    print("Top momentum: " + ", ".join(f"{s['id']} {s['name'][:30]} x{s['momentum']:.2f}" for s in rising[:5]))

    summaries = {t: v for t, v in old_summaries.items() if t in subs}
    generation = old.get("generation", {})
    if not a.dry_run:
        import anthropic
        client = anthropic.Anthropic()
        jobs = {t: paper_list(con, subs[t]) for t in todo}
        est_in = sum(len(p) for p, _ in jobs.values()) / 3.5 + 400 * len(jobs)
        print(f"{len(jobs)} calls, ~{est_in / 1e3:.0f}k input tokens (est. ${est_in * PRICE_IN + len(jobs) * 1500 * PRICE_OUT:.3f})")
        spent, lock, stop, failures = 0.0, threading.Lock(), threading.Event(), []

        def run(t):
            if stop.is_set():
                return t, None, 0.0, "skipped (budget)"
            try:
                return (t,) + summarise(client, subs[t], *jobs[t])
            except Exception as e:
                return t, None, 0.0, f"{type(e).__name__}: {e}"

        with ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(run, t) for t in todo]
            for i, f in enumerate(as_completed(futures), 1):
                t, res, cost, err = f.result()
                with lock:
                    spent += cost
                    # each further call costs at most ~$0.005: stop before the budget could be exceeded
                    if spent + WORKERS * 0.006 > BUDGET:
                        stop.set()
                if res:
                    res["window"] = subs[t]["window"]
                    res["n_papers"] = len(jobs[t][1])
                    summaries[t] = res
                else:
                    failures.append((t, err))
                print(f"  [{i}/{len(todo)}] {t:>5} {'ok ' if res else 'ERR'} ${cost:.4f}  total ${spent:.4f}"
                      + (f"  {err}" if err else ""), flush=True)
        print(f"Done: {len(todo) - len(failures)} summaries, {len(failures)} failures, cost ${spent:.4f}")
        generation = {"model": MODEL, "date": dt.date.today().isoformat(), "calls": len(todo),
                      "cost_usd": round(spent, 4), "failures": [t for t, _ in failures]}
        if a.only and old.get("generation"):
            generation["cost_usd"] = round(spent + old["generation"].get("cost_usd", 0), 4)

    for s in subs.values():
        s.pop("_members", None)
    out = {
        "version": 1,
        "computed": dt.datetime.now().isoformat(timespec="seconds"),
        "years": YEARS,
        "totals_per_year": [totals[y] for y in YEARS],
        "windows": {"recent": list(RECENT), "wide": list(WIDE), "base": list(BASE),
                    "min_recent": MIN_RECENT, "min_summary": MIN_SUMMARY},
        "momentum_definition": f"share of all papers in {RECENT[0]}-{RECENT[1]} / share in {BASE[0]}-{BASE[1]}"
                               f" (each share + {SMOOTH})",
        "generation": generation,
        "areas": areas,
        "subtopics": subs,
        "summaries": summaries,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
