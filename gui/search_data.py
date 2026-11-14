"""Data helpers for the Search and Paper pages (no UI code; everything here is safe to call off the UI loop).

Paper-level data (topics, outline, BibTeX) needs only the SQLite DB; ranking needs the loaded ask.Index.
"""
import html
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache

from common import connect

YEAR_MIN, YEAR_MAX = 2000, 2025

_con = None
_con_lock = threading.Lock()


def db():
    """One read connection for these pages (serialized SQLite build, check_same_thread=False)."""
    global _con
    with _con_lock:
        if _con is None:
            _con = connect()
        return _con


# ---------------------------------------------------------------- topics

@dataclass
class Topics:
    names: dict = field(default_factory=dict)      # topic id -> name ("2" -> area name, "2.6" -> subtopic)
    parent: dict = field(default_factory=dict)     # subtopic id -> area id
    primary: dict = field(default_factory=dict)    # paper key -> subtopic id (rank 1)
    secondary: dict = field(default_factory=dict)  # paper key -> subtopic id (rank 2)

    def areas(self):
        """[(area id, name)] in id order."""
        ids = [t for t in self.names if "." not in t]
        return [(t, self.names[t]) for t in sorted(ids, key=lambda s: int(s) if s.isdigit() else 999)]

    def label(self, tid):
        return f"{tid} {self.names[tid]}" if tid in self.names else ""

    def area_of(self, key):
        tid = self.primary.get(key)
        return self.parent.get(tid) if tid else None

    def papers_in_area(self, area):
        return {k for k, t in self.primary.items() if self.parent.get(t) == area}


@lru_cache(maxsize=1)
def topics():
    con = db()
    t = Topics()
    if con.execute("SELECT 1 FROM sqlite_master WHERE name='paper_topics'").fetchone() is None:
        return t
    for r in con.execute("SELECT id, parent, name FROM topics"):
        t.names[r["id"]] = r["name"] or r["id"]
        if r["parent"]:
            t.parent[r["id"]] = r["parent"]
    for key, tid, rank in con.execute("SELECT paper_key, topic_id, rank FROM paper_topics WHERE rank IN (1, 2)"):
        (t.primary if rank == 1 else t.secondary)[key] = tid
    return t


# ---------------------------------------------------------------- one paper

def paper_row(key):
    r = db().execute("SELECT * FROM papers WHERE key=?", (key,)).fetchone()
    return dict(r) if r else None


def outline(key):
    """The paper's sections in reading order: [(depth, label, first page, is_ref)].

    Chunk sections look like '2 The MMD / 2.1 Definition'; every prefix becomes an outline entry at its first page."""
    seen, out = set(), []
    rows = db().execute("SELECT page_start, section, is_ref FROM chunks WHERE paper_key=? ORDER BY seq", (key,))
    for page, section, is_ref in rows:
        if not section:
            continue
        parts = [p.strip() for p in section.split(" / ") if p.strip()]
        for depth in range(len(parts)):
            path = tuple(parts[:depth + 1])
            if path not in seen:
                seen.add(path)
                out.append((depth, nice_section(parts[depth]), page or 1, bool(is_ref)))
    return out


def bibtex(key_or_row):
    import cite
    p = paper_row(key_or_row) if isinstance(key_or_row, str) else key_or_row
    return cite.entry_for(p) if p else ""


# ---------------------------------------------------------------- formatting

def short_authors(authors, n=4):
    names = [a.strip() for a in (authors or "").split(",") if a.strip()]
    return ", ".join(names) if len(names) <= n else ", ".join(names[:n - 1]) + f", ... {names[-1]}"


def snippet_html(text, query, n=340):
    """Escaped snippet (around the first query-word match) with query words in <mark>."""
    text = re.sub(r"\s+", " ", text or "").strip()
    words = sorted({w for w in re.findall(r"[A-Za-z0-9]{3,}", query.lower())
                    if w not in ("and", "the", "for", "with", "intitle", "author")}, key=len, reverse=True)
    if len(text) > n:
        start = 0
        if words:
            m = re.search("|".join(map(re.escape, words)), text, re.I)
            if m and m.start() > n // 3:
                start = text.rfind(" ", 0, m.start() - n // 4) + 1
        piece = text[start:start + n]
        piece = piece.rsplit(" ", 1)[0] if start + n < len(text) else piece
        text = ("... " if start else "") + piece + (" ..." if start + n < len(text) else "")
    out = html.escape(text)
    if words:
        out = re.sub(r"\b(" + "|".join(map(re.escape, words)) + r")\w*", r"<mark>\g<0></mark>", out, flags=re.I)
    return out


def nice_section(label):
    """'3.3.1 L2 DISTANCE BETWEEN PARZEN WINDOW ESTIMATES' -> '3.3.1 L2 distance between parzen window estimates'."""
    if not (label.isupper() and len(label) > 6):
        return label
    m = re.match(r"([\d.]+\s+|[A-Z]\.\s+|Appendix\s+\S+\s+)?(.*)", label, re.I)
    head, words = m.group(1) or "", m.group(2).split()

    def fix(i, w):
        if any(c.isdigit() for c in w):
            return w
        return w[:1] + w[1:].lower() if i == 0 else w.lower()
    return head + " ".join(fix(i, w) for i, w in enumerate(words))


def where(hit):
    s = f"p. {hit['page_start']}"
    sec = nice_section((hit.get("section") or "").split(" / ")[-1].strip())
    return s + (f" · {sec}" if sec else "")


# ---------------------------------------------------------------- searching

def _year_ok(y, since, until):
    return (since is None or y >= since) and (until is None or y <= until)


@dataclass
class SearchResult:
    rows: list                      # [(paper, hit or None, [missing terms])]
    boolean: bool = False
    counts: list = field(default_factory=list)   # [(clause label, n papers)]
    n_strict: int = 0               # papers matching every term (within the filters' area)
    n_positive: int = 0             # number of positive clauses
    note: str = ""


def search(ix, q, k=10, since=None, until=None, area=None):
    """Plain query -> ix.rank_papers; Scholar-style query -> Boolean match + ranking (like ask.boolean_search,
    plus the research-area filter). No Claude calls."""
    import boolq
    allowed = topics().papers_in_area(area) if area else None
    if since == YEAR_MIN:
        since = None
    if until == YEAR_MAX:
        until = None
    if not boolq.is_boolean(q):
        rows = ix.rank_papers(q, k=k, since=since, until=until, allowed=allowed)
        return SearchResult([(p, hit, []) for p, hit in rows])
    r = boolq.match(ix.con, q)
    text = boolq.plain_text(r.clauses) or q
    res = SearchResult([], boolean=True, counts=r.counts, n_positive=sum(not c.negated for c in r.clauses))
    strict = r.strict & allowed if allowed is not None else r.strict
    if strict:
        strict = {key for key in strict if key in ix.papers
                  and _year_ok(ix.papers[key]["year"] or 0, since, until)}
    res.n_strict = len(strict)
    if strict:
        res.rows = [(p, hit, []) for p, hit in ix.rank_papers(text, k=k, since=since, until=until, allowed=strict)]
        return res
    if r.strict:
        res.note = f"{len(r.strict)} papers match every term, but none within the selected filters."
        return res
    partial = [(key, n, miss) for key, n, miss in r.partial if allowed is None or key in allowed]
    missing = {key: miss for key, _, miss in partial}
    for n in sorted({n for _, n, _ in partial}, reverse=True):
        tier = {key for key, m, _ in partial if m == n}
        ranked = ix.rank_papers(text, k=k - len(res.rows), since=since, until=until, allowed=tier)
        res.rows += [(p, hit, missing[p["key"]]) for p, hit in ranked]
        if len(res.rows) >= k:
            break
    return res
