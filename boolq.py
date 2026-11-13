"""Google Scholar-style Boolean queries over whole papers.

Syntax (switched on automatically when a query uses any of it):
  "exact phrase"        the words in this order (hyphens count as spaces: "privacy-preserving")
  word                  every plain word is required (stemmed: "prototypes" also matches "prototype")
  a OR b                either term; OR binds the terms next to it ("federated prototypes OR rules"
                        = federated AND (prototypes OR rules))
  -word, -"phrase"      the paper must NOT contain it
  intitle:word          in the title (intitle:"two sample" for a phrase)
  author:Name           an author whose name contains it (author:"Gretton")

Matching is per paper, as in Scholar: the terms can appear anywhere in the paper's text (bibliography
passages are excluded, so a paper doesn't match just because it *cites* a title with those words).

When no paper satisfies every term, `match()` still returns the papers that satisfy the most terms, with
what each one is missing, so a too-specific query never dead-ends.
"""
import re
from dataclasses import dataclass, field

TOKEN = re.compile(r'(-?)(intitle:|author:)?("([^"]*)"|\S+)', re.I)


@dataclass
class Term:
    kind: str            # "word" | "phrase" | "title" | "author"
    value: str

    def label(self):
        v = f'"{self.value}"' if (self.kind == "phrase" or " " in self.value) else self.value
        return {"title": f"intitle:{v}", "author": f"author:{v}"}.get(self.kind, v)


@dataclass
class Clause:
    alternatives: list = field(default_factory=list)   # OR-ed Terms
    negated: bool = False

    def label(self):
        s = " OR ".join(t.label() for t in self.alternatives)
        return f"-{s}" if self.negated else s


def is_boolean(q):
    """True if the query uses Scholar syntax (so plain natural-language questions stay as they are)."""
    return bool(re.search(r'"|\bOR\b|(^|\s)-\S|\b(intitle|author):', q))


def _words(s):
    return re.findall(r"[A-Za-z0-9]+", s.lower())


def parse(q):
    clauses, pending_or = [], False
    for m in TOKEN.finditer(q):
        neg, prefix, raw, quoted = m.group(1), (m.group(2) or "").lower(), m.group(3), m.group(4)
        if raw == "OR" and not neg and not prefix:
            pending_or = bool(clauses)
            continue
        value = quoted if quoted is not None else raw
        words = _words(value)
        if not words:
            continue
        if prefix == "intitle:":
            term = Term("title", " ".join(words))
        elif prefix == "author:":
            term = Term("author", value.strip())
        elif quoted is not None or len(words) > 1:       # "a b" or privacy-preserving
            term = Term("phrase", " ".join(words))
        else:
            term = Term("word", words[0])
        if pending_or and not neg and not clauses[-1].negated:
            clauses[-1].alternatives.append(term)
        else:
            clauses.append(Clause([term], negated=bool(neg)))
        pending_or = False
    return clauses


def plain_text(clauses):
    """The positive words, for ranking the matching papers by meaning."""
    return " ".join(t.value for c in clauses if not c.negated for t in c.alternatives if t.kind != "author")


# ---------------------------------------------------------------- matching

def _papers_for(con, term, cache):
    key = (term.kind, term.value)
    if key in cache:
        return cache[key]
    if term.kind in ("word", "phrase"):
        fts = '"' + term.value + '"'
        rows = con.execute("""SELECT DISTINCT c.paper_key FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
                              WHERE chunks_fts MATCH ? AND c.is_ref = 0""", (fts,))
    elif term.kind == "title":
        rows = con.execute("""SELECT p.key FROM papers_fts JOIN papers p ON p.rowid = papers_fts.rowid
                              WHERE papers_fts MATCH ?""", (f'title : "{term.value}"',))
    else:
        rows = con.execute("SELECT key FROM papers WHERE authors LIKE ?", (f"%{term.value}%",))
    cache[key] = {r[0] for r in rows}
    return cache[key]


@dataclass
class Result:
    clauses: list
    counts: list            # (clause label, number of papers) per clause
    strict: set             # papers satisfying every clause
    partial: list           # (paper key, n satisfied, [missing clause labels]) when strict is empty


def match(con, q, partial_limit=200):
    clauses = parse(q) if isinstance(q, str) else q
    cache, sets = {}, []
    for c in clauses:
        s = set()
        for t in c.alternatives:
            s |= _papers_for(con, t, cache)
        sets.append(s)
    counts = [(c.label(), len(s)) for c, s in zip(clauses, sets)]
    pos = [(c, s) for c, s in zip(clauses, sets) if not c.negated]
    excluded = set().union(*(s for c, s in zip(clauses, sets) if c.negated))
    if not pos:
        return Result(clauses, counts, set(), [])
    strict = set.intersection(*(s for _, s in pos)) - excluded
    partial = []
    if not strict:
        for key in set().union(*(s for _, s in pos)) - excluded:
            missing = [c.label() for c, s in pos if key not in s]
            partial.append((key, len(pos) - len(missing), missing))
        partial.sort(key=lambda x: -x[1])
        partial = partial[:partial_limit]
    return Result(clauses, counts, strict, partial)


def describe(result):
    """One line per clause: how many papers contain it (exclusions say how many they remove)."""
    return [f"{n:>6} papers  {label}" if not label.startswith("-") else
            f"{n:>6} papers  contain {label[1:]} and are excluded" for label, n in result.counts]
