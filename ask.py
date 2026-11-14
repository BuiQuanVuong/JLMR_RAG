"""Step 3: search and ask questions over the JMLR collection.

  python ask.py search "kernel two-sample tests"          rank papers (local, free)
  python ask.py ask "How do GP bandits handle ...?"         answer with citations (Claude Haiku 5.5)
  python ask.py ask "..." --opus                            same, with Claude Opus 5.5 for hard questions
  python ask.py reading "causal discovery"                  ordered reading list (Claude)
  python ask.py compare "How do they choose the bandwidth?" compare papers side by side (Claude)
  python ask.py compare "..." --papers v13/gretton12a v24/21-1289    compare specific papers
  python ask.py similar v20/13-124                          papers similar to a given paper (local)
  python ask.py open v20/13-124                             open a paper's PDF
  python ask.py cite v20/13-124 v13/gretton12a --out refs.bib   BibTeX for papers
  python ask.py topics [3 | 3.2] [--rising] [--for "question"]   browse the topic map (local)
  python ask.py update [--check]                            fetch and index papers JMLR published since
  python ask.py chat                                        interactive mode with conversation memory

Filters for search/ask/reading: --since YEAR, --until YEAR, -k N (number of passages/papers).
--no-rewrite skips the Claude Haiku query rewrite (rewrite.py) and searches with your words only.
"""
import argparse
import os
import re
import sys
import textwrap

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the embedding model is cached by embed.py

import numpy as np

from common import (CHUNK_EMB, CHUNK_IDS, ENV_FILE, JMLR_DIR, PAPER_EMB, PAPER_KEYS, QUERY_PREFIX,
                    RERANK_MODEL, RERANK_POOL, REWRITE, connect, load_encoder, load_reranker, utf8_console)
from rewrite import bm25_text, rewrite as rewrite_query
import ui

MODELS = {"haiku": "claude-haiku-5-5", "opus": "claude-opus-5-5"}
STOP = set("""a an the of and or in on for to with by from is are was were be been being as at that this
these those it its how what which who whom why when where do does did can could should would will
about into than then there their them they we our you your i me my not no yes vs via using use used
between over under more most less such any all each other some""".split())

SYSTEM = """You are a research assistant helping a reader navigate papers from the Journal of Machine \
Learning Research (JMLR). You are given numbered source passages retrieved from JMLR papers.

- Answer from the sources. Cite every claim drawn from them with bracketed source numbers like [2] or [1][4].
- Name the papers you rely on (short title or first author + year) so the reader knows what to open.
- If the sources only partly cover the question, say what is missing. You may add well-established \
background knowledge, but mark it clearly as not coming from the sources and do not cite it.
- Be precise with mathematical statements; use plain-text or LaTeX-style notation.
- End with a short "Where to read next" line pointing to the 1-3 most relevant papers."""

READING_SYSTEM = """You are helping a reader build a reading path through Journal of Machine Learning \
Research (JMLR) papers. You are given a numbered list of candidate papers (title, year, authors, abstract).

Pick the papers that genuinely fit the topic (usually 6-12) and order them as a learning path: \
foundations/surveys first, then core methods, then recent advances or specialised directions. Group them \
under short headings. For each paper give its number in brackets, title and year, and one or two sentences on \
what the reader gets from it and why it sits at that point in the path. Skip candidates that are off-topic; \
if important sub-areas are missing from the candidates, say so briefly."""


# ---------------------------------------------------------------- retrieval

REWRITE_MODES = ("off", "bm25", "both", "fuse", "hyde", "multi")


class Index:
    def __init__(self, rerank_model=RERANK_MODEL, rerank_pool=RERANK_POOL, section_in_rerank=False,
                 rewrite=REWRITE, rewrite_search=False):
        # section_in_rerank=True measured slightly worse (hard passage@12 60% vs 62%), so it is off
        # rewrite: how the Claude Haiku query rewrite (rewrite.py) is used by passages(), see REWRITE_MODES;
        # rewrite_search: also use it in rank_papers() (search / reading)
        if rewrite not in REWRITE_MODES:
            raise ValueError(f"rewrite must be one of {REWRITE_MODES}")
        self.rewrite = rewrite
        self.rewrite_search = rewrite_search
        self.con = connect()
        self.rerank_pool = rerank_pool
        self.section_in_rerank = section_in_rerank
        self.reranker = load_reranker(rerank_model) if rerank_model else None
        if not CHUNK_IDS.exists():
            sys.exit("No embeddings yet - run: python embed.py")
        if not ui.fancy():                      # in a terminal, main() shows a spinner instead
            print("Loading index...", end="\r", file=sys.stderr)
        self.chunk_ids = np.load(CHUNK_IDS)
        self.chunk_emb = np.load(CHUNK_EMB).astype(np.float32)
        self.paper_keys = np.load(PAPER_KEYS)
        self.paper_emb = np.load(PAPER_EMB).astype(np.float32)
        self.papers = {r["key"]: dict(r) for r in self.con.execute("SELECT * FROM papers")}
        key_of = dict(self.con.execute("SELECT id, paper_key FROM chunks WHERE is_ref=0").fetchall())
        keep = np.array([int(i) in key_of for i in self.chunk_ids], dtype=bool)  # drop re-flagged refs
        self.chunk_ids, self.chunk_emb = self.chunk_ids[keep], self.chunk_emb[keep]
        self.chunk_paper = np.array([key_of[int(i)] for i in self.chunk_ids])
        year = {k: (p["year"] or 0) for k, p in self.papers.items()}
        self.chunk_year = np.array([year.get(k, 0) for k in self.chunk_paper])
        self.paper_year = np.array([year.get(k, 0) for k in self.paper_keys])
        self.model, _ = load_encoder()
        if not ui.fancy():
            print(" " * 30, end="\r", file=sys.stderr)

    def embed_query(self, q):
        return self.model.encode([QUERY_PREFIX + q], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)

    def query_variants(self, q, mode=None):
        """(dense, sparse) query lists for retrieval. dense items are (text, is_query): a hypothetical
        passage (HyDE) is embedded like a document, without the query prefix. Each item yields one ranked
        list; all lists are fused with RRF. Falls back to [q] / [q] when no rewrite is available."""
        mode = self.rewrite if mode is None else mode
        dense, sparse = [(q, True)], [q]
        rw = None if mode == "off" else rewrite_query(q)
        if not rw:
            return dense, sparse
        kw, rq = bm25_text(rw), rw["rewrite"]
        if mode == "bm25":
            sparse = [kw]
        elif mode == "both":
            dense, sparse = [(rq, True)], [kw]
        elif mode == "fuse":
            dense, sparse = [(q, True), (rq, True)], [q, kw]
        elif mode == "hyde":
            dense, sparse = [(q, True), (rw["passage"], False)], [q, kw]
        elif mode == "multi":
            dense = [(q, True), (rq, True), *((a, True) for a in rw["alternatives"]), (rw["passage"], False)]
            sparse = [q, kw, *rw["alternatives"]]
        return dense, sparse

    @staticmethod
    def fts_query(q):
        terms = [t for t in re.findall(r"[A-Za-z0-9]+", q.lower()) if t not in STOP and len(t) > 1]
        return " OR ".join(f'"{t}"' for t in terms)

    @staticmethod
    def year_ok(y, since, until):
        return (since is None or y >= since) and (until is None or y <= until)

    def _allow(self, allowed):
        """Load a paper filter into a temp table so FTS queries can join on it."""
        self.con.execute("CREATE TEMP TABLE IF NOT EXISTS allowed_keys(k TEXT PRIMARY KEY)")
        self.con.execute("DELETE FROM allowed_keys")
        self.con.executemany("INSERT INTO allowed_keys VALUES(?)", [(k,) for k in allowed])

    def passages(self, q, k=12, per_paper=3, since=None, until=None, pool=200, mode="hybrid", rerank=True,
                 rewrite=None, allowed=None):
        """Hybrid retrieval: dense (bge) + BM25 (SQLite FTS5), fused with reciprocal rank fusion, then
        (optionally) the top RERANK_POOL candidates re-scored by a cross-encoder against the ORIGINAL q.
        rewrite = None (use the Index setting) or one of REWRITE_MODES: which retrieval lists use the
        Claude rewrite of q. mode = 'hybrid' | 'dense' | 'sparse' (single-signal modes, for evaluation).
        allowed = optional set of paper keys to search within (e.g. the matches of a Boolean query)."""
        dense_q, sparse_q = self.query_variants(q, rewrite)
        lists = []
        if allowed is not None:
            self._allow(allowed)
        if mode in ("hybrid", "dense"):
            vecs = self.model.encode([(QUERY_PREFIX + t) if is_q else t for t, is_q in dense_q],
                                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            mask = np.ones(len(self.chunk_ids), bool)
            if since:
                mask &= self.chunk_year >= since
            if until:
                mask &= self.chunk_year <= until
            if allowed is not None:
                mask &= np.isin(self.chunk_paper, list(allowed))
            for qv in vecs:
                scores = self.chunk_emb @ qv
                scores[~mask] = -1
                top = np.argsort(-scores)[:pool]
                lists.append(self.chunk_ids[top[mask[top]]].tolist())

        for text in (sparse_q if mode in ("hybrid", "sparse") else []):
            fq = self.fts_query(text)
            if not fq:
                continue
            if allowed is None:
                # rank inside the FTS index first, then join only those rows: an expanded rewrite query can
                # match nearly every passage, and joining all of them before sorting took ~1 s per list
                rows = self.con.execute("""SELECT c.id, p.year FROM
                                             (SELECT rowid, bm25(chunks_fts) AS s FROM chunks_fts
                                              WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?) f
                                           JOIN chunks c ON c.id = f.rowid JOIN papers p ON p.key = c.paper_key
                                           WHERE c.is_ref = 0 ORDER BY f.s""", (fq, pool * 3)).fetchall()
            else:   # filtered: join first so the top results all come from allowed papers
                rows = self.con.execute("""SELECT c.id, p.year FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
                                           JOIN papers p ON p.key = c.paper_key
                                           WHERE chunks_fts MATCH ? AND c.is_ref = 0
                                             AND c.paper_key IN (SELECT k FROM allowed_keys)
                                           ORDER BY bm25(chunks_fts) LIMIT ?""", (fq, pool * 3)).fetchall()
            lists.append([r[0] for r in rows if self.year_ok(r[1] or 0, since, until)][:pool])

        fused = {}
        for ranked in lists:
            for rank, cid in enumerate(ranked):
                fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
        ranked = sorted(fused, key=fused.get, reverse=True)
        if rerank and self.reranker is not None and ranked:
            cands = [dict(self.con.execute("SELECT * FROM chunks WHERE id=?", (cid,)).fetchone(), score=fused[cid])
                     for cid in ranked[: self.rerank_pool]]
            def doc(c):
                head = self.papers[c["paper_key"]]["title"]
                if self.section_in_rerank and c.get("section"):
                    head += f" - {c['section']}"
                return f"{head}. {c['text']}"
            pairs = [(q, doc(c)) for c in cands]
            scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)
            for c, s in zip(cands, scores):
                c["score"] = float(s)
            candidates = sorted(cands, key=lambda c: c["score"], reverse=True)
        else:
            candidates = (dict(self.con.execute("SELECT * FROM chunks WHERE id=?", (cid,)).fetchone(),
                               score=fused[cid]) for cid in ranked)
        out, per = [], {}
        for c in candidates:
            if per.get(c["paper_key"], 0) >= per_paper:
                continue
            per[c["paper_key"]] = per.get(c["paper_key"], 0) + 1
            out.append(c)
            if len(out) >= k:
                break
        return out

    def rank_papers(self, q, k=10, since=None, until=None, allowed=None):
        """Paper ranking: fuse title/abstract similarity, abstract keyword match and best passage hits.
        With rewrite_search, the Claude rewrite adds its own title/abstract lists and passage lists.
        allowed = optional set of paper keys to rank (e.g. Boolean-query matches); every allowed paper is
        returned (up to k), the ones no ranking list reached last, newest first."""
        mode = self.rewrite if self.rewrite_search else "off"
        dense_q, sparse_q = self.query_variants(q, mode)
        ok = (lambda key: key in allowed) if allowed is not None else (lambda key: True)
        lists = []
        vecs = self.model.encode([(QUERY_PREFIX + t) if is_q else t for t, is_q in dense_q],
                                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        for qv in vecs:
            ps = self.paper_emb @ qv
            lists.append([self.paper_keys[i] for i in np.argsort(-ps)
                          if self.year_ok(self.paper_year[i], since, until) and ok(self.paper_keys[i])][:200])
        for text in sparse_q:
            fq = self.fts_query(text)
            if not fq:
                continue
            rows = self.con.execute("""SELECT p.key, p.year FROM papers_fts JOIN papers p ON p.rowid = papers_fts.rowid
                                       WHERE papers_fts MATCH ? ORDER BY bm25(papers_fts, 5.0, 1.0, 2.0)
                                       LIMIT 300""", (fq,)).fetchall()
            lists.append([r[0] for r in rows if self.year_ok(r[1] or 0, since, until) and ok(r[0])][:200])
        hits = self.passages(q, k=200, per_paper=1, since=since, until=until, rerank=False, rewrite=mode,
                             allowed=allowed)
        lists.append([h["paper_key"] for h in hits])
        best_passage = {h["paper_key"]: h for h in hits}
        fused = {}
        for ranked in lists:
            for rank, key in enumerate(ranked):
                fused[key] = fused.get(key, 0) + 1 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)
        if allowed is not None:       # matches no list reached still belong in the result
            rest = [key for key in allowed if key not in fused and key in self.papers
                    and self.year_ok(self.papers[key]["year"] or 0, since, until)]
            top += sorted(rest, key=lambda key: -(self.papers[key]["year"] or 0))
        return [(self.papers[key], best_passage.get(key)) for key in top[:k]]

    def similar(self, key, k=10):
        idx = np.where(self.paper_keys == key)[0]
        if not len(idx):
            sys.exit(f"Unknown paper key {key!r} (format: v20/13-124)")
        s = self.paper_emb @ self.paper_emb[idx[0]]
        return [(self.papers[self.paper_keys[i]], float(s[i])) for i in np.argsort(-s)[1:k + 1]]


# ---------------------------------------------------------------- Claude

class ClaudeError(Exception):
    """A Claude call failed; the message is meant for the user."""


PRICES = {"haiku": (0.10, 0.50), "opus": (4.0, 20.0)}


def stream_claude(model_name, system, user_text, history=None, render=None):
    """Core Claude call used by the terminal and the full-screen app. render(text_iterator) consumes the
    streamed text (default: Markdown in the terminal). Returns (final message, cost in USD); raises
    ClaudeError with a user-facing message on failure."""
    import anthropic
    try:
        client = anthropic.Anthropic()
    except Exception as e:
        raise ClaudeError(f"Could not create Anthropic client ({e}). Set ANTHROPIC_API_KEY - see README.md.")
    render = render or ui.stream_answer
    model = MODELS[model_name]
    common = dict(model=model, max_tokens=16000, system=system,
                  messages=[*(history or []), {"role": "user", "content": user_text}])
    try:
        if model_name == "opus":
            # Opus 5.5: server-side fallback re-runs a safety-declined request on a fallback model
            stream_ctx = client.beta.messages.stream(**common, output_config={"effort": "high"},
                                                     betas=["server-side-fallback-2026-07-01"],
                                                     fallbacks="default")
        else:
            stream_ctx = client.messages.stream(**common, output_config={"effort": "medium"})
        with stream_ctx as stream:
            render(stream.text_stream)
            msg = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise ClaudeError(f"Anthropic API key rejected - check ANTHROPIC_API_KEY in {ENV_FILE}.")
    except TypeError as e:
        if "authentication" not in str(e).lower():
            raise
        raise ClaudeError(f"No Anthropic API key found. Put it in {ENV_FILE} as ANTHROPIC_API_KEY=sk-ant-... "
                          "('search', 'similar' and 'open' work without one).")
    except anthropic.RateLimitError:
        raise ClaudeError("Rate limited by the Anthropic API - wait a minute and retry.")
    except anthropic.APIConnectionError:
        raise ClaudeError("Could not reach the Anthropic API - check your internet connection.")
    except anthropic.APIStatusError as e:
        raise ClaudeError(f"Anthropic API error {e.status_code}: {e.message}")
    u = msg.usage
    price = PRICES[model_name]
    cost = (u.input_tokens * price[0] + u.output_tokens * price[1]) / 1e6
    ui.add_cost(cost)
    return msg, cost


def call_claude(model_name, system, user_text, history=None):
    """Stream Claude's answer to the terminal. Returns the final message.
    history: optional earlier messages (alternating user/assistant, starting with user) sent before
    user_text, for multi-turn chat."""
    try:
        msg, cost = stream_claude(model_name, system, user_text, history)
    except ClaudeError as e:
        sys.exit(f"\n{e}")
    if msg.stop_reason == "refusal":
        ui.note("[The model declined to answer this request.]", style="yellow")
    elif msg.stop_reason == "max_tokens":
        ui.note("[Answer was cut off at the length limit.]", style="yellow")
    u = msg.usage
    ui.note(f"\n({msg.model}: {u.input_tokens} in / {u.output_tokens} out tokens, ~${cost:.4f})")
    return msg


# ---------------------------------------------------------------- commands

def paper_line(p):
    return f"{p['title']} ({p['year']}) - {p['authors']}"


def pdf_path(p):
    return str(JMLR_DIR / p["pdf_path"]).replace("/", os.sep)


def wrap(s, n=300, indent="     "):
    s = s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " ..."
    return textwrap.fill(s, 100, initial_indent=indent, subsequent_indent=indent)


def boolean_search(ix, q, k=10, since=None, until=None):
    """Scholar-style query (boolq.py): papers matching every term, ranked by hybrid relevance to the query's
    words. If none match, the papers matching the most terms, best tier first.
    Returns (boolq.Result, [(paper, best passage, [missing terms])])."""
    import boolq
    r = boolq.match(ix.con, q)
    text = boolq.plain_text(r.clauses) or q
    if r.strict:
        return r, [(p, hit, []) for p, hit in ix.rank_papers(text, k=k, since=since, until=until,
                                                             allowed=r.strict)]
    missing = {key: miss for key, _, miss in r.partial}
    out = []
    for n in sorted({n for _, n, _ in r.partial}, reverse=True):
        tier = {key for key, m, _ in r.partial if m == n}
        ranked = ix.rank_papers(text, k=k - len(out), since=since, until=until, allowed=tier)
        out += [(p, hit, missing[p["key"]]) for p, hit in ranked]
        if len(out) >= k:
            break
    return r, out


def require_filter(ix, query):
    """--filter for ask/compare: the set of papers matching a Scholar query; exits with term counts if empty."""
    import boolq
    r = boolq.match(ix.con, query)
    if not r.strict:
        sys.exit("No paper matches the filter " + repr(query) + ":\n" + "\n".join(boolq.describe(r))
                 + "\nTry removing a term, or `search` with the same query to see the closest matches.")
    print(f"Filter {query!r}: searching within {len(r.strict)} matching papers.")
    return r.strict


def cmd_search(ix, a):
    import boolq
    import topics
    topic_of = topics.paper_topic_names(ix.con)
    if boolq.is_boolean(a.query):
        r, rows = boolean_search(ix, a.query, k=a.k or 10, since=a.since, until=a.until)
        print("Scholar-style query: " + " AND ".join(f"({c.label()})" for c in r.clauses))
        print("\n".join(boolq.describe(r)))
        if r.strict:
            print(f"\n{len(r.strict)} papers match every term" + (f"; the top {len(rows)}:" if len(rows) <
                                                                 len(r.strict) else ":") + "\n")
        elif rows:
            n_pos = sum(not c.negated for c in r.clauses)
            print(f"\nNo paper matches every term. Closest matches (most terms first, of {n_pos}):\n")
        else:
            print("\nNo paper matches any of the terms.")
    else:
        rows = [(p, hit, []) for p, hit in ix.rank_papers(a.query, k=a.k or 10, since=a.since, until=a.until)]
    for i, (p, hit, missing) in enumerate(rows, 1):
        tag = f"  [topic {topic_of[p['key']]}]" if p["key"] in topic_of else ""
        print(f"{i:2}. {paper_line(p)}\n     [{p['key']}]  {pdf_path(p)}{tag}")
        if missing:
            print(f"     missing: {', '.join(missing)}")
        if hit:
            where = f"p.{hit['page_start']}" + (f", {hit['section']}" if hit.get("section") else "")
            print(wrap(f"{where}: {hit['text']}", 240))
        print()


def source_blocks(ix, hits):
    """The retrieved passages as numbered <source> blocks for the prompt."""
    blocks = []
    for i, h in enumerate(hits, 1):
        p = ix.papers[h["paper_key"]]
        pages = f"{h['page_start']}" + (f"-{h['page_end']}" if h["page_end"] != h["page_start"] else "")
        section = f' section="{h["section"]}"' if h.get("section") else ""
        blocks.append(f'<source id="{i}" title="{p["title"]}" authors="{p["authors"]}" year="{p["year"]}" '
                      f'pages="{pages}"{section}>\n{h["text"]}\n</source>')
    return "<sources>\n" + "\n".join(blocks) + "\n</sources>"


def answer_text(msg):
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def cmd_ask(ix, a):
    allowed = require_filter(ix, a.filter) if getattr(a, "filter", None) else None
    hits = ix.passages(a.query, k=a.k or 12, since=a.since, until=a.until, allowed=allowed)
    if not hits:
        sys.exit("No matching passages found.")
    user = source_blocks(ix, hits) + f"\n\nQuestion: {a.query}"
    print(f"Retrieved {len(hits)} passages from {len({h['paper_key'] for h in hits})} papers. "
          f"Asking {MODELS[a.model]}...\n")
    call_claude(a.model, SYSTEM, user)
    print("\nSources:")
    for i, h in enumerate(hits, 1):
        p = ix.papers[h["paper_key"]]
        sec = f", {h['section']}" if h.get("section") else ""
        print(f" [{i}] {p['title']} ({p['year']}), p.{h['page_start']}{sec}  [{p['key']}]")
    print("\nOpen a paper with:  python ask.py open <key>")


def cmd_reading(ix, a):
    ranked = ix.rank_papers(a.query, k=a.k or 25, since=a.since, until=a.until)
    lines = [f"[{i}] {p['title']} ({p['year']}) - {p['authors']}\nAbstract: {p['abstract'] or '(none)'}"
             for i, (p, _) in enumerate(ranked, 1)]
    print(f"Found {len(ranked)} candidate papers. Asking {MODELS[a.model]} to build a reading path...\n")
    call_claude(a.model, READING_SYSTEM, f"Topic: {a.query}\n\nCandidate papers:\n\n" + "\n\n".join(lines))
    print("\nCandidates:")
    for i, (p, _) in enumerate(ranked, 1):
        print(f" [{i}] {p['title']} ({p['year']})  [{p['key']}]")


def cmd_compare(ix, a):
    import compare
    compare.run(ix, a, call_claude, MODELS)


def cmd_similar(ix, a):
    p0 = ix.papers.get(a.query)
    if p0:
        print(f"Papers similar to: {paper_line(p0)}\n")
    for i, (p, s) in enumerate(ix.similar(a.query, k=a.k or 10), 1):
        print(f"{i:2}. {paper_line(p)}\n     [{p['key']}]  similarity {s:.2f}")


def cmd_open(a):
    con = connect()
    p = con.execute("SELECT * FROM papers WHERE key=?", (a.query,)).fetchone()
    if not p:
        sys.exit(f"Unknown paper key {a.query!r} (format: v20/13-124)")
    path = pdf_path(p)
    print(f"Opening {p['title']}\n  {path}")
    os.startfile(path)


def source_label(ix, h):
    """Short label for a retrieved passage, used in the chat history: title, first author, year, page."""
    p = ix.papers[h["paper_key"]]
    authors = p["authors"] or ""
    first = authors.split(",")[0].strip() + (" et al." if "," in authors else "")
    sec = f", {h['section']}" if h.get("section") else ""
    return f"{p['title']} ({first}, {p['year']}), p.{h['page_start']}{sec}  [{p['key']}]"


def prepare_chat_turn(ix, conv, question, k=None, since=None, until=None):
    """Everything for one conversational turn except the answer call (shared by the line chat and the
    full-screen app): condense a follow-up into a standalone query, retrieve, build the prompt.
    Returns a dict: query, condense_cost, condense_error, hits, user, history, system."""
    from conversation import CHAT_SYSTEM_ADDENDUM
    query, cost, err = conv.condense(question) if len(conv) else (question, 0.0, None)
    hits = ix.passages(query, k=k or 12, since=since, until=until)
    searched = f"\n(These sources were retrieved with the search query: {query})" if query != question else ""
    user = (f"{conv.current_label()}. New sources for this turn (cite these as [n]):\n"
            + source_blocks(ix, hits) + f"{searched}\n\nQuestion: {question}")
    return dict(query=query, condense_cost=cost, condense_error=err, hits=hits, user=user,
                history=conv.history_messages(), system=SYSTEM + CHAT_SYSTEM_ADDENDUM)


def record_chat_turn(ix, conv, question, turn, msg):
    """Save the turn to the conversation (and its session file) unless the answer is empty or refused."""
    answer = answer_text(msg)
    labels = [source_label(ix, h) for h in turn["hits"]]
    if answer and msg.stop_reason != "refusal":
        conv.add(question, turn["query"], answer, labels)
    return labels


def chat_ask(ix, a, conv):
    """One conversational turn in the line chat."""
    turn = prepare_chat_turn(ix, conv, a.query, a.k, a.since, a.until)
    ui.add_cost(turn["condense_cost"])
    if turn["condense_error"]:
        print(f"(Could not condense the follow-up - {turn['condense_error']}; searching for the question as typed.)")
    elif len(conv):
        ui.note(f"Searching for: {turn['query']}  (condensed by {MODELS['haiku']}, "
                f"~${turn['condense_cost']:.4f})", style="cyan")
    hits = turn["hits"]
    if not hits:
        print("No matching passages found.")
        return
    history = turn["history"]
    mem = f", with {len(history) // 2} earlier turn(s) as context" if history else ""
    print(f"Retrieved {len(hits)} passages from {len({h['paper_key'] for h in hits})} papers. "
          f"Asking {MODELS[a.model]}{mem}...\n")
    msg = call_claude(a.model, turn["system"], turn["user"], history=history)
    labels = record_chat_turn(ix, conv, a.query, turn, msg)
    print("\nSources:")
    for i, label in enumerate(labels, 1):
        print(f" [{i}] {label}")
    print("\nOpen a paper with:  /open <key>")


CHAT_HELP = """JMLR assistant with conversation memory. Type a question to get a cited answer; follow-ups
("which of those handles noise?", "how does [2] do it?") build on the earlier turns.
  /new (or /reset)   start a fresh conversation      /opus  /haiku   switch model (memory is kept)
  /sessions   list saved conversations    /resume [N]   continue one (default: the most recent)
              conversations are saved automatically after every answer (data/sessions/)
  /search <query>   /reading <topic>   /compare <question>   /similar <key>   /open <key>
  /topics [a | a.b | rising | for <question>]   browse the topic map
                     one-shot commands: they neither use nor add to the conversation memory
  /help   /quit"""


CHAT_COMMANDS = ["/new", "/reset", "/sessions", "/resume", "/opus", "/haiku", "/search", "/reading", "/compare",
                 "/similar", "/open", "/topics", "/help", "/quit"]


def print_sessions():
    from conversation import list_sessions
    sessions = list_sessions()
    if not sessions:
        print("No saved conversations yet.")
        return
    print("Saved conversations (most recent first):")
    for i, s in enumerate(sessions, 1):
        print(f" {i:>2}. {s['updated'][:16].replace('T', ' ')}  {s['turns']:>2} turns  {s['title']}")
    print("Continue one with:  /resume <number>")


def resume_session(ref):
    """Load a saved conversation by list number, id or 'last'; prints its last exchange. None if not found."""
    from conversation import Conversation, resolve_session
    sid = resolve_session(ref)
    if not sid:
        print("No such saved conversation (see /sessions).")
        return None
    conv = Conversation.load(sid)
    last = conv.turns[-1]
    print(f"Resumed \"{conv.title}\" ({len(conv)} turns). Last question: {last.question}")
    return conv


def cmd_chat(ix, a):
    from conversation import Conversation
    ui.panel(CHAT_HELP, "JMLR assistant")
    state = {"model": a.model}
    conv = (resume_session(a.resume) if getattr(a, "resume", None) else None) or Conversation()

    def toolbar():
        n = len(conv)
        return (f"{MODELS[state['model']]}  |  memory: {n} turn{'' if n == 1 else 's'}  |  "
                f"session ~${ui.SESSION['cost']:.4f}  |  Tab completes, Up = history, /help")

    prompt = ui.ChatPrompt(CHAT_COMMANDS, sorted(ix.papers), toolbar)
    while True:
        model = state["model"]
        try:
            line = prompt.ask(f"[{model} | memory: {len(conv)} turn{'' if len(conv) == 1 else 's'}]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line in ("/opus", "/haiku"):
            state["model"] = model = line[1:]
            print(f"Model set to {MODELS[model]}")
            continue
        if line in ("/new", "/reset"):
            conv = Conversation()
            print("Started a new conversation (the previous one stays saved; /sessions lists them).")
            continue
        if line == "/sessions":
            print_sessions()
            continue
        if line == "/resume" or line.startswith("/resume "):
            conv = resume_session(line[len("/resume"):].strip() or "last") or conv
            continue
        if line in ("/help", "/?"):
            ui.panel(CHAT_HELP, "JMLR assistant")
            continue
        cmd, _, rest = line.partition(" ") if line.startswith("/") else ("/ask", "", line)
        sub = argparse.Namespace(query=rest.strip(), model=model, k=a.k, since=a.since, until=a.until, papers=None)
        fn = {"/search": cmd_search, "/reading": cmd_reading, "/similar": cmd_similar,
              "/compare": cmd_compare}.get(cmd)
        try:
            if cmd == "/ask":
                chat_ask(ix, sub, conv)
            elif cmd == "/open":
                cmd_open(sub)
            elif cmd == "/topics":
                import topics
                arg = sub.query
                if arg.startswith("for "):
                    topics.for_query(ix, arg[4:].strip())
                else:
                    topics.show("" if arg in ("", "rising") else arg, rising=arg == "rising")
            elif fn:
                fn(ix, sub)
            else:
                print("Unknown command (/help lists them).")
        except SystemExit as e:
            print(e)


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default=None,
                    choices=["search", "ask", "reading", "compare", "similar", "open", "cite", "topics", "chat",
                             "app", "update", "gui"],
                    help="default: the full-screen app in a terminal, otherwise chat")
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("more_keys", nargs="*", help=argparse.SUPPRESS)   # cite: further paper keys
    ap.add_argument("--out", help="cite: write a UTF-8 .bib file instead of printing")
    ap.add_argument("--opus", dest="model", action="store_const", const="opus", default="haiku",
                    help="use Claude Opus 5.5 instead of Claude Haiku 5.5")
    ap.add_argument("-k", type=int, help="number of passages (ask) or papers (search/reading/similar/compare)")
    ap.add_argument("--papers", nargs="+", metavar="KEY", help="compare: these papers instead of the top-ranked")
    ap.add_argument("--filter", metavar="QUERY",
                    help='ask/compare: only papers matching a Scholar-style query, e.g. \'federated "differential privacy" -bandit\'')
    ap.add_argument("--since", type=int, help="only papers from this year on")
    ap.add_argument("--until", type=int, help="only papers up to this year")
    ap.add_argument("--no-rewrite", action="store_true",
                    help="don't rewrite the question into ML terminology with Claude before searching")
    ap.add_argument("--resume", nargs="?", const="last", metavar="N",
                    help="chat: continue a saved conversation (default: the most recent)")
    ap.add_argument("--rising", action="store_true", help="topics: subtopics that grew or shrank most")
    ap.add_argument("--check", action="store_true", help="update: only report what JMLR has that isn't indexed")
    ap.add_argument("--browser", action="store_true", help="gui: serve in the web browser instead of a window")
    ap.add_argument("--for", dest="for_query", metavar="QUESTION", help="topics: which subtopics a question falls into")
    a = ap.parse_args()
    if a.command is None:
        a.command = "app" if (sys.stdin.isatty() and sys.stdout.isatty()) else "chat"
    if a.command == "update":
        import update
        return update.run(check_only=a.check)
    if a.command == "gui":
        from gui import app as gui_app
        return gui_app.run(native=not a.browser, no_rewrite=a.no_rewrite)
    if a.command == "app":
        try:
            import tui
        except ImportError as e:
            sys.exit(f"The full-screen app needs Textual ({e}); run 'jmlr-rag chat' instead.")
        return tui.run(a)
    if a.command not in ("chat", "topics") and not a.query:
        ap.error(f"'{a.command}' needs a query")
    if a.command == "open":
        return cmd_open(a)
    if a.command == "topics":
        import topics
        if a.for_query:   # needs retrieval, but not the reranker
            with ui.status("Loading the index..."):
                ix = Index(rerank_model=None, rewrite="off" if a.no_rewrite else REWRITE)
            return topics.for_query(ix, a.for_query)
        return topics.show(a.query, rising=a.rising)
    if a.command == "cite":
        import cite
        return cite.run([a.query] + a.more_keys, a.out)
    with ui.status("Loading the index, search model and reranker (about 30 s)..."):
        ix = Index(rewrite="off") if a.no_rewrite else Index()
    {"search": cmd_search, "ask": cmd_ask, "reading": cmd_reading, "compare": cmd_compare,
     "similar": cmd_similar, "chat": cmd_chat}[a.command](ix, a)


if __name__ == "__main__":
    main()
