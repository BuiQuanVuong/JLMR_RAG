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

    def passages(self, q, k=12, per_paper=3, since=None, until=None, pool=200, mode="hybrid", rerank=True,
                 rewrite=None):
        """Hybrid retrieval: dense (bge) + BM25 (SQLite FTS5), fused with reciprocal rank fusion, then
        (optionally) the top RERANK_POOL candidates re-scored by a cross-encoder against the ORIGINAL q.
        rewrite = None (use the Index setting) or one of REWRITE_MODES: which retrieval lists use the
        Claude rewrite of q. mode = 'hybrid' | 'dense' | 'sparse' (single-signal modes, for evaluation)."""
        dense_q, sparse_q = self.query_variants(q, rewrite)
        lists = []
        if mode in ("hybrid", "dense"):
            vecs = self.model.encode([(QUERY_PREFIX + t) if is_q else t for t, is_q in dense_q],
                                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            mask = np.ones(len(self.chunk_ids), bool)
            if since:
                mask &= self.chunk_year >= since
            if until:
                mask &= self.chunk_year <= until
            for qv in vecs:
                scores = self.chunk_emb @ qv
                scores[~mask] = -1
                lists.append(self.chunk_ids[np.argsort(-scores)[:pool]].tolist())

        for text in (sparse_q if mode in ("hybrid", "sparse") else []):
            fq = self.fts_query(text)
            if not fq:
                continue
            rows = self.con.execute("""SELECT c.id, p.year FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
                                       JOIN papers p ON p.key = c.paper_key
                                       WHERE chunks_fts MATCH ? AND c.is_ref = 0
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

    def rank_papers(self, q, k=10, since=None, until=None):
        """Paper ranking: fuse title/abstract similarity, abstract keyword match and best passage hits.
        With rewrite_search, the Claude rewrite adds its own title/abstract lists and passage lists."""
        mode = self.rewrite if self.rewrite_search else "off"
        dense_q, sparse_q = self.query_variants(q, mode)
        lists = []
        vecs = self.model.encode([(QUERY_PREFIX + t) if is_q else t for t, is_q in dense_q],
                                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        for qv in vecs:
            ps = self.paper_emb @ qv
            lists.append([self.paper_keys[i] for i in np.argsort(-ps)
                          if self.year_ok(self.paper_year[i], since, until)][:200])
        for text in sparse_q:
            fq = self.fts_query(text)
            if not fq:
                continue
            rows = self.con.execute("""SELECT p.key, p.year FROM papers_fts JOIN papers p ON p.rowid = papers_fts.rowid
                                       WHERE papers_fts MATCH ? ORDER BY bm25(papers_fts, 5.0, 1.0, 2.0)
                                       LIMIT 300""", (fq,)).fetchall()
            lists.append([r[0] for r in rows if self.year_ok(r[1] or 0, since, until)][:200])
        hits = self.passages(q, k=200, per_paper=1, since=since, until=until, rerank=False, rewrite=mode)
        lists.append([h["paper_key"] for h in hits])
        best_passage = {h["paper_key"]: h for h in hits}
        fused = {}
        for ranked in lists:
            for rank, key in enumerate(ranked):
                fused[key] = fused.get(key, 0) + 1 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(self.papers[key], best_passage.get(key)) for key in top]

    def similar(self, key, k=10):
        idx = np.where(self.paper_keys == key)[0]
        if not len(idx):
            sys.exit(f"Unknown paper key {key!r} (format: v20/13-124)")
        s = self.paper_emb @ self.paper_emb[idx[0]]
        return [(self.papers[self.paper_keys[i]], float(s[i])) for i in np.argsort(-s)[1:k + 1]]


# ---------------------------------------------------------------- Claude

def call_claude(model_name, system, user_text, history=None):
    """Stream Claude's answer to the terminal. Returns the final message.
    history: optional earlier messages (alternating user/assistant, starting with user) sent before
    user_text, for multi-turn chat."""
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as e:
        sys.exit(f"Could not create Anthropic client ({e}). Set ANTHROPIC_API_KEY first - see README.md.")
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
            for text in stream.text_stream:
                print(text, end="", flush=True)
            msg = stream.get_final_message()
    except anthropic.AuthenticationError:
        sys.exit(f"\nAnthropic API key rejected - check ANTHROPIC_API_KEY in {ENV_FILE}.")
    except TypeError as e:
        if "authentication" not in str(e).lower():
            raise
        sys.exit(f"No Anthropic API key found. Put it in {ENV_FILE} as ANTHROPIC_API_KEY=sk-ant-... "
                 "('search', 'similar' and 'open' work without one).")
    except anthropic.RateLimitError:
        sys.exit("\nRate limited by the Anthropic API - wait a minute and retry.")
    except anthropic.APIConnectionError:
        sys.exit("\nCould not reach the Anthropic API - check your internet connection.")
    except anthropic.APIStatusError as e:
        sys.exit(f"\nAnthropic API error {e.status_code}: {e.message}")
    print()
    if msg.stop_reason == "refusal":
        print("[The model declined to answer this request.]")
    elif msg.stop_reason == "max_tokens":
        print("[Answer was cut off at the length limit.]")
    u = msg.usage
    price = {"haiku": (0.10, 0.50), "opus": (4.0, 20.0)}[model_name]
    cost = (u.input_tokens * price[0] + u.output_tokens * price[1]) / 1e6
    print(f"\n({msg.model}: {u.input_tokens} in / {u.output_tokens} out tokens, ~${cost:.4f})")
    return msg


# ---------------------------------------------------------------- commands

def paper_line(p):
    return f"{p['title']} ({p['year']}) - {p['authors']}"


def pdf_path(p):
    return str(JMLR_DIR / p["pdf_path"]).replace("/", os.sep)


def wrap(s, n=300, indent="     "):
    s = s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " ..."
    return textwrap.fill(s, 100, initial_indent=indent, subsequent_indent=indent)


def cmd_search(ix, a):
    for i, (p, hit) in enumerate(ix.rank_papers(a.query, k=a.k or 10, since=a.since, until=a.until), 1):
        print(f"{i:2}. {paper_line(p)}\n     [{p['key']}]  {pdf_path(p)}")
        if hit:
            where = f"p.{hit['page_start']}" + (f", {hit['section']}" if hit.get("section") else "")
            print(wrap(f"{where}: {hit['text']}", 240))
        print()


def cmd_ask(ix, a):
    hits = ix.passages(a.query, k=a.k or 12, since=a.since, until=a.until)
    if not hits:
        sys.exit("No matching passages found.")
    blocks = []
    for i, h in enumerate(hits, 1):
        p = ix.papers[h["paper_key"]]
        pages = f"{h['page_start']}" + (f"-{h['page_end']}" if h["page_end"] != h["page_start"] else "")
        section = f' section="{h["section"]}"' if h.get("section") else ""
        blocks.append(f'<source id="{i}" title="{p["title"]}" authors="{p["authors"]}" year="{p["year"]}" '
                      f'pages="{pages}"{section}>\n{h["text"]}\n</source>')
    user = "<sources>\n" + "\n".join(blocks) + f"\n</sources>\n\nQuestion: {a.query}"
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


def chat_ask(ix, a, conv):
    """One conversational turn: condense a follow-up into a standalone query, retrieve, answer with the
    earlier turns as context, and record the turn in conv."""
    from conversation import CHAT_SYSTEM_ADDENDUM
    query = a.query
    if len(conv):
        query, cost, err = conv.condense(a.query)
        if err:
            print(f"(Could not condense the follow-up - {err}; searching for the question as typed.)")
        else:
            print(f"Searching for: {query}  (condensed by {MODELS['haiku']}, ~${cost:.4f})")
    hits = ix.passages(query, k=a.k or 12, since=a.since, until=a.until)
    if not hits:
        print("No matching passages found.")
        return
    blocks = []
    for i, h in enumerate(hits, 1):
        p = ix.papers[h["paper_key"]]
        pages = f"{h['page_start']}" + (f"-{h['page_end']}" if h["page_end"] != h["page_start"] else "")
        section = f' section="{h["section"]}"' if h.get("section") else ""
        blocks.append(f'<source id="{i}" title="{p["title"]}" authors="{p["authors"]}" year="{p["year"]}" '
                      f'pages="{pages}"{section}>\n{h["text"]}\n</source>')
    searched = f"\n(These sources were retrieved with the search query: {query})" if query != a.query else ""
    user = (f"{conv.current_label()}. New sources for this turn (cite these as [n]):\n<sources>\n"
            + "\n".join(blocks) + f"\n</sources>{searched}\n\nQuestion: {a.query}")
    history = conv.history_messages()
    mem = f", with {len(history) // 2} earlier turn(s) as context" if history else ""
    print(f"Retrieved {len(hits)} passages from {len({h['paper_key'] for h in hits})} papers. "
          f"Asking {MODELS[a.model]}{mem}...\n")
    msg = call_claude(a.model, SYSTEM + CHAT_SYSTEM_ADDENDUM, user, history=history)
    answer = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    labels = [source_label(ix, h) for h in hits]
    if answer and msg.stop_reason != "refusal":
        conv.add(a.query, query, answer, labels)
    print("\nSources:")
    for i, label in enumerate(labels, 1):
        print(f" [{i}] {label}")
    print("\nOpen a paper with:  /open <key>")


CHAT_HELP = """JMLR assistant with conversation memory. Type a question to get a cited answer; follow-ups
("which of those handles noise?", "how does [2] do it?") build on the earlier turns.
  /new (or /reset)   start a fresh conversation      /opus  /haiku   switch model (memory is kept)
  /search <query>   /reading <topic>   /compare <question>   /similar <key>   /open <key>
                     one-shot commands: they neither use nor add to the conversation memory
  /help   /quit"""


def cmd_chat(ix, a):
    from conversation import Conversation
    print(CHAT_HELP)
    model = a.model
    conv = Conversation()
    while True:
        try:
            line = input(f"\n[{model} | memory: {len(conv)} turn{'' if len(conv) == 1 else 's'}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line in ("/opus", "/haiku"):
            model = line[1:]
            print(f"Model set to {MODELS[model]}")
            continue
        if line in ("/new", "/reset"):
            conv.reset()
            print("Started a new conversation (memory cleared).")
            continue
        if line in ("/help", "/?"):
            print(CHAT_HELP)
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
            elif fn:
                fn(ix, sub)
            else:
                print("Unknown command (/help lists them).")
        except SystemExit as e:
            print(e)


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["search", "ask", "reading", "compare", "similar", "open", "cite", "chat"])
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("more_keys", nargs="*", help=argparse.SUPPRESS)   # cite: further paper keys
    ap.add_argument("--out", help="cite: write a UTF-8 .bib file instead of printing")
    ap.add_argument("--opus", dest="model", action="store_const", const="opus", default="haiku",
                    help="use Claude Opus 5.5 instead of Claude Haiku 5.5")
    ap.add_argument("-k", type=int, help="number of passages (ask) or papers (search/reading/similar/compare)")
    ap.add_argument("--papers", nargs="+", metavar="KEY", help="compare: these papers instead of the top-ranked")
    ap.add_argument("--since", type=int, help="only papers from this year on")
    ap.add_argument("--until", type=int, help="only papers up to this year")
    ap.add_argument("--no-rewrite", action="store_true",
                    help="don't rewrite the question into ML terminology with Claude before searching")
    a = ap.parse_args()
    if a.command != "chat" and not a.query:
        ap.error(f"'{a.command}' needs a query")
    if a.command == "open":
        return cmd_open(a)
    if a.command == "cite":
        import cite
        return cite.run([a.query] + a.more_keys, a.out)
    ix = Index(rewrite="off") if a.no_rewrite else Index()
    {"search": cmd_search, "ask": cmd_ask, "reading": cmd_reading, "compare": cmd_compare,
     "similar": cmd_similar, "chat": cmd_chat}[a.command](ix, a)


if __name__ == "__main__":
    main()
