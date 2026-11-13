"""Comparison mode: two-stage, citation-grounded synthesis across several papers.

Stage 1 (per paper): retrieve the best passages *within* each paper (hybrid search + reranker),
then Claude Haiku 5.5 extracts a structured evidence record: problem, approach, setting,
findings with citations, limitations, relevance. Records are built in parallel.
Stage 2: Claude (Haiku 5.5, or Opus 5.5 with --opus) compares the records: short answer,
comparison table, agreements and tensions, open questions. Citations look like [P2.3] =
paper 2, passage 3, and are listed with page and section at the end.

Keeping one record per paper makes it much harder to attribute a finding to the wrong paper.

  python ask.py compare "How do these approaches choose the kernel bandwidth?"
  python ask.py compare "..." -k 4                       compare the top 4 papers (default 6)
  python ask.py compare "..." --papers v13/gretton12a v24/21-1289
  python ask.py compare "..." --since 2015 --opus
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

PASSAGES_PER_PAPER = 5
EXTRACT_MODEL = "claude-haiku-5-5"
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.0, 20.0)}

RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "problem": {"type": "string"},
        "approach": {"type": "string"},
        "setting": {"type": "string"},
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {"claim": {"type": "string"},
                           "sources": {"type": "array", "items": {"type": "integer"}}},
            "required": ["claim", "sources"], "additionalProperties": False}},
        "limitations": {"type": "string"},
        "relevance": {"type": "string"},
    },
    "required": ["relevant", "problem", "approach", "setting", "findings", "limitations", "relevance"],
    "additionalProperties": False,
}

EXTRACT_PROMPT = """You are building an evidence record for one research paper, for a later comparison of \
several papers on this question:

Question: {question}

Paper: {title} ({year}) - {authors}
Abstract: {abstract}

Retrieved passages from this paper (numbered):
{passages}

Fill the record using ONLY the abstract and passages above. Be concise and concrete (names of methods, \
assumptions, datasets, rates, numbers).
- problem: what problem the paper addresses (1-2 sentences).
- approach: the method or idea it proposes or uses, as it bears on the question (1-3 sentences).
- setting: assumptions, data or experimental setup that its results depend on (1-2 sentences, "" if not shown).
- findings: 1-4 key findings relevant to the question, each with the passage numbers that support it \
(use [] only if it comes from the abstract).
- limitations: limitations or caveats stated or evident in the passages ("" if none shown).
- relevance: one sentence on how this paper bears on the question.
- relevant: false if the passages show the paper does not actually address the question.
Do not invent anything that is not in the text."""

SYNTH_SYSTEM = """You compare research papers from the Journal of Machine Learning Research for a reader. \
You are given one structured evidence record per paper (P1, P2, ...); citations inside the records look like \
[P2.3] = paper 2, passage 3.

Write, in this order:
1. **Answer** - 2-4 sentences answering the question directly across the papers.
2. **Comparison** - a markdown table with one row per paper (short name + year) and 3-5 columns chosen to fit \
the question (for example approach, assumptions, guarantees or results, evaluation, limitations). Short cells, \
with citations.
3. **Agreements and tensions** - where the papers agree, and where they differ or seem to conflict. Say \
whether an apparent conflict is explained by different settings or assumptions.
4. **Open questions** - gaps or unresolved points suggested by *these* papers. Never claim that no work \
anywhere addresses something; say "among these papers".
5. **What to read** - one line per paper on when to read it.

Rules: use only the records; cite every factual claim with [Pn.m] (or [Pn] for abstract-level claims); \
keep each paper's findings attributed to that paper; skip records marked not relevant except to note them."""


# ---------------------------------------------------------------- retrieval

def pick_papers(ix, question, n, since, until):
    """Top-n distinct papers by reranked passage relevance."""
    hits = ix.passages(question, k=n, per_paper=1, since=since, until=until)
    return [h["paper_key"] for h in hits]


def passages_in_paper(ix, question, key, k=PASSAGES_PER_PAPER, cand=15):
    """Best passages within one paper: dense + keyword candidates, re-scored by the reranker."""
    idx = np.where(ix.chunk_paper == key)[0]
    if not len(idx):
        return []
    qv = ix.embed_query(question)
    dense = ix.chunk_ids[idx[np.argsort(-(ix.chunk_emb[idx] @ qv))[:cand]]].tolist()
    sparse = []
    fq = ix.fts_query(question)
    if fq:
        sparse = [r[0] for r in ix.con.execute(
            """SELECT c.id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
               WHERE chunks_fts MATCH ? AND c.paper_key = ? AND c.is_ref = 0
               ORDER BY bm25(chunks_fts) LIMIT ?""", (fq, key, cand))]
    ids = list(dict.fromkeys(dense + sparse))
    rows = [dict(ix.con.execute("SELECT * FROM chunks WHERE id=?", (i,)).fetchone()) for i in ids]
    if ix.reranker is not None:
        scores = ix.reranker.predict([(question, r["text"]) for r in rows], batch_size=16,
                                     show_progress_bar=False)
        rows = [r for _, r in sorted(zip(scores, rows), key=lambda x: -x[0])]
    top = rows[:k]
    return sorted(top, key=lambda r: r["seq"])          # reading order helps extraction


# ---------------------------------------------------------------- Claude calls

def usage_cost(model, usage):
    p = PRICES.get(model, (0, 0))
    return (usage.input_tokens * p[0] + usage.output_tokens * p[1]) / 1e6


def extract_record(client, question, paper, chunks):
    passages = "\n\n".join(
        f"[{i}] (p.{c['page_start']}{', ' + c['section'] if c.get('section') else ''})\n{c['text']}"
        for i, c in enumerate(chunks, 1))
    prompt = EXTRACT_PROMPT.format(question=question, title=paper["title"], year=paper["year"],
                                   authors=paper["authors"], abstract=paper["abstract"] or "(none)",
                                   passages=passages)
    msg = client.messages.create(
        model=EXTRACT_MODEL, max_tokens=4000,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": RECORD_SCHEMA}},
        messages=[{"role": "user", "content": prompt}])
    cost = usage_cost(EXTRACT_MODEL, msg.usage)
    if msg.stop_reason != "end_turn":
        return None, cost, f"extraction stopped ({msg.stop_reason})"
    return json.loads(next(b.text for b in msg.content if b.type == "text")), cost, None


def format_record(i, paper, rec):
    def cite(nums):
        return "".join(f"[P{i}.{n}]" for n in nums) or f"[P{i}]"
    lines = [f"P{i}: {paper['title']} ({paper['year']}) - {paper['authors']}"]
    if not rec["relevant"]:
        lines.append("  (marked NOT relevant to the question)")
    lines += [f"  Problem: {rec['problem']}", f"  Approach: {rec['approach']}"]
    if rec["setting"]:
        lines.append(f"  Setting: {rec['setting']}")
    lines.append("  Findings:")
    lines += [f"    - {f['claim']} {cite(f['sources'])}" for f in rec["findings"]]
    if rec["limitations"]:
        lines.append(f"  Limitations: {rec['limitations']}")
    lines.append(f"  Relevance: {rec['relevance']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- command

def run(ix, a, call_claude, models):
    import anthropic
    question = a.query
    keys = a.papers or pick_papers(ix, question, a.k or 6, a.since, a.until)
    missing = [k for k in keys if k not in ix.papers]
    if missing:
        sys.exit(f"Unknown paper key(s): {', '.join(missing)} (format: v20/13-124)")
    if len(keys) < 2:
        sys.exit("Need at least 2 papers to compare.")

    print(f"Comparing {len(keys)} papers:")
    for i, k in enumerate(keys, 1):
        p = ix.papers[k]
        print(f"  P{i}  {p['title']} ({p['year']})  [{k}]")
    print("\nStage 1: retrieving evidence within each paper...", flush=True)
    evidence = {k: passages_in_paper(ix, question, k) for k in keys}

    try:
        client = anthropic.Anthropic()
    except Exception as e:
        sys.exit(f"Could not create Anthropic client ({e}).")
    print(f"Stage 1: extracting {len(keys)} evidence records with {EXTRACT_MODEL} (in parallel)...", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda k: extract_record(client, question, ix.papers[k], evidence[k]), keys))
    except TypeError as e:
        if "authentication" in str(e).lower():
            sys.exit("No Anthropic API key found - see README.md (.env).")
        raise
    except anthropic.AuthenticationError:
        sys.exit("Anthropic API key rejected - check .env")
    except anthropic.APIError as e:
        sys.exit(f"Anthropic API error during extraction: {e}")

    stage1_cost = sum(c for _, c, _ in results)
    import ui
    ui.add_cost(stage1_cost)
    records = []
    for i, (k, (rec, _, err)) in enumerate(zip(keys, results), 1):
        if err or rec is None:
            print(f"  P{i}: skipped ({err})")
            continue
        records.append(format_record(i, ix.papers[k], rec))
    if len(records) < 2:
        sys.exit("Too few evidence records to compare.")

    print(f"Stage 2: comparing with {models[a.model]}...\n")
    msg = call_claude(a.model, SYNTH_SYSTEM, f"Question: {question}\n\nEvidence records:\n\n" + "\n\n".join(records))
    total = stage1_cost + usage_cost(models[a.model], msg.usage)
    print(f"(stage 1 extraction ~${stage1_cost:.4f}; total ~${total:.4f})")

    print("\nSources:")
    for i, k in enumerate(keys, 1):
        p = ix.papers[k]
        print(f" P{i}  {p['title']} ({p['year']})  [{k}]")
        for j, c in enumerate(evidence[k], 1):
            sec = f", {c['section']}" if c.get("section") else ""
            print(f"      P{i}.{j}  p.{c['page_start']}{sec}")
    print("\nOpen a paper with:  python ask.py open <key>")
