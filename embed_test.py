"""Compare embedding models on a subset index before committing to a full re-embed (hours on the GPU).

Subset = every paper that is the gold answer of an eval question + N random other papers.
Each candidate model embeds the subset's passages; then meaning-only (dense) retrieval is scored on the
standard and hard eval questions. The current model (bge-small) is scored on the same subset from the
existing embeddings, so the comparison is like for like. The subset is easier than the full collection
(fewer distractors), so read the numbers relative to each other, not as full-index scores.

  python embed_test.py                       all candidates, 300 distractor papers
  python embed_test.py --models bge-base     one candidate
  python embed_test.py --distractors 600

Embeddings are cached in data/embed_test/ so a rerun only scores.
"""
import argparse
import json
import random
import time

import numpy as np

from common import CHUNK_EMB, CHUNK_IDS, DATA_DIR, QUERY_PREFIX, ROOT, connect, utf8_console

CANDIDATES = {
    # name: (model id, query prefix or prompt_name, batch size)
    "bge-base": ("BAAI/bge-base-en-v1.5", {"prefix": QUERY_PREFIX}, 64),
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", {"prompt_name": "query"}, 16),
}
CACHE = DATA_DIR / "embed_test"


def load(model_id):
    import torch
    from sentence_transformers import SentenceTransformer
    cuda = torch.cuda.is_available()
    m = SentenceTransformer(model_id, device="cuda" if cuda else "cpu",
                            model_kwargs={"torch_dtype": torch.float16} if cuda else {})
    m.max_seq_length = 512
    return m


def encode_queries(model, spec, qs):
    if "prompt_name" in spec:
        return model.encode(qs, prompt_name=spec["prompt_name"], normalize_embeddings=True, batch_size=32)
    return model.encode([spec["prefix"] + q for q in qs], normalize_embeddings=True, batch_size=32)


def score(name, emb, ids, chunk_paper, chunk_seq, qemb, qs, out):
    """Dense-only retrieval over the subset: paper@12, passage@12 (3 per paper cap), gold passage in top 100."""
    sims = qemb @ emb.T
    res = {"standard": [], "hard": []}
    for qi, q in enumerate(qs):
        order = np.argsort(-sims[qi])
        top100 = order[:100]
        picked, per = [], {}
        for j in order:
            pk = chunk_paper[j]
            if per.get(pk, 0) >= 3:
                continue
            per[pk] = per.get(pk, 0) + 1
            picked.append(j)
            if len(picked) == 12:
                break
        gold = lambda j: chunk_paper[j] == q["gold_paper"] and abs(chunk_seq[j] - q["gold_seq"]) <= 1  # noqa
        res[q.get("difficulty", "standard")].append((
            any(chunk_paper[j] == q["gold_paper"] for j in picked),
            any(gold(j) for j in picked),
            any(gold(j) for j in top100)))
    row = {"model": name}
    for g, r in res.items():
        row[g] = tuple(sum(x[i] for x in r) / len(r) for i in range(3))
    out.append(row)
    print(f"{name:<12} standard paper/passage@12/passage@100 "
          f"{row['standard'][0]:.0%} / {row['standard'][1]:.0%} / {row['standard'][2]:.0%}    "
          f"hard {row['hard'][0]:.0%} / {row['hard'][1]:.0%} / {row['hard'][2]:.0%}", flush=True)


class QueryEncoder:
    """Stands in for Index.model so passages() embeds queries with a candidate model: the bge query prefix
    passages() adds is swapped for the candidate's own query prompt."""

    def __init__(self, model, spec):
        self.model, self.spec = model, spec

    def encode(self, texts, **kw):
        out = []
        for t in texts:
            if t.startswith(QUERY_PREFIX):
                out.append(encode_queries(self.model, self.spec, [t[len(QUERY_PREFIX):]])[0])
            else:                                   # document-style text (HyDE), no query prompt
                out.append(self.model.encode([t], normalize_embeddings=True)[0])
        return np.asarray(out, dtype=np.float32)


def pipeline(a):
    """End-to-end comparison on the subset: the complete retrieval pipeline (query rewrite, dense + BM25,
    RRF, reranker), restricted to the subset papers, once with bge-small vectors and once with a
    candidate's cached vectors. Only the embedding model differs."""
    from ask import Index
    con = connect()
    qs = [json.loads(l) for l in open(ROOT / "eval" / "questions.jsonl", encoding="utf-8") if l.strip()]
    qs = [q for q in qs if q["type"] == "passage"]
    gold = {q["gold_paper"] for q in qs}
    others = [r[0] for r in con.execute("SELECT key FROM papers WHERE status='done'") if r[0] not in gold]
    random.Random(a.seed).shuffle(others)
    subset = gold | set(others[: a.distractors])
    rows = [r for r in con.execute("SELECT c.id, c.paper_key FROM chunks c WHERE c.is_ref = 0 ORDER BY c.id")
            if r[1] in subset]
    sub_ids = np.array([r[0] for r in rows])
    print(f"Subset: {len(subset)} papers, {len(sub_ids)} passages; {len(qs)} questions; full pipeline\n")

    ix = Index()

    def run(label):
        res = {"standard": [], "hard": []}
        t0 = time.time()
        for q in qs:
            hits = ix.passages(q["question"], k=12, allowed=subset)
            papers = [h["paper_key"] for h in hits]
            rank = next((i for i, p in enumerate(papers, 1) if p == q["gold_paper"]), None)
            psg = any(h["paper_key"] == q["gold_paper"] and abs(h["seq"] - q["gold_seq"]) <= 1 for h in hits)
            res[q.get("difficulty", "standard")].append((rank is not None, psg, 1 / rank if rank else 0.0))
        m = lambda g, i: sum(x[i] for x in res[g]) / len(res[g])  # noqa: E731
        print(f"{label:<12} standard paper/passage/MRR {m('standard', 0):.0%} / {m('standard', 1):.0%} / "
              f"{m('standard', 2):.2f}    hard {m('hard', 0):.0%} / {m('hard', 1):.0%} / {m('hard', 2):.2f}"
              f"    ({(time.time() - t0) / len(qs):.1f} s/q)", flush=True)

    run("bge-small")
    for name in a.models:
        cache = CACHE / f"{name}_d{a.distractors}_s{a.seed}.npy"
        if not cache.exists():
            print(f"{name}: no cached subset embeddings - run without --pipeline first")
            continue
        emb = np.load(cache).astype(np.float32)
        cached_ids = np.array([r[0] for r in con.execute(
            "SELECT c.id, c.paper_key FROM chunks c WHERE c.is_ref = 0 ORDER BY c.id") if r[1] in subset])
        if len(cached_ids) != len(emb):
            print(f"{name}: cached embeddings don't match the current subset ({len(emb)} vs {len(cached_ids)})")
            continue
        model_id, spec, _ = CANDIDATES[name]
        key_of = dict(con.execute("SELECT id, paper_key FROM chunks").fetchall())
        year = {k: (p["year"] or 0) for k, p in ix.papers.items()}
        ix.chunk_ids, ix.chunk_emb = cached_ids, emb
        ix.chunk_paper = np.array([key_of[int(i)] for i in cached_ids])
        ix.chunk_year = np.array([year.get(k, 0) for k in ix.chunk_paper])
        ix.model = QueryEncoder(load(model_id), spec)
        run(name)


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CANDIDATES), choices=list(CANDIDATES))
    ap.add_argument("--distractors", type=int, default=300)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--pipeline", action="store_true",
                    help="end-to-end comparison with the full retrieval pipeline (uses cached subset embeddings)")
    a = ap.parse_args()
    if a.pipeline:
        return pipeline(a)

    con = connect()
    qs = [json.loads(l) for l in open(ROOT / "eval" / "questions.jsonl", encoding="utf-8") if l.strip()]
    qs = [q for q in qs if q["type"] == "passage"]
    gold = {q["gold_paper"] for q in qs}
    others = [r[0] for r in con.execute("SELECT key FROM papers WHERE status='done'") if r[0] not in gold]
    random.Random(a.seed).shuffle(others)
    subset = gold | set(others[: a.distractors])
    rows = [r for r in con.execute(
        "SELECT c.id, c.paper_key, c.seq, c.text, p.title FROM chunks c JOIN papers p ON p.key = c.paper_key "
        "WHERE c.is_ref = 0 ORDER BY c.id") if r[1] in subset]
    ids = np.array([r[0] for r in rows])
    chunk_paper = np.array([r[1] for r in rows])
    chunk_seq = np.array([r[2] for r in rows])
    docs = [f"{r[4]}: {r[3]}" for r in rows]           # same document text as embed.py
    print(f"Subset: {len(subset)} papers ({len(gold)} gold + {a.distractors} distractors), {len(rows)} passages; "
          f"{len(qs)} questions\n")

    results = []
    # baseline: current bge-small embeddings, same subset
    all_ids, all_emb = np.load(CHUNK_IDS), np.load(CHUNK_EMB)
    pos = {int(i): n for n, i in enumerate(all_ids)}
    keep = [n for n, i in enumerate(ids) if int(i) in pos]
    base = all_emb[[pos[int(ids[n])] for n in keep]].astype(np.float32)
    from common import load_encoder
    small, _ = load_encoder()
    qb = small.encode([QUERY_PREFIX + q["question"] for q in qs], normalize_embeddings=True)
    del small
    score("bge-small", base, ids[keep], chunk_paper[keep], chunk_seq[keep], qb, qs, results)

    CACHE.mkdir(parents=True, exist_ok=True)
    for name in a.models:
        model_id, spec, bs = CANDIDATES[name]
        cache = CACHE / f"{name}_d{a.distractors}_s{a.seed}.npy"
        model = load(model_id)
        if cache.exists():
            emb = np.load(cache).astype(np.float32)
        else:
            t0 = time.time()
            emb = model.encode(docs, batch_size=bs, normalize_embeddings=True, show_progress_bar=False,
                               convert_to_numpy=True)
            np.save(cache, emb.astype(np.float16))
            rate = len(docs) / (time.time() - t0)
            print(f"  {name}: embedded {len(docs)} passages at {rate:.0f}/s "
                  f"(full collection of ~250k would take ~{250_000 / rate / 3600:.1f} h)", flush=True)
        qe = encode_queries(model, spec, [q["question"] for q in qs])
        score(name, emb.astype(np.float32), ids, chunk_paper, chunk_seq, np.asarray(qe, dtype=np.float32), qs,
              results)
        del model
        import torch
        torch.cuda.empty_cache()

    (CACHE / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
