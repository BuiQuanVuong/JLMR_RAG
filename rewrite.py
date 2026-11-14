"""LLM query rewriting: translate a plain-words question into the terminology an ML paper would use.

One Claude Haiku 5.5 call per question returns
  rewrite       the question restated in standard ML terminology
  keywords      technical terms / phrases a relevant passage would likely contain
  alternatives  2 rewrites that assume different possible terminology
  passage       a short hypothetical passage from a paper answering it (HyDE)
Results are cached in eval/rewrite_cache.json (keyed by prompt version + question), so repeated
runs never re-call the API. Without an API key, or if the call fails, rewrite() returns None and
search silently uses the original question.

  python rewrite.py "a way to pick how wide the similarity function should be"   show a rewrite
  python rewrite.py --eval                    pre-fill the cache for eval/questions.jsonl (+ cost/latency)
"""
import json
import os
import sys
import threading
import time

from common import ROOT

MODEL = "claude-haiku-5-5"
PROMPT_VERSION = "v1"
CACHE_FILE = ROOT / "eval" / "rewrite_cache.json"
PRICE = (0.10, 0.50)   # $ per 1M input / output tokens

SCHEMA = {
    "type": "object",
    "properties": {
        "rewrite": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "alternatives": {"type": "array", "items": {"type": "string"}},
        "passage": {"type": "string"},
    },
    "required": ["rewrite", "keywords", "alternatives", "passage"],
    "additionalProperties": False,
}

PROMPT = """You help search a collection of machine learning research papers (Journal of Machine Learning \
Research). Users often describe an idea in plain words without knowing the field's terminology, so keyword \
and embedding search miss the right passage. Translate the user's question into the vocabulary that a \
machine-learning paper answering it would actually use.

Return:
- rewrite: the question restated in one sentence using standard ML / statistics / optimization terminology \
(name the concepts, methods, problem settings and quantities the user is describing).
- keywords: 6 to 12 technical terms or short phrases a passage answering the question would likely contain \
(include common synonyms and acronyms, e.g. "kernel bandwidth", "median heuristic", "RKHS").
- alternatives: 2 further one-sentence rewrites that assume different plausible terminology or framings.
- passage: a 2-4 sentence excerpt, written as it might appear in such a paper, that answers the question \
(plausible technical content is fine; it is only used for search).

Keep everything specific to the question; do not add unrelated topics.

Question: {question}"""

_lock = threading.Lock()
_cache = None
_client = None
_disabled = False


def _load():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


def _save():
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, CACHE_FILE)
    except Exception:
        pass


def _call(question):
    """One API call. Returns (result dict, meta) or raises."""
    global _client
    import anthropic
    if _client is None:
        _client = anthropic.Anthropic(max_retries=1, timeout=20.0)
    t0 = time.time()
    msg = _client.messages.create(
        model=MODEL, max_tokens=2000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": PROMPT.format(question=question)}])
    secs = time.time() - t0
    if msg.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={msg.stop_reason}")
    data = json.loads(next(b.text for b in msg.content if b.type == "text"))
    u = msg.usage
    meta = {"secs": round(secs, 2), "in": u.input_tokens, "out": u.output_tokens,
            "cost": round((u.input_tokens * PRICE[0] + u.output_tokens * PRICE[1]) / 1e6, 6)}
    return data, meta


def rewrite(question, use_api=True):
    """The cached rewrite of `question` (a dict, see module docstring), or None if unavailable."""
    global _disabled
    key = f"{PROMPT_VERSION}|{question.strip()}"
    with _lock:
        hit = _load().get(key)
    if hit:
        return hit["result"]
    if not use_api or _disabled:
        return None
    try:
        data, meta = _call(question.strip())
    except Exception as e:
        # no key / auth / network problems: stop trying for the rest of this process
        name = type(e).__name__
        if name in ("AuthenticationError", "PermissionDeniedError", "APIConnectionError", "TypeError"):
            _disabled = True
        return None
    with _lock:
        _load()[key] = {"result": data, "meta": meta}
        _save()
    return data


def bm25_text(rw):
    """Text whose terms are OR-ed together for the keyword search."""
    return " ".join([rw["rewrite"], *rw["keywords"]])


def main():
    from common import utf8_console
    import argparse
    ap = argparse.ArgumentParser(description="Show the Claude rewrite of a question, or fill the rewrite cache "
                                 "for the evaluation set.")
    ap.add_argument("--eval", action="store_true", help="rewrite every eval/questions.jsonl question (cached)")
    ap.add_argument("question", nargs="*", help="a question to rewrite")
    a = ap.parse_args()
    utf8_console()
    args = ["--eval"] if a.eval else a.question
    if args[:1] == ["--eval"]:
        from concurrent.futures import ThreadPoolExecutor
        qs = [json.loads(l)["question"] for l in open(ROOT / "eval" / "questions.jsonl", encoding="utf-8")
              if l.strip()]
        before = len(_load())
        with ThreadPoolExecutor(max_workers=4) as pool:
            res = list(pool.map(rewrite, qs))
        metas = [_load()[f"{PROMPT_VERSION}|{q.strip()}"]["meta"] for q in qs
                 if f"{PROMPT_VERSION}|{q.strip()}" in _load()]
        print(f"{sum(r is not None for r in res)}/{len(qs)} rewritten ({len(_load()) - before} new API calls)")
        if metas:
            secs = sorted(m["secs"] for m in metas)
            print(f"latency: mean {sum(secs) / len(secs):.2f}s, median {secs[len(secs) // 2]:.2f}s, "
                  f"max {secs[-1]:.2f}s; cost: mean ${sum(m['cost'] for m in metas) / len(metas):.5f}/question, "
                  f"total ${sum(m['cost'] for m in metas):.4f}; tokens: mean "
                  f"{sum(m['in'] for m in metas) / len(metas):.0f} in / {sum(m['out'] for m in metas) / len(metas):.0f} out")
        return
    q = " ".join(args) or "a way to pick how wide the similarity function should be"
    print(json.dumps(rewrite(q), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
