"""Score retrieval quality against eval/questions.jsonl (built by make_eval.py).

Free and local: no API calls. Run it before and after every retrieval change and compare.

  python eval.py                      score the current system, label 'hybrid'
  python eval.py --label reranker     name this run (shows up in eval/history.csv)
  python eval.py --modes              also score keyword-only and meaning-only search

Question groups
  standard  passage questions that reuse the paper's own terms (easy for keyword search)
  hard      passage questions in plain words, without the paper's terminology (the realistic case)
  topic     short "find papers about ..." queries (the `search` pipeline)

Metrics
  paper@12    the gold paper is among the 12 retrieved passages' papers (the `ask` pipeline)
  passage@12  a retrieved passage overlaps the gold passage (same or neighbouring chunk)
  MRR         1 / rank of the first gold-paper hit (0 if absent), averaged
  R@10, R@3   topic: the gold paper is in the top 10 / top 3 papers
"""
import argparse
import csv
import json
import time
from datetime import datetime

from ask import Index
from common import ROOT, utf8_console

EVAL = ROOT / "eval"


def score_passage_q(ix, q, mode):
    hits = ix.passages(q["question"], k=12, mode=mode)
    papers = [h["paper_key"] for h in hits]
    rank = next((i for i, p in enumerate(papers, 1) if p == q["gold_paper"]), None)
    passage_hit = any(h["paper_key"] == q["gold_paper"] and abs(h["seq"] - q["gold_seq"]) <= 1 for h in hits)
    return dict(paper_hit=rank is not None, passage_hit=passage_hit, top3=bool(rank and rank <= 3),
                rr=1 / rank if rank else 0.0, rank=rank or "", top=papers[0] if papers else "")


def score_topic_q(ix, q):
    ranked = [p["key"] for p, _ in ix.rank_papers(q["question"], k=10)]
    rank = next((i for i, k in enumerate(ranked, 1) if k == q["gold_paper"]), None)
    return dict(paper_hit=rank is not None, passage_hit="", top3=bool(rank and rank <= 3),
                rr=1 / rank if rank else 0.0, rank=rank or "", top=ranked[0] if ranked else "")


def mean(rows, field):
    return sum(bool(r[field]) if field != "rr" else r[field] for r in rows) / len(rows) if rows else None


def group(q):
    return "topic" if q["type"] == "topic" else q.get("difficulty", "standard")


def main():
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="hybrid")
    ap.add_argument("--modes", action="store_true", help="also run dense-only and sparse-only")
    a = ap.parse_args()

    qs = [json.loads(l) for l in open(EVAL / "questions.jsonl", encoding="utf-8") if l.strip()]
    ix = Index()

    runs = [("hybrid", a.label)] + ([("dense", "dense-only"), ("sparse", "keyword-only")] if a.modes else [])
    summary = []
    for mode, label in runs:
        rows, t0, n_pass = [], time.time(), 0
        for q in qs:
            if q["type"] == "topic":
                if mode != "hybrid":
                    continue
                res = score_topic_q(ix, q)
            else:
                res = score_passage_q(ix, q, mode)
                n_pass += 1
            rows.append(dict(id=q["id"], group=group(q), question=q["question"], gold=q["gold_paper"], **res))
        secs = (time.time() - t0) / max(len(rows), 1)
        g = {name: [r for r in rows if r["group"] == name] for name in ("standard", "hard", "topic")}
        s = {"label": label,
             "std_n": len(g["standard"]), "std_paper@12": mean(g["standard"], "paper_hit"),
             "std_passage@12": mean(g["standard"], "passage_hit"), "std_MRR": mean(g["standard"], "rr"),
             "hard_n": len(g["hard"]), "hard_paper@12": mean(g["hard"], "paper_hit"),
             "hard_passage@12": mean(g["hard"], "passage_hit"), "hard_MRR": mean(g["hard"], "rr"),
             "topic_n": len(g["topic"]), "topic_R@10": mean(g["topic"], "paper_hit"),
             "topic_R@3": mean(g["topic"], "top3"), "topic_MRR": mean(g["topic"], "rr"),
             "sec_per_query": secs}
        summary.append(s)
        with open(EVAL / f"results_{label}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    pct = lambda v: f"{v:.0%}" if v is not None else "-"   # noqa: E731
    num = lambda v: f"{v:.2f}" if v is not None else "-"   # noqa: E731
    print(f"\n{'':<14}{'--- standard ---':^22}{'------ hard ------':^24}{'---- topic ----':^20}")
    print(f"{'run':<14}{'paper':>7}{'passage':>8}{'MRR':>7}  {'paper':>7}{'passage':>8}{'MRR':>7}  "
          f"{'R@10':>6}{'R@3':>6}{'MRR':>6}{'s/q':>7}")
    for s in summary:
        print(f"{s['label']:<14}{pct(s['std_paper@12']):>7}{pct(s['std_passage@12']):>8}{num(s['std_MRR']):>7}  "
              f"{pct(s['hard_paper@12']):>7}{pct(s['hard_passage@12']):>8}{num(s['hard_MRR']):>7}  "
              f"{pct(s['topic_R@10']):>6}{pct(s['topic_R@3']):>6}{num(s['topic_MRR']):>6}{s['sec_per_query']:>7.2f}")
    s0 = summary[0]
    print(f"\nQuestions: {s0['std_n']} standard, {s0['hard_n']} hard, {s0['topic_n']} topic. "
          f"paper/passage = hit within the 12 passages sent to Claude. Details: eval/results_<run>.csv")

    hist = EVAL / "history.csv"
    new = not hist.exists()
    with open(hist, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time"] + list(s0.keys()))
        if new:
            w.writeheader()
        for s in summary:
            w.writerow({"time": datetime.now().isoformat(timespec="seconds"),
                        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}})


if __name__ == "__main__":
    main()
