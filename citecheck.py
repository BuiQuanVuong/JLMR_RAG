"""Citation-accuracy check: do the passages an answer cites actually support its claims?

For a sample of eval questions, the normal `ask` pipeline produces an answer (retrieval + Claude Haiku 5.5 by
default). Then Claude Opus 5.5 audits each answer against the exact numbered passages it was given:
every factual claim is listed with the sources it cites and one verdict:
  supported    the cited passages state it or directly imply it
  partial      they support part of it, or it overstates / over-generalises them
  unsupported  the cited passages don't support it (wrong source, or not in them)
  uncited      a factual claim about the papers with no citation
Sentences the answer itself marks as background or as "not in the sources" are not audited as claims.

  python citecheck.py                  30 questions (15 hard, 15 standard), answers by Haiku
  python citecheck.py --n 10 --opus    answers by Opus instead (to compare)
  python citecheck.py --budget 1.5     stop when the estimated API spend reaches this many dollars

Writes eval/citecheck_<model>.jsonl (every claim) and prints a summary. Costs real money; see --budget.
"""
import argparse
import json
import random
import sys
import time

import anthropic

import ask
import ui
from common import ROOT, utf8_console

JUDGE = "claude-opus-5-5"
JUDGE_PRICE = (4.0, 20.0)       # $ per million input / output tokens

SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "cited": {"type": "array", "items": {"type": "integer"}},
                "verdict": {"type": "string", "enum": ["supported", "partial", "unsupported", "uncited"]},
                "note": {"type": "string"},
            },
            "required": ["claim", "cited", "verdict", "note"], "additionalProperties": False}},
        "honest_about_gaps": {"type": "boolean"},
    },
    "required": ["claims", "honest_about_gaps"], "additionalProperties": False,
}

JUDGE_PROMPT = """You audit a research assistant's answer for citation accuracy. The assistant was given the numbered \
source passages below (from Journal of Machine Learning Research papers) and wrote the answer below, citing \
sources as [n].

{sources}

<answer>
{answer}
</answer>

List every factual claim the answer makes about the papers or their content (split compound sentences into \
separate claims; skip headings, hedges, meta statements such as "the sources don't cover X", reading \
suggestions, and sentences the answer explicitly marks as background knowledge not from the sources). For each \
claim give the source numbers it cites and one verdict:
- supported: the cited passages state it or directly imply it.
- partial: the cited passages support part of it, or it overstates, over-generalises or slightly distorts them.
- unsupported: the cited passages do not support it (wrong source, or the content isn't in them), even if \
another source in the list would.
- uncited: a factual claim about the papers with no citation.
Judge only against the passages shown, not your own knowledge of the papers. Keep each claim under 25 words \
and each note to one short sentence (empty for "supported").
Also set honest_about_gaps: false if the answer presents something as coming from the sources when the \
sources clearly don't contain it, true otherwise."""


def judge(client, sources, answer):
    msg = client.beta.messages.create(
        model=JUDGE, max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(sources=sources, answer=answer)}])
    u = msg.usage
    cost = (u.input_tokens * JUDGE_PRICE[0] + u.output_tokens * JUDGE_PRICE[1]) / 1e6
    if msg.stop_reason != "end_turn":
        return None, cost
    return json.loads(next(b.text for b in msg.content if b.type == "text")), cost


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--opus", action="store_true", help="answers by Claude Opus 5.5 instead of Haiku 5.5")
    ap.add_argument("--budget", type=float, default=2.5, help="stop when estimated spend reaches this ($)")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    model = "opus" if a.opus else "haiku"

    qs = [json.loads(l) for l in open(ROOT / "eval" / "questions.jsonl", encoding="utf-8") if l.strip()]
    rng = random.Random(a.seed)
    hard = [q for q in qs if q.get("difficulty") == "hard"]
    std = [q for q in qs if q["type"] == "passage" and q.get("difficulty", "standard") == "standard"]
    sample = rng.sample(hard, a.n // 2) + rng.sample(std, a.n - a.n // 2)

    with ui.status("Loading the index..."):
        ix = ask.Index()
    client = anthropic.Anthropic()
    out_path = ROOT / "eval" / f"citecheck_{model}.jsonl"
    spent, results = 0.0, []
    with open(out_path, "w", encoding="utf-8") as out:
        for n, q in enumerate(sample, 1):
            if spent >= a.budget:
                print(f"Budget of ${a.budget:.2f} reached after {n - 1} answers; stopping.")
                break
            hits = ix.passages(q["question"], k=12)
            sources = ask.source_blocks(ix, hits)
            user = sources + f"\n\nQuestion: {q['question']}"
            try:
                msg, cost_a = ask.stream_claude(model, ask.SYSTEM, user, render=lambda it: [*it])
                answer = ask.answer_text(msg)
                verdicts, cost_j = judge(client, sources, answer)
            except (ask.ClaudeError, anthropic.APIError) as e:
                print(f"  {q['id']}: API error, skipped ({e})")
                continue
            spent += cost_a + cost_j
            if verdicts is None:
                print(f"  {q['id']}: judge did not finish, skipped")
                continue
            claims = verdicts["claims"]
            c = {v: sum(x["verdict"] == v for x in claims) for v in ("supported", "partial", "unsupported", "uncited")}
            rec = dict(id=q["id"], difficulty=q.get("difficulty", "standard"), question=q["question"],
                       answer=answer, claims=claims, counts=c, honest=verdicts["honest_about_gaps"],
                       cost=round(cost_a + cost_j, 4))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results.append(rec)
            print(f"  [{n:>2}/{len(sample)}] {q['id']:<5} claims {len(claims):>2}: "
                  f"{c['supported']} supported, {c['partial']} partial, {c['unsupported']} unsupported, "
                  f"{c['uncited']} uncited   (spent ~${spent:.2f})", flush=True)

    if not results:
        sys.exit("No answers audited.")
    tot = {v: sum(r["counts"][v] for r in results) for v in ("supported", "partial", "unsupported", "uncited")}
    claims = sum(tot.values())
    cited = claims - tot["uncited"]
    clean = sum(1 for r in results if r["counts"]["unsupported"] == 0 and r["counts"]["uncited"] == 0)
    print(f"\nAnswers by {ask.MODELS[model]}, audited by {JUDGE}: {len(results)} answers, {claims} claims, "
          f"~${spent:.2f}")
    print(f"  supported   {tot['supported']:>4}  ({tot['supported'] / claims:.0%} of all claims)")
    print(f"  partial     {tot['partial']:>4}  ({tot['partial'] / claims:.0%})")
    print(f"  unsupported {tot['unsupported']:>4}  ({tot['unsupported'] / claims:.0%})")
    print(f"  uncited     {tot['uncited']:>4}  ({tot['uncited'] / claims:.0%})")
    print(f"  citation precision (supported or partial, of cited claims): "
          f"{(tot['supported'] + tot['partial']) / max(cited, 1):.0%}")
    print(f"  answers with no unsupported and no uncited claim: {clean}/{len(results)}")
    print(f"  answers judged honest about gaps: {sum(r['honest'] for r in results)}/{len(results)}")
    print(f"Details: {out_path}")


if __name__ == "__main__":
    main()
