# Plan: extending the collection with PMLR (ICML, AISTATS, COLT, ...)

Status: **plan only, not started** (2026-11-14). Goal: answer questions JMLR can't. Much modern
deep-learning work, for example multimodal or vision-language models and federated learning at scale, appears
at conferences rather than in JMLR. PMLR (Proceedings of Machine Learning Research) publishes those
proceedings open access.

## What PMLR is (measured 2026-11-14)

- `https://proceedings.mlr.press/` lists **331 volumes**: ICML, AISTATS, COLT, UAI, CoRL, workshops.
- Major recent volumes: ICML 2023 = v202, **ICML 2024 = v235**, ICML 2025 = v267; AISTATS 2023/24/25 =
  v206 / v238 / v258; COLT 2023/24/25 = v195 / v247 / v291.
- **ICML 2024 alone: 2,610 papers** (63% of the whole JMLR archive). The volume page is 2.5 MB of HTML with
  one `<div class="paper">` per paper: title, authors, "PMLR 235:1-30", abstract page link, PDF link.
- PDFs are served from **GitHub** (`raw.githubusercontent.com/mlresearch/v235/main/assets/<id>/<id>.pdf`),
  not a small server. Still polite: 1–3 requests at a time with pauses, plus backoff on HTTP 429.

## Size estimates per ICML year (to verify on the pilot)

| Item | Estimate | Basis |
|---|---|---|
| Papers | ~2,600 | measured for 2024 |
| PDFs on disk | ~5–8 GB | 2–3 MB per paper with appendices |
| Download time | ~45–90 min | 1 PDF/s, polite |
| Passages | ~130–150k (+55% of today's 250k) | ~25–30 pages per paper |
| Text extraction | ~5 min | today: 4,138 papers in ~5 min |
| Embedding (bge-small, GPU) | ~20 min | 135 passages/s |
| Extra RAM for the index | ~+250 MB per year | 384-dim float32 |

## What has to change in the code

1. **A source/venue field.** Add `source` (`jmlr`, `pmlr`) and `venue` (`ICML 2024`) columns to `papers`.
   Keys become `pmlr/v235/abe24a` (JMLR keys stay as they are). Filters: `--venue ICML` and `--source jmlr`,
   plus a `venue:` term in the Scholar-style syntax.
2. **A PMLR downloader** (`pmlr_dl.py`): parse a volume page, fetch the PDFs and per-paper BibTeX (the
   abstract pages carry it), write a catalog, resume and back off like `jmlr_dl.py`. `update` gains
   `--pmlr v235,v202`.
3. **Index build:** `build_index.py` reads both catalogs; the rest of the pipeline is source-agnostic.
4. **Keyword-search speed:** today a broad rewrite query takes about 1.3 s in FTS ranking over 250k
   passages, and it grows with the collection. Cap the rewrite keywords at about 20 terms (measure with
   `eval.py`), and if needed restrict BM25 to the top-N papers from the paper-level ranking.
5. **Topic map:** re-cluster with `topics.py build` over everything. The sweep will choose a new
   granularity, probably 25–35 areas. Venue mix per topic becomes a useful view ("this subtopic lives
   mostly at ICML").
6. **Evaluation:** extend `eval/questions.jsonl` with about 60 PMLR questions (`make_eval.py` from PMLR
   papers, about $0.20). Re-check that the JMLR scores don't drop when the collection grows, since more
   distractors make search harder.
7. **Citations:** answers and `cite` show the venue ("ICML 2024"), and `cite` uses PMLR's BibTeX.

## Steps

| Step | What | Time | Cost |
|---|---|---|---|
| 1 | Schema + downloader + filters, tested on a small volume (e.g. a workshop, ~50 papers) | ~2 h of work | $0 |
| 2 | **Pilot: ICML 2024** (v235): download, index, embed | ~1.5 h unattended | $0 |
| 3 | Measure: eval (old + new questions), latency, RAM, disk; fix keyword speed if needed | ~1 h | ~$0.20 |
| 4 | Rebuild the topic map over both sources | ~10 min | ~$0.05 |
| 5 | **Decision:** keep and add ICML 2023/2025, AISTATS, COLT, one volume at a time | ~1.5 h each | $0 |

## Decision points and risks

- **Search quality with more distractors:** checked in step 3 against the JMLR scores (hard: 82% / 67% /
  0.53). If JMLR questions get noticeably worse, add venue weighting or keep sources as separate filters.
- **Latency:** keyword ranking cost grows with the collection; step 3 measures it, item 4 above fixes it.
- **Disk:** ~5–8 GB per ICML year; check free space before starting.
- **Licence and use:** PMLR papers are open access for reading. This is a personal research archive, as with
  JMLR; don't redistribute the PDFs.
- **GitHub throttling:** back off on 429s, never more than 3 parallel requests, and resume on rerun.
