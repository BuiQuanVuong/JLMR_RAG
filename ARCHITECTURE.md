# Architecture

A local retrieval-augmented generation (RAG) system over the 4,138 papers of the Journal of Machine
Learning Research (volumes 1–26, 2000–2025). Search runs on the laptop: SQLite plus a 4 GB NVIDIA GPU.
Only the answer-writing steps call the Claude API.

This file explains how the parts fit together and why they are built the way they are. For usage, see
`README.md`.

## 1. The big picture

```
 INGESTION (run once, then incrementally when new papers appear)

  jmlr.org ──jmlr_dl.py──► JMLR/vNN/*.pdf, abs/*.html, bib/*.bib, catalog.csv
                                 │
                       build_index.py (CPU, ~5 min)
                                 │  PDF text → 300-word passages, page numbers,
                                 │  section labels, bibliography flags
                                 ▼
                          data/jmlr.db (SQLite + FTS5)
                                 │
                          embed.py (GPU, ~30 min)
                                 ▼
               data/chunk_emb.npy, paper_emb.npy (bge-small vectors)


 QUERY (every question)

  question ──► [rewrite.py] Claude Haiku rewrites into paper terminology (cached)
                   │
                   ▼
  Index.passages():  dense search ×2 (original + rewrite) ─┐
                     BM25 search  ×2 (original + rewrite) ─┴─► RRF fusion
                                                                │ top 40
                                                                ▼
                                       cross-encoder reranker (bge-reranker-v2-m3, GPU)
                                       scored against the ORIGINAL question
                                                                │ ≤3 per paper
                                                                ▼
                                                          12 passages
                                                                │
           ask ─────────► Claude (Haiku 5.5 default / Opus 5.5) ─► cited answer [n]
           compare ─────► per-paper evidence records (Haiku) ─► synthesis ─► [Pn.m]
           chat ────────► condense follow-up ─► retrieve ─► answer with history
           search/similar/open/cite/topics ──► local only, no API


 TOPIC MAP (topics.py, once, CPU)

  paper vectors (abstract + mean passage) ─► sweep 3 representations × Ward/k-means × 6–25 areas,
  scored against JMLR's own topic pages ─► 18 areas ─► 70 subtopics ─► Haiku names ─► topic_map.png
```

## 2. Files

| File | Role |
|---|---|
| `jmlr_dl.py` | Polite downloader: 3 parallel requests with pauses. Fetches PDFs, abstract pages and BibTeX, and writes `catalog.csv`. Resumable. |
| `build_index.py` | Text extraction, chunking, section labels, bibliography detection, FTS indexes. `--sections-only` relabels without changing chunk ids. |
| `embed.py` | Embeds papers (title + abstract) and passages. Incremental and resumable. |
| `common.py` | Paths (overridable by environment variables), schema, `.env` loader, model loaders, tuning constants. |
| `ask.py` | CLI entry point. Holds `Index` (all retrieval), `call_claude` (streaming, cost reporting) and the commands. |
| `rewrite.py` | Claude Haiku query rewriting, with a JSON cache. |
| `compare.py` | Two-stage comparison mode. |
| `conversation.py` | Chat memory: turn history, follow-up condensation, history budget. |
| `cite.py` | BibTeX export. |
| `topics.py` | Topic map: granularity sweep, two-level clustering, naming, storage, map image, browse views. |
| `tui.py` | Full-screen Textual app (Chat / Search / Topics / Sessions). Retrieval and Claude calls run in worker threads and stream into Markdown widgets. |
| `ui.py` | Terminal rendering for the line interface: Rich Markdown streaming, spinner, prompt_toolkit input. |
| `tests/test_tui.py` | Headless app test with a fake index and a fake Claude stream. |
| `boolq.py` | Google Scholar-style queries: parser and whole-paper matching, with closest matches when nothing matches strictly. |
| `tests/test_boolq.py` | Parser and matching tests against the real database. |
| `make_eval.py` / `eval.py` | Builds the 140-question evaluation set; scores retrieval. |
| `embed_test.py` | Compares candidate embedding models on a subset index. |

## 3. Data

**Papers on disk** (`D:\Academic_Papers\JMLR`): `vNN/<id>.pdf`, `vNN/abs/<id>.html` (abstract pages),
`vNN/bib/<id>.bib` (volumes 10+), `catalog.csv` (volume, id, title, authors, citation, URLs).
That is 6.3 GB. Paper `v24/22-1047` is missing because its PDF is empty on JMLR's server.

**`data/jmlr.db`** (SQLite, WAL mode, about 860 MB):

| Table | Contents |
|---|---|
| `papers` | `key` (`v20/13-124`), volume, title, authors, citation, year, abstract, PDF path, `n_pages`, `n_chunks`, `status` |
| `chunks` | `id`, `paper_key`, `seq`, `page_start`, `page_end`, `is_ref`, `text`, `section` |
| `chunks_fts` | FTS5 index over passage text (porter stemming), external content = `chunks` |
| `papers_fts` | FTS5 index over title, authors and abstract |
| `topics` | 18 areas (`3`) and 70 subtopics (`3.2`): name, description, size, distinctive terms, centroid |
| `paper_topics` | paper → subtopic: rank 1 = primary, rank 2 = secondary (22% of papers), cosine score |

**Embeddings** (`data/*.npy`): float16 matrices with id arrays alongside: `chunk_emb` (about 250k × 384)
and `paper_emb` (4,138 × 384). They are loaded into RAM as float32 (about 400 MB). Search is a brute-force
matrix product: about 50 ms over all passages, so no vector database is needed.

**Collection size:** 164,319 pages, 274,030 passages. About 27,000 are flagged as bibliography and left
out of search, so about 247,000 are searchable.

## 4. Ingestion

1. **Text:** PyMuPDF reads the PDF text layer (under 0.1 s per paper). Hyphenated line breaks are re-joined
   and whitespace is collapsed. Math comes through only approximately.
2. **Chunking:** the paper's words are flattened, with the page each word came from, and cut into
   300-word windows with a 50-word overlap. Each passage keeps its page range. 300 words is about 400–500
   tokens, which fits the 512-token limit of both the embedder and the reranker.
3. **Section labels:** taken from the PDF bookmarks when there are at least 3 (69% of papers). Otherwise
   they come from numbered heading lines (`4.2 Experimental Results`), accepted only if the numbers
   increase and the line doesn't end in punctuation, which filters out footnotes and cross-references.
   Each heading is located in the word stream, and a passage gets the section containing its midpoint.
   Nested sections become `Parent / Child`. 98.8% of passages are labelled.
4. **Bibliography flags:** a passage is flagged if it is dense in years (over 3.5 per 100 words) or in
   years plus venue patterns (over 8), or if it sits in a "References" section. Flags are only ever added,
   so existing embeddings stay valid.
5. **Embeddings:** `BAAI/bge-small-en-v1.5`, fp16 on the GPU, about 135 passages/s. The embedded text is
   `"<title>: <passage>"`, so each passage carries its paper's topic. The CPU path is about 13× slower.

## 5. Retrieval (`Index.passages`)

| Stage | What happens | Default |
|---|---|---|
| Query rewrite | Claude Haiku turns the question into paper terminology plus keywords. Cached in `eval/rewrite_cache.json`; 20 s timeout; falls back to the original on any failure or without an API key. | `REWRITE = "fuse"` |
| Dense search | The original and the rewrite are each embedded (with the bge query prefix), giving a top-200 list each. | 2 lists |
| Keyword search | The original and the rewrite keywords each run as an FTS5 OR query (stopwords removed), ranked by BM25, top 200 each. | 2 lists |
| Fusion | Reciprocal rank fusion, score = Σ 1/(60 + rank) over the lists. | k = 60 |
| Reranking | The top 40 fused candidates are re-scored by `BAAI/bge-reranker-v2-m3` (fp16, GPU) on `(original question, "<title>. <passage>")`. | `RERANK_POOL = 40` |
| Diversity | At most 3 passages per paper, and the first 12 are kept. | `per_paper=3, k=12` |
| Filters | `--since` / `--until` mask by publication year in both searches. | |

**Boolean queries** (`boolq.py`). When a query uses Scholar syntax (quotes, `OR`, `-x`, `intitle:`,
`author:`), it is parsed into AND-ed clauses of OR-ed terms.
- **Per-term lookup:** each term becomes a set of papers. Words and phrases use FTS5 over non-bibliography
  passages, `intitle:` uses `papers_fts` limited to the title column, and `author:` uses a LIKE match.
  Results are cached per term.
- **Combining:** a paper matches if it satisfies every positive clause and no negated one. Matching is per
  paper, not per passage, as in Scholar.
- **Ranking:** the matches go to `rank_papers(..., allowed=keys)` and are ranked by the query's plain
  words. The filter reaches the dense mask and, through a temporary `allowed_keys` table, the FTS joins.
- **Fallback:** with no strict match, papers are grouped by how many clauses they satisfy, and the tiers are
  ranked best first.
- **`--filter`:** on `ask` and `compare`, the same matching passes `allowed` to `passages()`.

**Paper ranking** (`rank_papers`, used by `search` and `reading`) fuses three lists without the reranker
or the rewrite: title+abstract embedding similarity, BM25 over `papers_fts` (title weighted 5×, abstract
2×), and each paper's best passage hit.

**Why it's built this way** (all measured with `eval.py` on the hard questions; see §8):

| Decision | Evidence |
|---|---|
| Hybrid over either signal alone | Right paper found: hybrid 68%, keyword-only 60%, dense-only 47% |
| Cross-encoder reranking | Right passage found: 50% → 57% (40 candidates) → 62% (100 candidates) |
| bge-reranker-v2-m3 over smaller rerankers | MiniLM dropped to 38% and bge-reranker-base gave mixed results; only v2-m3 improved hard questions |
| Section heading kept out of the reranker input | Measured slightly worse (60% vs 62%) |
| Rewrite fused with the original, not replacing it | At 40 candidates, right passage found: fuse 67%; HyDE 65%; keyword-only rewrite 62%; replacing the original 55% |
| Reranker scores the original question | Your exact intent is the best judge; the rewrite only widens the candidate pool |
| 40 reranker candidates, not 100 | With rewriting on, 40 beat 100 (82% / 67% / 0.53 against 77% / 65% / 0.50) and take 3.5 s per query instead of 6 s. The extra candidates seem to give the reranker more distractors to promote. |

Cumulative effect on hard questions: right paper 68% → 82%, right passage 50% → 67%, MRR 0.45 → 0.53.

## 6. Generation

All Claude calls use the Anthropic Python SDK. Answers stream to the terminal, and each call prints its
token use and approximate cost.

- **`ask`:** the 12 passages are sent as `<source id title authors year pages section>` blocks. The
  system prompt requires a [n] citation for every claim, an explicit note when the sources don't cover
  something, and clearly marked background knowledge.
  - Default: Claude Haiku 5.5 at medium effort, about $0.002 per question.
  - `--opus`: Claude Opus 5.5 at high effort, about $0.03–0.08, with server-side refusal fallback
    (`fallbacks: "default"`).
- **`reading`:** the top 25 papers (title, year, authors, abstract) are ordered by Claude into a learning
  path.
- **`compare`** (`compare.py`), two stages:
  1. Pick the top papers (`-k`, default 6, or `--papers`). For each, retrieve 5 passages *within* that
     paper: 15 dense + 15 BM25 candidates, reranked, put back in reading order. Then build a structured
     evidence record (problem, approach, setting, findings with passage numbers, limitations, relevance)
     with Haiku and JSON-schema output. The records are built in parallel on 6 threads.
  2. Haiku or Opus compares the records: answer, comparison table, agreements and tensions, open
     questions *among these papers*, what to read. Citations are `[Pn.m]`.

  One record per paper keeps each finding attributed to its own paper. A 4-paper comparison costs about
  $0.006 on Haiku.
- **`chat`** (`conversation.py`):
  1. A follow-up is condensed into a standalone search query by Haiku, using the last 3 turns.
  2. Earlier turns are sent as real user/assistant messages: at most 4 turns and about 24k characters,
     newest kept first.
  3. Earlier turns carry only their source titles, labelled "Turn N", so [n] numbers from different turns
     don't get mixed up.
  4. A turn is recorded only after a non-empty, non-refused answer.
  5. Every recorded turn saves the conversation to `data/sessions/<timestamp>.json` (atomic write).
     `/sessions`, `/resume [N]`, `--resume` and the app's Sessions tab reload it, history included.

- **Front ends:** `ask.prepare_chat_turn` / `stream_claude(render=...)` / `record_chat_turn` are the shared
  core. `stream_claude` raises `ClaudeError` instead of exiting. The line chat renders with Rich, and the
  Textual app renders into Markdown widgets from a worker thread, throttled to about 7 updates per second.

- **Topic map** (`topics.py`): paper vectors combine the title+abstract embedding with the mean of the
  paper's passage embeddings. A sweep picks the granularity, scoring each setting on adjusted mutual
  information against 182 papers from 18 of JMLR's own topic pages, plus silhouette and size balance. The
  result is Ward clustering into 18 areas (AMI 0.42), then each area split into subtopics of about 60
  papers (70 in total).
  - Subtopics are named by Haiku from their central papers and distinctive terms (c-TF-IDF). Areas are
    then named from their subtopics' names, so a name covers its whole area.
  - A paper gets a secondary subtopic when it's at least as close to another subtopic's centroid.
  - `--for` tallies the subtopics of the top 40 retrieved passages.
  - `--rising` compares each subtopic's share of papers in 2020–25 with 2005–19.
  - New papers join their nearest existing subtopic (`topics.py assign`), so names stay stable.

API constraints honoured throughout: no sampling parameters, no assistant prefill, no disabled thinking
on the 5.5 models, and the stop reason is checked before content is read.

## 7. Hardware budget

Measured on an Intel Core Ultra 7 155U, 32 GB RAM and an NVIDIA MX570 with 4 GB VRAM:

| Resource | Use |
|---|---|
| RAM per process | About 1.9 GB (index + embedder) plus the reranker |
| GPU memory | About 0.4 GB CUDA context + 0.1 GB embedder + about 1.2 GB reranker. Run one GPU process at a time. |
| Startup | About 28 s (index, embedder, 2 GB reranker). `chat` pays it once per session. |
| Per question | About 3.5 s of retrieval (40 reranked candidates), plus the rewrite (≈ 4 s uncached), plus Claude's answer |
| Disk | Papers 6.3 GB, index about 1 GB, models about 3.5 GB in the Hugging Face cache |

A local LLM was considered and rejected for now. Qwen3 8B doesn't fit in 4 GB of VRAM, and Qwen3 4B
answers clearly worse than Haiku 5.5, which costs about $0.002 per question.

## 8. Evaluation

`eval/questions.jsonl` has 140 questions with known answers, generated by `make_eval.py` from papers
sampled evenly across 2000–2011, 2012–2019 and 2020–2025:

- **60 standard:** written from a passage in the paper's own terms.
- **60 hard:** the same kind of question in plain words, without the paper's terminology. This is the
  realistic case, and the one that tells versions apart.
- **20 topic:** short "find papers about" queries written from abstracts.

`eval.py` is free and local. It reports:
- **paper@12:** the gold paper is among the 12 passages sent to Claude;
- **passage@12:** a passage within one chunk of the gold passage is among them;
- **MRR:** 1 / rank of the first gold-paper hit, averaged;
- topic **R@10** and **R@3**.

Each run is appended to `eval/history.csv`. About a third of hard-question misses return a closely
related paper that arguably answers the question too, so absolute scores understate quality. Use the
scores to compare versions.

## 9. Configuration

| Setting | Where | Default |
|---|---|---|
| `RERANK_MODEL`, `RERANK_POOL` | `common.py` | `BAAI/bge-reranker-v2-m3`, 40 |
| Topic names | `data/topic_names.json` | generated by `topics.py build`; edit freely, then `topics.py map` |
| `REWRITE` | `common.py` | `"fuse"` (`"off"` disables; `--no-rewrite` on the CLI) |
| `CHUNK_WORDS`, `CHUNK_OVERLAP` | `common.py` | 300, 50 (changing these requires a full rebuild and re-embed) |
| `ANTHROPIC_API_KEY` | `.env` (gitignored) | required for `ask`, `reading`, `compare`, `chat` |
| `JMLR_PAPERS_DIR`, `JMLR_RAG_DATA`, `JMLR_RAG_ENV_FILE` | environment | let a git worktree use the shared papers, index and key |

## 10. Known limits and next steps

- **Embedding model, tested and kept (2026-11-14):** on a 420-paper subset, Qwen3-Embedding-0.6B was much
  better than bge-small at meaning-only search: right paper for hard questions 95% vs 83%.
  - **No end-to-end gain:** in the full pipeline (`embed_test.py --pipeline`) both models reached hard
    97% / 80%, and only MRR moved (0.81 vs 0.77). Keyword search, query rewriting and the reranker already
    recover what the small model misses, so a ~10 h full re-embed (Qwen3 runs at 6.8 passages/s here) isn't
    worth it.
  - **bge-base:** no better than bge-small, even on meaning-only search.
- **Earlier diagnosis:** without rewriting, on hard questions the gold passage wasn't among the top 200
  fused candidates about 23% of the time. A better embedder
  would also sharpen the topic map, which is built on the same vectors. `embed_test.py`
  compares bge-base-en-v1.5 and Qwen3-Embedding-0.6B on a subset before committing to a full re-embed
  (about 1.5 h or about 8 h).
- **Topic map:** one area ("Supervised and Semi-Supervised Learning") is a loose mix. JMLR's
  learning-theory papers spread across several areas (only 15% in the most common one), because theory
  papers cluster by what they analyse rather than by being theory.
- **Math:** formulas are approximate in the extracted text. Check them in the PDF.
- **Chat:** memory is per session only. A chat follow-up is condensed, then rewritten again by
  `rewrite.py`; whether that double rewrite helps hasn't been measured yet.
- **Answer quality:** only retrieval is evaluated. A citation-accuracy check (does each cited passage
  support its claim?) would cover the generation side.
- **Coverage:** JMLR only. Much recent deep-learning work appears at NeurIPS, ICML and ICLR instead.
