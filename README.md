# JMLR RAG

A local search engine plus Claude-powered Q&A over the 4,138 JMLR papers in `..\JMLR`.

`search` and `similar` run locally and cost nothing. `ask`, `reading`, `compare` and `chat` call the Claude
API; for `ask` and `compare` that includes a query rewrite before retrieval (about $0.0004 and 4 s per new
question, cached; `--no-rewrite` turns it off, and without an API key retrieval silently uses your words).

## Setup (one time)

All commands run from this folder, using the project's own Python:

```
cd D:\Academic_Papers\jmlr_rag
.venv\Scripts\activate
```

1. `python build_index.py` extracts the text of every PDF into `data\jmlr.db`. It takes about 5 minutes and is already done.
2. `python embed.py` computes semantic embeddings on the GPU. You can stop and resume it.
3. For the Claude features, you need an Anthropic API key from https://platform.claude.com. Open `.env` in
   this folder and replace `paste-your-key-here` with your key:
   `ANTHROPIC_API_KEY=sk-ant-...`
   The `.env` file stays on your machine. Don't share it or commit it anywhere. An `ANTHROPIC_API_KEY` that's
   already set in your system environment takes priority over the file.

## Usage

**Full-screen app:** `jmlr-rag` with no arguments, in a terminal, opens a Textual app (`tui.py`) with four
tabs:
- **F1 Chat:** conversation with memory and streaming Markdown answers. A sources pane lists the cited
  passages, and Enter on one opens its PDF. `/compare <question>` and `/reading <topic>` work here too.
- **F2 Search:** ranked papers. The highlighted paper's abstract and best-matching passage are shown, and
  Enter opens the PDF.
- **F3 Topics:** the topic map as a tree (areas → subtopics → papers), with trends, top authors and
  abstracts.
- **F4 Sessions:** saved conversations. Enter resumes one.

Ctrl+N starts a new conversation, Ctrl+O switches between Haiku and Opus, and Ctrl+Q quits.
`jmlr-rag chat` keeps the line-based chat. `python tests\test_tui.py` checks the app headlessly, with no GPU
and no API calls.

**Saved sessions:** every conversation is saved after each answer in `data\sessions\`. In the line chat,
`/sessions` lists them, `/resume [N]` continues one, and `jmlr-rag chat --resume` reopens the most recent.
`/new` starts a fresh conversation without deleting the old one.

**One-word start:** `jmlr-rag search "..."`, `jmlr-rag topics` and so on work like `python ask.py ...`. It's installed in the project environment with
`.venv\Scripts\python -m pip install --no-deps -e .`. To use it from any terminal, copy `bin\jmlr-rag.cmd`
into a folder on your PATH, such as `%LOCALAPPDATA%\Microsoft\WindowsApps`.

**In a terminal**, answers are rendered as formatted Markdown (headings, tables, bold) while they stream, and
a spinner shows while models load. In chat:
- Tab completes commands and paper keys.
- Up/Down browse your earlier questions, which are saved in `data\chat_history.txt`.
- The bottom bar shows the model, the memory turns and this session's API cost.

When output is piped or redirected, everything falls back to plain text.

```
python ask.py search "kernel two-sample tests"            # rank papers (free, local)
python ask.py ask "How is the MMD estimator's variance controlled?"      # cited answer, Claude Haiku 5.5
python ask.py ask "Compare PAC-Bayes and Rademacher bounds for deep nets" --opus   # Claude Opus 5.5
python ask.py reading "causal discovery" --since 2015     # ordered reading list
python ask.py compare "How do these tests choose the kernel bandwidth?" -k 4     # side-by-side comparison
python ask.py compare "..." --papers v13/gretton12a v24/21-1289                  # compare chosen papers
python ask.py similar v20/13-124                          # papers like this one (free)
python ask.py open v20/13-124                             # open the PDF
python ask.py cite v20/13-124 v13/gretton12a --out refs.bib    # BibTeX (UTF-8; avoid PowerShell '>')
python ask.py topics                                      # browse 18 research areas (free, local)
python ask.py topics 14.1                                 # a subtopic: trend, top authors, central papers
python ask.py topics --rising                             # what grew / faded, 2020-25 vs 2005-19
python ask.py topics --for "your question"                # which subtopics a question falls into
python ask.py chat                                        # interactive, remembers the conversation (see below)
```

**Google Scholar-style queries (`boolq.py`):** `search` (and the app's Search tab) switches to Boolean
matching when a query uses the syntax:
- `"exact phrase"`
- `a OR b`
- `-exclude`
- `intitle:word`
- `author:Name`
- every other word is required

For example: `search 'federated "differential privacy" -bandit'` or `search 'intitle:"two-sample" kernel author:Gretton'`.
- **Whole papers:** as in Scholar, terms may appear anywhere in a paper (bibliographies are ignored), and
  matching papers are ranked by relevance.
- **No dead ends:** if no paper matches every term, it prints how many papers contain each term and lists
  the papers matching the most terms, with what each one is missing.
- **Filtering questions:** `ask` and `compare` take `--filter '<query>'` to answer only from matching papers.
- **Literal matching:** matching is literal, like Scholar's. A common word such as `distributed` also
  matches "distributed as Gaussian".

Options: `--since YEAR`, `--until YEAR`, `-k N` for the number of passages or papers, and `--no-rewrite` to
search with your own words only (no API call before retrieval).

**Cost per question (approx.):** Haiku 5.5 about $0.001, Opus 5.5 about $0.03–0.08. Each answer prints its actual token use and cost.

## How it works

- **Passages:** each paper is split into overlapping 300-word passages that keep their page numbers. Bibliography passages are detected and excluded.
- **Hybrid retrieval:** BM25 keyword search (SQLite FTS5) and semantic search (`BAAI/bge-small-en-v1.5` embeddings) are fused with reciprocal rank fusion.
- **Query rewriting (`rewrite.py`):** before retrieval, Claude Haiku 5.5 restates the question in the
  terminology a paper would use (plus keywords), e.g. "a way to pick how wide the similarity function should
  be" becomes "kernel bandwidth selection, Gaussian/RBF kernel, median heuristic". Four ranked lists are fused:
  dense and BM25 for the original question, dense for the rewrite, BM25 for rewrite + keywords. The reranker
  still scores against your original question. `ask` and `compare` (paper selection) use it; `search` and
  `reading` don't (topic search is already at 100%). Rewrites are cached in `eval\rewrite_cache.json`; if
  the API is unreachable or there is no key, the original question is used.
- **Reranking:** the top 40 fused candidates are re-scored by a cross-encoder (`BAAI/bge-reranker-v2-m3`, on the GPU). It reads the title and the passage together. At most 3 passages come from any one paper, so answers draw on several papers.
- **Section labels:** every passage knows its section (for example "Numerical Experiments / Empirical Size"). Labels come from the PDF's bookmarks or from detected numbered headings, and they're shown in answers and source lists. Re-run them with `python build_index.py --sections-only`.
- **Answers:** Claude answers only from the retrieved passages and cites them as [n]. Each answer ends with the source list (paper, year, page, key) so you can open the PDF.

## Topic map (`topics.py`)

`python topics.py build` clusters every paper into **18 research areas** and **70 subtopics** (about 60
papers each), names them with Claude Haiku (about $0.02, once), and draws `topic_map.png`. Browsing is free and
local. It takes about 1 minute, plus 20 seconds for the map.

- **What is clustered:** each paper's title+abstract embedding combined with the average of its passage
  embeddings. This reflects the abstract and the full text.
- **Choosing the granularity:** a sweep over 3 representations × Ward / k-means × 6–25 areas. Each
  setting is scored on agreement (adjusted mutual information) with 182 papers on 18 of JMLR's own topic and
  special-issue pages, silhouette, and size balance. The best setting with 8–25 areas, no area over 25% and
  none under 15 papers won: combined / Ward / 18 areas (AMI 0.42, against 0.25–0.40 for most others).
  Subtopic size was chosen the same way.
- **Names:** subtopics are named first from their central papers and distinctive terms. Each area is then
  named from all its subtopics, so the name covers everything inside it. Edit `data\topic_names.json` to
  rename anything, then run `python topics.py map` to redraw the image.
- **Secondary topics:** 912 papers (22%) are at least as close to another subtopic as to their own, so they
  are listed there too.
- **New papers:** `python topics.py assign` puts them in their nearest existing subtopic, so names stay
  stable. Re-run `build` to re-cluster from scratch.

`search` results show each paper's subtopic, and in `chat` the same views are available as
`/topics [a | a.b | rising | for <question>]`.

## Measuring retrieval quality

`eval\questions.jsonl` holds 140 test questions with known answers, generated by `make_eval.py`:

- **60 standard:** use the paper's own terms.
- **60 hard:** plain words, none of the paper's terminology. This is the realistic case.
- **20 topic:** short "find papers about" queries.

`python eval.py --label <name>` scores the current system in about 8 minutes (12 with `--pool 100`). The
query rewrites come from `eval\rewrite_cache.json`; fill it once with `python rewrite.py --eval` (about $0.05). Results go to
`eval\results_<name>.csv`, and each run is appended to `eval\history.csv`. Add `--modes` to also score
keyword-only and meaning-only search. Run it before and after every retrieval change.

| Setup (2026-11-13) | Hard: paper / passage / MRR | Standard passage / MRR |
| --- | --- | --- |
| Hybrid, no reranker | 68% / 50% / 0.45 | 85% / 0.94 |
| + bge-reranker-v2-m3, 40 candidates | 72% / 57% / 0.49 | 97% / 1.00 |
| + 100 candidates (`--no-rewrite`) | 73% / 62% / 0.49 | 97% / 1.00 |
| + Claude query rewrite "fuse", 40 candidates (**current default**, 3.5 s/query) | 82% / 67% / 0.53 | 97% / 1.00 |
| + Claude query rewrite "fuse", 100 candidates (6 s/query) | 77% / 65% / 0.50 | 97% / 1.00 |

With rewriting on, 40 candidates beat 100: the extra 60 seem to give the reranker more distractors to
promote. The 40-candidate result was reproduced exactly on the merged code (2026-11-14).

Query-rewrite variants, all with 40 candidates (`python eval.py --pool 40 --rewrite MODE`; hard paper / passage /
MRR): `off` 72% / 57% / 0.49; `bm25` (rewrite for BM25 only) 80% / 62% / 0.54; `both` (rewrite replaces the
question for BM25 and dense) 72% / 55% / 0.51, and standard passage drops to 90%; `fuse` 82% / 67% / 0.53;
`hyde` (original + hypothetical passage for dense) 77% / 65% / 0.51. Topic stays at 100% in all of them.

Also tested: MiniLM and bge-reranker-base rerankers (both worse on hard questions), and section headings in
the reranker input (slightly worse). Topic search stays at 100%. Standard and topic questions are near
100%, so they only catch regressions.

## Comparison mode (`compare.py`)

`compare` runs in two stages, so findings stay attached to the right paper:

1. **Evidence records:** it picks the top papers (6 by default, `-k N`, or `--papers KEY ...`) and retrieves
   the best 5 passages *within* each paper with hybrid search and the reranker. Claude Haiku 5.5 then turns
   each paper into a structured record (problem, approach, setting, findings with citations, limitations,
   relevance). These calls run in parallel.
2. **Synthesis:** Claude (Haiku 5.5, or Opus 5.5 with `--opus`) compares the records. It writes a short
   answer, a comparison table, agreements and tensions, open questions among these papers, and what to read.

Citations look like `[P2.3]` (paper 2, passage 3), and each one is listed with its page and section. A
4-paper comparison costs about $0.006 with Haiku, or about $0.06–0.10 with `--opus` for the synthesis.

## Chat with memory (`conversation.py`)

`python ask.py chat` remembers the conversation, so follow-ups like "Which of those papers handles
functional data?" or "How does source 2 from your first answer pick its kernel?" work:

1. **Query condensing:** from the second question on, Claude Haiku 5.5 rewrites the follow-up into a
   standalone search query, using the last 3 turns (questions, answers and source titles). The query is
   printed as `Searching for: ...` and costs about $0.0003. If that call fails, the question is searched
   as typed.
2. **Answering with context:** the last 4 turns (up to about 6k tokens) go to Claude as real
   user/assistant messages. Older turns carry only their source *titles*, not the passages, and each turn
   is labelled "Turn N". Citations [n] always refer to the current turn's source list, and Claude names a
   paper from an earlier turn instead of reusing an old number.

`/new` (or `/reset`) starts a fresh conversation. `/opus` and `/haiku` switch models and keep the memory.
The prompt shows how many turns are remembered. `/search`, `/reading`, `/compare`, `/similar` and `/open`
are one-shot: they neither use nor add to the memory. `python ask.py ask` has no memory. A follow-up costs
a little more than a fresh question, because the history is sent again (Haiku: about $0.002–0.003).

## Measuring answer quality (`citecheck.py`)

`python citecheck.py` has Haiku answer 30 eval questions, then Claude Opus 5.5 audits every claim against
the passages it cites, labelling each supported, partial, unsupported or uncited. It costs about $2.50 and
stops at the `--budget` limit. `--opus` audits Opus answers instead.

| Haiku 5.5 answers, 2026-11-14 | Result |
| --- | --- |
| Claims audited | 438 in 30 answers |
| Supported / partial / unsupported / uncited | 85% / 6% / 1% / 8% |
| Citation precision (cited claims at least partly supported) | 99% |

The few errors were a definition cited to the wrong source, proof details, and over-generalised
summaries. Uncited claims are mostly Haiku's own inferences, occasionally a derived formula, so check
numbers, formulas and attributions on the cited page. Details are in `eval\citecheck_haiku.jsonl`.

## Adding new papers later

`jmlr-rag update` does it in one go:
1. downloads what JMLR has published since the last run, at the same polite pace;
2. extracts text and refreshes the keyword index;
3. embeds the new passages;
4. places the new papers in the topic map.

It then lists the new papers, and appends a summary to `data\update_log.txt`. With nothing new, a run takes about 3 minutes. `jmlr-rag update --check` only compares JMLR's listing with your index, without downloading anything. Close the full-screen app first, so it doesn't compete with the embedding step for the GPU.

## Known gaps

- `v24/22-1047`: the PDF on JMLR's server is empty (0 bytes), so it isn't indexed.
- Text comes from the PDF text layer, so math formulas come through only roughly. Claude handles this reasonably, but exact equations should be checked in the PDF.
