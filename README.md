# JMLR RAG

A local research assistant for the **Journal of Machine Learning Research** (JMLR, about 4,100 open-access
papers since 2000). It downloads the archive to your computer and builds a search index over the full text
of every paper. You can then:

- **search** by meaning, by keywords, or with Google Scholar-style queries (`"exact phrase" a OR b -exclude author:Name`);
- **ask questions** and get answers written by Claude that cite the exact paper, page and section, with the
  PDF opening at the cited page;
- **compare** several papers side by side, or get an **ordered reading list** for a topic;
- **browse a topic map** of 18 research areas and 70 subtopics, with trends, top authors, and a
  **"recent advances" dashboard** per subtopic;
- keep **conversations** with memory and resume them later, and export **BibTeX**.

Use it from a **desktop app**, a **full-screen terminal app**, or the **command line**. Searching, browsing and
BibTeX are local and free; only the Claude features use the Anthropic API, typically about $0.002 per
question with Claude Haiku.

## Requirements

| | |
|---|---|
| Python | 3.11 or newer |
| Disk | about 8 GB (6-7 GB of PDFs + about 1 GB index) plus about 3.5 GB of downloaded models |
| Memory | 8 GB RAM minimum, 16 GB+ recommended |
| GPU | optional. An NVIDIA GPU with 4 GB+ makes indexing about 13× faster (~30 min instead of several hours) and answers faster; everything also runs on the CPU. |
| Anthropic API key | optional; needed only for Claude features (answers, chat, compare, reading lists, topic names, the recent-advances summaries). Get one at https://platform.claude.com. |
| OS | Windows, macOS or Linux. The desktop window uses the system web view (Edge WebView2 on Windows; see the [pywebview docs](https://pywebview.flowrl.com/) for macOS/Linux), and `--browser` works everywhere. |

## Installation

```bash
git clone https://github.com/BuiQuanVuong/JLMR_RAG.git
cd JLMR_RAG
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 1. PyTorch first, matching your hardware (pick the command at https://pytorch.org/get-started/locally/), e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu128     # NVIDIA GPU (CUDA 12.8)
# pip install torch                                                       # CPU only

# 2. Everything else, plus the `jmlr-rag` command
pip install -e .            # or: pip install -e ".[dev]"  to include the test tools

# 3. API key (optional, for the Claude features)
cp .env.example .env        # Windows: copy .env.example .env
# then edit .env and replace paste-your-key-here with your key
```

`.env` is ignored by git; keep it private. A key already set in your environment takes priority.

## Building the collection (one time)

Run these from the project folder with the virtual environment active. Every step can be stopped and
re-run; finished work is skipped.

| Step | Command | Time | Cost |
|---|---|---|---|
| 1. Download the archive | `python jmlr_dl.py` | 2-3 h, 6-7 GB | free |
| 2. Extract text, passages, sections | `python build_index.py` | ~5 min | free |
| 3. Compute embeddings | `python embed.py` | ~30 min on a GPU, several hours on CPU | free |
| 4. Build the topic map | `python topics.py build` | ~2 min | ~$0.02 (topic names; `--no-names` for free) |
| 5. Recent-advances summaries *(optional)* | `python advances.py` | ~2 min | ~$0.06 |

- **Polite downloading:** the downloader makes at most 3 requests at a time with pauses. JMLR is run by
  volunteers; please don't raise that.
- **Locations:** papers go to `./papers` and the index to `./data`. Both can be changed in `.env` (see
  Configuration).

Afterwards, `jmlr-rag update` keeps everything current: it downloads, indexes, embeds and places new papers
in the topic map in one go (a run with nothing new takes ~3 minutes; `--check` only reports what's missing).

## Using it

### Desktop app

```bash
jmlr-rag gui              # its own window
jmlr-rag gui --browser    # the same app in your web browser (http://127.0.0.1:8765)
```

| Page | What it does |
|---|---|
| **Chat** | Conversation with memory. Answers stream in as formatted text, and each [n] is a clickable citation that opens the PDF inside the app at the cited page. A sources list sits under each answer. `/compare <question>` and `/reading <topic>` work in the input box. |
| **Search** | Ranked papers with topic, highlighted best passage, "Open PDF at p. N" and BibTeX. Scholar syntax with per-term counts. Year and research-area filters. |
| **Paper page** | Abstract, topic, clickable section outline (opens the PDF at that section), similar papers, BibTeX. |
| **Topic map** | All papers on an interactive map by research area: zoom, hover, click. Selecting a topic shows papers per year, top authors and central papers. Year filter and rising/fading view. |
| **Recent advances** | Momentum charts and a card per subtopic: where it's heading, 3-5 recent advances linked to the papers they come from, top authors. |
| **Sessions** | Saved conversations; Resume reopens one. |

The search models load in the background when the app starts, in about 20-30 s. Pages that don't need
them (topic map, recent advances, sessions) work immediately.

### Terminal app

`jmlr-rag` with no arguments opens a full-screen terminal app with four tabs: Chat, Search, Topics and Sessions
(F1-F4). Ctrl+N starts a new chat, Ctrl+O switches between Claude Haiku and Opus, and Ctrl+Q quits.
`jmlr-rag chat` gives a simpler line-by-line chat with Tab completion and saved input history.

### Command line

```bash
jmlr-rag search "kernel two-sample tests"                          # rank papers (free)
jmlr-rag search 'federated "differential privacy" -bandit'         # Scholar-style query (free)
jmlr-rag ask "How is the bandwidth of an MMD test chosen?"         # cited answer (Haiku)
jmlr-rag ask "Compare PAC-Bayes and Rademacher bounds" --opus      # harder questions (Opus)
jmlr-rag compare "How do these tests choose the kernel bandwidth?" -k 4
jmlr-rag reading "causal discovery" --since 2015                   # ordered reading list
jmlr-rag similar v13/gretton12a                                    # papers like this one (free)
jmlr-rag open v13/gretton12a                                       # open the PDF
jmlr-rag cite v13/gretton12a v24/21-1289 --out refs.bib            # BibTeX (free)
jmlr-rag topics | topics 14 | topics 14.1 | topics --rising | topics --for "your question"
jmlr-rag chat --resume                                             # continue the last conversation
jmlr-rag update                                                    # fetch and index new papers
```

Paper keys look like `v13/gretton12a` (volume / JMLR id); every result shows them.

| Option | Effect |
|---|---|
| `--since YEAR`, `--until YEAR` | restrict to a range of years |
| `-k N` | number of passages (`ask`) or papers (`search`, `reading`, `compare`) |
| `--opus` | answer with Claude Opus instead of Claude Haiku |
| `--filter '<query>'` | `ask`/`compare` only from papers matching a Scholar-style query |
| `--papers KEY ...` | `compare` these specific papers |
| `--no-rewrite` | search with your exact words, with no rewrite call |
| `--out FILE` | `cite` writes a UTF-8 `.bib` file (better than shell redirection on Windows) |

To run `jmlr-rag` without activating the virtual environment, put the `bin/` folder on your PATH. It holds
launchers for Windows (`jmlr-rag.cmd`) and macOS/Linux (`jmlr-rag`).

### Scholar-style queries

| Write | Means |
|---|---|
| `"exact phrase"` | these words in this order (hyphens count as spaces) |
| plain words | all required (plural forms match too) |
| `a OR b` | either term; binds to its neighbours |
| `-word` | exclude papers containing it |
| `intitle:word`, `author:Name` | in the title / an author's name |

Terms may appear anywhere in a paper, bibliographies excluded, as in Scholar, and matches are ranked by
relevance. If nothing matches every term, you see how many papers contain each term and the closest
matches, each with what it's missing. Matching is literal, so a common word like `distributed` also
matches "distributed as Gaussian".

### Costs

| Action | Approximate cost |
|---|---|
| Question with Claude Haiku (default) | $0.002 |
| Question with Claude Opus (`--opus`) | $0.03-0.08 |
| `compare` of 4 papers (Haiku) | $0.006 |
| Follow-up in chat | slightly more than a fresh question (the history is resent) |
| Search, similar, topics, BibTeX, browsing | free |

Each answer prints its exact token use and cost; the app shows the running total for the session.

## How it works

1. **Passages:** each paper is split into overlapping 300-word passages that keep their page numbers and
   section titles. Bibliography passages are excluded.
2. **Query rewrite:** Claude Haiku restates your question in the terminology papers use, for example "how
   wide the similarity function should be" becomes "kernel bandwidth selection". The result is cached;
   without an API key, your own words are used.
3. **Hybrid search:** keyword search (SQLite FTS5 / BM25) and meaning search (`BAAI/bge-small-en-v1.5`
   embeddings), each run on both the original and the rewritten question, fused with reciprocal rank fusion.
4. **Reranking:** a cross-encoder (`BAAI/bge-reranker-v2-m3`) re-scores the top 40 candidates against your
   original question. At most 3 passages per paper are kept, 12 in total.
5. **Answer:** Claude answers only from those passages, citing them as [n]. It says what the sources
   don't cover, and marks background knowledge as such.
6. **Comparison mode** builds a structured evidence record per paper first and compares the records,
   which keeps findings attached to the right paper.
7. **Chat memory:** a follow-up ("which of those handles noise?") is condensed into a standalone search.
   The last few turns go to Claude as context, and conversations are saved to `data/sessions/`.
8. **Topic map:** papers are clustered by Ward clustering of their abstract and full-text embeddings into
   18 areas and 70 subtopics. The granularity was chosen by a sweep scored against JMLR's own topic pages;
   Claude names the clusters.

[ARCHITECTURE.md](ARCHITECTURE.md) has the full design, with the measurements behind each decision.

## Quality

Every design choice above was measured with an evaluation set included in the repo.
`eval/questions.jsonl` has 140 questions with known answers:
- **60 "standard":** use a paper's own terms.
- **60 "hard":** plain words, none of the paper's terminology.
- **20 topic queries.**

`python eval.py` scores retrieval for free in a few minutes.

| Retrieval setup | Hard: right paper / right passage / MRR | Standard: right passage |
| --- | --- | --- |
| Hybrid search only | 68% / 50% / 0.45 | 85% |
| + reranker (40 candidates) | 72% / 57% / 0.49 | 97% |
| + Claude query rewrite (**default**) | **82% / 67% / 0.53** | 97% |

"Right paper" means it is among the 12 passages given to Claude. Also tested and rejected: smaller
rerankers, 100 rerank candidates, and a larger embedding model (Qwen3-Embedding-0.6B). The larger model
was much better at meaning-only search but gave no end-to-end gain.

**Answer faithfulness:** `python citecheck.py` has Claude Opus audit every claim in 30 Haiku answers
against the passages they cite, for about $2.50:

| Claims | 438 |
| --- | --- |
| Supported / partial / unsupported / uncited | 85% / 6% / 1% / 8% |
| Citation precision (cited claims at least partly supported) | 99% |

Uncited claims are mostly the model's own summaries, occasionally a derived formula. As with any
generated text, check numbers, formulas and attributions on the cited page.

## Configuration

Settings in `.env` (or the environment):

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Claude features |
| `JMLR_PAPERS_DIR` | `./papers` | downloaded PDFs and catalog |
| `JMLR_RAG_DATA` | `./data` | index, embeddings, topic map, sessions |

Retrieval settings live in `common.py`: `RERANK_MODEL`, `RERANK_POOL` (40), `REWRITE` (`"fuse"`; `"off"`
disables the rewrite), and `CHUNK_WORDS` / `CHUNK_OVERLAP` (changing these requires rebuilding the index).

## Tests

```bash
pip install -e ".[dev]"
python -m pytest gui/ -p nicegui.testing.user_plugin -o asyncio_mode=auto -o main_file=   # desktop app
python tests/test_tui.py        # terminal app
python tests/test_boolq.py      # Scholar-style queries
```

None of these use the GPU or the API. They use a stand-in index and a fake Claude stream. The GUI and
query tests read the local database, so build the collection first.

## Limitations

- **Formulas** are taken from the PDF text layer and come through only approximately; check equations in
  the PDF.
- **Coverage** is JMLR only. Much recent deep-learning work appears at conferences (NeurIPS, ICML, ICLR);
  [docs/PLAN_PMLR.md](docs/PLAN_PMLR.md) sketches adding PMLR proceedings.
- **Scholar matching is literal,** so common words also match unrelated uses.
- **Startup:** loading the search models takes 20-30 seconds per start (mostly PyTorch and the reranker);
  keep the app open between questions.
- **One empty PDF:** one JMLR paper (`v24/22-1047`) has an empty PDF on the JMLR server and can't be
  indexed.

## Notes

- The papers are © their authors and published open access by JMLR. This tool keeps a personal local copy
  for research; don't redistribute the PDFs.
- Answers are generated by Claude and can contain mistakes. The citations are there so you can check them.
