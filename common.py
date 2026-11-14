"""Shared paths, database schema and helpers for the JMLR RAG system."""
import os
import sqlite3
import sys
import warnings
from pathlib import Path

# keep the terminal quiet: no HF progress bars / symlink warnings / transformers chatter
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
warnings.filterwarnings("ignore")
import logging  # noqa: E402
for _name in ("torch.distributed.elastic.multiprocessing.redirects", "sentence_transformers",
              "huggingface_hub", "transformers"):
    logging.getLogger(_name).setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent


def _default_papers_dir():
    """Where the downloaded papers live: ./papers inside the project (the default for new setups), or a
    sibling ../JMLR folder if one already exists (older setups)."""
    inside, sibling = ROOT / "papers", ROOT.parent / "JMLR"
    if not inside.exists() and (sibling / "catalog.csv").exists():
        return sibling
    return inside


# All three locations can be overridden with environment variables (or lines in .env):
#   JMLR_PAPERS_DIR   downloaded PDFs + catalog       (default: ./papers)
#   JMLR_RAG_DATA     index, embeddings, sessions     (default: ./data)
#   JMLR_RAG_ENV_FILE the file holding the API key    (default: ./.env)
JMLR_DIR = Path(os.environ.get("JMLR_PAPERS_DIR", _default_papers_dir()))
DATA_DIR = Path(os.environ.get("JMLR_RAG_DATA", ROOT / "data"))
DB_PATH = DATA_DIR / "jmlr.db"
CHUNK_EMB = DATA_DIR / "chunk_emb.npy"    # float16 [n_chunks, dim], row i <-> CHUNK_IDS[i]
CHUNK_IDS = DATA_DIR / "chunk_ids.npy"
PAPER_EMB = DATA_DIR / "paper_emb.npy"
PAPER_KEYS = DATA_DIR / "paper_keys.npy"

EMB_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # best of 3 tested with eval.py; None disables reranking
RERANK_POOL = 40                           # with query rewriting, 40 beat 100 (hard 67% vs 65%) and is ~2.5 s faster
REWRITE = "fuse"   # Claude query rewrite in ask.Index.passages (see rewrite.py); "off" = your words only

ENV_FILE = Path(os.environ.get("JMLR_RAG_ENV_FILE", ROOT / ".env"))
PLACEHOLDER = "paste-your-key-here"

_INFO = None


def collection_info():
    """(number of indexed papers, first year, last year) of the local collection, for labels and time
    windows. Read once from the index; sensible fallbacks before anything has been indexed."""
    global _INFO
    if _INFO is None:
        _INFO = (0, 2000, 2025)
        if DB_PATH.exists():
            try:
                con = sqlite3.connect(DB_PATH)
                n, lo, hi = con.execute("SELECT COUNT(*), MIN(year), MAX(year) FROM papers "
                                        "WHERE status='done'").fetchone()
                con.close()
                if n:
                    _INFO = (n, lo or 2000, hi or 2025)
            except sqlite3.Error:
                pass
    return _INFO


def collection_label():
    """e.g. '4,138 JMLR papers, 2000-2025' (or a hint when nothing is indexed yet)."""
    n, lo, hi = collection_info()
    return f"{n:,} JMLR papers, {lo}-{hi}" if n else "no papers indexed yet"


def open_path(path):
    """Open a file with the system's default application (PDF reader), on Windows, macOS or Linux."""
    path = str(path)
    if sys.platform.startswith("win"):
        os.startfile(path)
    else:
        import subprocess
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_env():
    """Load KEY=value lines from jmlr_rag/.env into the environment.
    Variables already set in the environment win; the placeholder value is ignored."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if value and value != PLACEHOLDER:
            os.environ.setdefault(key, value)


load_env()

CHUNK_WORDS = 300     # words per chunk (~400-500 tokens; the embedder truncates at 512)
CHUNK_OVERLAP = 50

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers(
    key TEXT PRIMARY KEY,           -- e.g. 'v20/13-124'
    volume INTEGER, pid TEXT, title TEXT, authors TEXT, citation TEXT, year INTEGER,
    abstract TEXT, pdf_path TEXT, pdf_url TEXT, abs_url TEXT,
    n_pages INTEGER, n_chunks INTEGER, status TEXT
);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY,
    paper_key TEXT NOT NULL, seq INTEGER, page_start INTEGER, page_end INTEGER,
    is_ref INTEGER DEFAULT 0,       -- 1 = looks like a bibliography chunk (excluded from search)
    text TEXT,
    section TEXT                    -- e.g. 'Experiments / 4.2 Results on noisy data' ('' if unknown)
);
CREATE INDEX IF NOT EXISTS chunks_paper ON chunks(paper_key);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, authors, abstract, content='papers', content_rowid='rowid', tokenize='porter unicode61');
"""


def connect(path=None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: the full-screen app loads the index in one worker thread and queries it from
    # others. Safe here because this SQLite build is compiled serialized (sqlite3.threadsafety == 3).
    con = sqlite3.connect(path or DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(chunks)")}
    if "section" not in cols:          # migrate indexes built before section labels existed
        con.execute("ALTER TABLE chunks ADD COLUMN section TEXT")
        con.commit()
    return con


def utf8_console():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_reranker(name):
    """A cross-encoder that scores (query, passage) pairs jointly; fp16 on the GPU when available."""
    import torch
    from sentence_transformers import CrossEncoder
    cuda = torch.cuda.is_available()
    return CrossEncoder(name, device="cuda" if cuda else "cpu", max_length=512,
                        model_kwargs={"torch_dtype": torch.float16} if cuda else {})


def load_encoder():
    """bge-small on the GPU in fp16 when available, else CPU."""
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMB_MODEL, device=device)
    if device == "cuda":
        model.half()
    model.max_seq_length = 512
    return model, device
