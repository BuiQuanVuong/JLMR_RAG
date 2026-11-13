"""Step 1: extract text from the downloaded PDFs into a SQLite database with full-text search.

Resumable: papers already marked 'done' are skipped, so you can re-run it after more
PDFs arrive.  Usage:  python build_index.py
"""
import csv
import html
import re
import time
from multiprocessing import Pool

import pymupdf

from common import CHUNK_OVERLAP, CHUNK_WORDS, JMLR_DIR, connect, utf8_console

YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
VENUE_RE = re.compile(r"\b(Journal|Proceedings|Conference|Annals|Transactions|arXiv|preprint|Advances in|"
                      r"In Proc|Press|pp\.|pages)\b|\d+\(\d+\)[:,]\s*\d+|\b[A-Z]\. ?[A-Z]?\.? ?[A-Z][a-z]+,")


def parse_abstract(abs_path):
    try:
        page = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.search(r"<h3>\s*Abstract\s*</h3>(.*)", page, re.S | re.I)
    if not m:
        return ""
    body = m.group(1)
    end = re.search(r"<font|\[abs\]|<table|</div>|<h3", body, re.I)
    body = body[: end.start()] if end else body[:5000]
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body))).strip()


def clean_page(text):
    text = text.replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", text).strip()


def is_reference_chunk(words):
    """Bibliography chunks are dense with years, venue names and volume(issue):page patterns."""
    text = " ".join(words)
    per100 = 100 / max(len(words), 1)
    years = len(YEAR_RE.findall(text)) * per100
    venues = len(VENUE_RE.findall(text)) * per100
    return years > 3.5 or years + venues > 8


def extract(job):
    key, pdf_path = job
    try:
        doc = pymupdf.open(pdf_path)
        pages = [clean_page(p.get_text()) for p in doc]
        doc.close()
    except Exception as e:
        return key, None, f"error: {e}"
    # flatten to (word, page) and cut into overlapping windows
    words, page_of = [], []
    for i, ptxt in enumerate(pages, start=1):
        w = ptxt.split()
        words += w
        page_of += [i] * len(w)
    chunks, step = [], CHUNK_WORDS - CHUNK_OVERLAP
    for seq, start in enumerate(range(0, max(len(words) - CHUNK_OVERLAP, 1), step)):
        win = words[start: start + CHUNK_WORDS]
        if len(win) < 30:
            continue
        chunks.append((seq, page_of[start], page_of[start + len(win) - 1],
                       int(is_reference_chunk(win)), " ".join(win)))
    if not chunks:
        return key, (len(pages), []), "no text (scanned PDF?)"
    return key, (len(pages), chunks), "done"


def main():
    utf8_console()
    con = connect()
    catalog = JMLR_DIR / "catalog.csv"
    rows = list(csv.DictReader(open(catalog, encoding="utf-8-sig")))
    for r in rows:
        v = int(r["volume"])
        key = f"v{v:02d}/{r['id']}"
        ym = re.search(r"(\d{4})\.?\s*$", r["citation"])
        abstract = parse_abstract(JMLR_DIR / f"v{v:02d}" / "abs" / f"{r['id']}.html")
        con.execute("""INSERT INTO papers(key, volume, pid, title, authors, citation, year, abstract,
                           pdf_path, pdf_url, abs_url, status)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?, 'pending')
                       ON CONFLICT(key) DO UPDATE SET title=excluded.title, authors=excluded.authors,
                           citation=excluded.citation, year=excluded.year,
                           abstract=CASE WHEN excluded.abstract != '' THEN excluded.abstract ELSE papers.abstract END""",
                    (key, v, r["id"], r["title"], r["authors"], r["citation"],
                     int(ym.group(1)) if ym else None, abstract,
                     r["local_pdf"], r["pdf_url"], r["abs_url"]))
    con.commit()

    todo = []
    for row in con.execute("SELECT key, pdf_path FROM papers WHERE status != 'done' ORDER BY key"):
        p = JMLR_DIR / row["pdf_path"]
        if p.exists():
            todo.append((row["key"], str(p)))
    missing = con.execute("SELECT COUNT(*) FROM papers WHERE status != 'done'").fetchone()[0] - len(todo)
    print(f"{len(rows)} papers in catalog; extracting {len(todo)} (PDF not downloaded yet: {missing})")

    t0, n = time.time(), 0
    with Pool(8) as pool:
        for key, result, status in pool.imap_unordered(extract, todo, chunksize=4):
            n += 1
            con.execute("DELETE FROM chunks WHERE paper_key=?", (key,))
            if result:
                n_pages, chunks = result
                con.executemany("INSERT INTO chunks(paper_key, seq, page_start, page_end, is_ref, text) "
                                "VALUES(?,?,?,?,?,?)", [(key, *c) for c in chunks])
                con.execute("UPDATE papers SET n_pages=?, n_chunks=?, status=? WHERE key=?",
                            (n_pages, len(chunks), status, key))
            else:
                con.execute("UPDATE papers SET status=? WHERE key=?", (status, key))
            if status != "done":
                print(f"  {key}: {status}")
            if n % 200 == 0:
                con.commit()
                print(f"  {n}/{len(todo)} papers  ({time.time() - t0:.0f}s)")
    con.commit()

    print("Rebuilding full-text search indexes...")
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    con.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    con.commit()
    s = con.execute("""SELECT COUNT(*) n, SUM(status='done') done, (SELECT COUNT(*) FROM chunks) c,
                              (SELECT SUM(is_ref) FROM chunks) r FROM papers""").fetchone()
    print(f"Done: {s['done']}/{s['n']} papers indexed, {s['c']} chunks ({s['r']} bibliography chunks excluded "
          f"from search), {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
