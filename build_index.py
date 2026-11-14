"""Step 1: extract text from the downloaded PDFs into a SQLite database with full-text search.

Resumable: papers already marked 'done' are skipped, so you can re-run it after more
PDFs arrive.  Usage:  python build_index.py
       python build_index.py --sections-only   (add/refresh section labels on an existing index)
"""
import csv
import html
import re
import sys
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


NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][^\n]{2,80})$")
NAMED_HEADING = re.compile(r"^(Abstract|References|Bibliography|Acknowledge?ments?|Appendix(?: [A-Z0-9][^\n]{0,60})?)$")
REF_SECTION = re.compile(r"^(\d+(\.\d+)*\.?\s*)?(references|bibliography)\b", re.I)


def norm_word(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def norm_tokens(s):
    """Same normalisation as the page words (so 'Multi-Modal' matches 'multimodal')."""
    return [t for t in (norm_word(w) for w in s.split()) if t]


def toc_headings(doc):
    """(page, label) from the PDF's bookmarks; nested entries become 'Parent / Child'."""
    out, parents = [], {}
    for level, title, page in doc.get_toc(simple=True):
        title = re.sub(r"\s+", " ", title).strip()
        if not title or page < 1 or level > 3:
            continue
        parents[level] = title
        label = title if level == 1 else f"{parents.get(1, '')} / {title}".strip(" /")
        out.append((page, title, label))
    return out


def text_headings(raw_pages):
    """Fallback for PDFs without bookmarks: numbered heading lines whose numbers keep increasing."""
    out, last, parents = [], (0,), {}
    for pno, raw in enumerate(raw_pages, start=1):
        for line in raw.split("\n"):
            line = re.sub(r"\s+", " ", line).strip()
            m = NUMBERED_HEADING.match(line)
            if m and len(line.split()) <= 10 and not line.endswith((".", ",", ";", ":")):
                num = tuple(int(x) for x in m.group(1).split("."))
                if num > last and num[0] <= last[0] + 3:   # tolerate a heading or two missed by extraction
                    last = num
                    title = f"{m.group(1)} {m.group(2).strip()}"
                    parents[len(num)] = title
                    label = title if len(num) == 1 else f"{parents.get(1, '')} / {title}".strip(" /")
                    out.append((pno, m.group(2).strip(), label))
            elif NAMED_HEADING.match(line) and not line.endswith("."):   # "Appendix B.2." = a cross-reference
                out.append((pno, line, line))
    return out


def locate_headings(headings, words, page_start):
    """Word index where each heading's text appears, searching from its page and never before the
    previous heading (headings come in document order); falls back to that start point."""
    lowered = [norm_word(w) for w in words]
    located, prev = [], 0
    for page, title, label in headings:
        if page > len(page_start):
            continue
        toks = norm_tokens(title)[:4]
        lo = max(page_start[page - 1], prev)
        hi = page_start[page + 1] if page + 1 < len(page_start) else len(words)
        pos = lo
        if toks:
            for i in range(lo, max(hi - len(toks) + 1, lo)):
                if lowered[i: i + len(toks)] == toks:
                    pos = i
                    break
        located.append((pos, label))
        prev = pos
    return located


def section_at(located, idx):
    label = ""
    for pos, lab in located:
        if pos > idx:
            break
        label = lab
    return label


def extract(job):
    key, pdf_path = job
    try:
        doc = pymupdf.open(pdf_path)
        raw_pages = [p.get_text() for p in doc]
        headings = toc_headings(doc)
        doc.close()
    except Exception as e:
        return key, None, f"error: {e}"
    pages = [clean_page(t) for t in raw_pages]
    if len(headings) < 3:
        headings = text_headings(raw_pages)
    # flatten to (word, page) and cut into overlapping windows
    words, page_of, page_start = [], [], []
    for i, ptxt in enumerate(pages, start=1):
        w = ptxt.split()
        page_start.append(len(words))
        words += w
        page_of += [i] * len(w)
    located = locate_headings(headings, words, page_start)
    chunks, step = [], CHUNK_WORDS - CHUNK_OVERLAP
    for seq, start in enumerate(range(0, max(len(words) - CHUNK_OVERLAP, 1), step)):
        win = words[start: start + CHUNK_WORDS]
        if len(win) < 30:
            continue
        section = section_at(located, start + len(win) // 2)
        is_ref = is_reference_chunk(win) or bool(REF_SECTION.match(section.split(" / ")[-1]))
        chunks.append((seq, page_of[start], page_of[start + len(win) - 1], int(is_ref), " ".join(win), section))
    if not chunks:
        return key, (len(pages), []), "no text (scanned PDF?)"
    return key, (len(pages), chunks), "done"


def update_sections(con, workers=8):
    """Add section labels to an existing index without changing chunk ids (no re-embedding needed).
    Bibliography flags are only ever added, never removed, so the embeddings stay valid."""
    papers = [(r["key"], str(JMLR_DIR / r["pdf_path"]), r["n_chunks"])
              for r in con.execute("SELECT key, pdf_path, n_chunks FROM papers WHERE status='done'")]
    t0, labelled, mismatched = time.time(), 0, []
    with Pool(workers) as pool:
        for n, (key, result, status) in enumerate(pool.imap_unordered(extract, [(k, p) for k, p, _ in papers],
                                                                       chunksize=4), 1):
            have = {r["seq"]: r for r in con.execute("SELECT seq, text FROM chunks WHERE paper_key=?", (key,))}
            chunks = result[1] if result else []
            if len(chunks) != len(have) or any(have.get(c[0]) is None or have[c[0]]["text"] != c[4] for c in chunks):
                mismatched.append(key)
                continue
            con.executemany("UPDATE chunks SET section=?, is_ref=MAX(is_ref, ?) WHERE paper_key=? AND seq=?",
                            [(c[5], c[3], key, c[0]) for c in chunks])
            labelled += sum(1 for c in chunks if c[5])
            if n % 500 == 0:
                con.commit()
                print(f"  {n}/{len(papers)} papers ({time.time() - t0:.0f}s)", flush=True)
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"Section labels on {labelled}/{total} passages; {len(mismatched)} papers skipped "
          f"(text differs from the index){': ' + ', '.join(mismatched[:5]) if mismatched else ''}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Extract text, passages and section labels from the downloaded PDFs "
                                 "into the SQLite index (incremental: only new PDFs are processed).")
    ap.add_argument("--sections-only", action="store_true",
                    help="only (re)compute section labels on the existing index (chunk ids unchanged)")
    ap.add_argument("--db", help="--sections-only: work on this database file instead (e.g. a copy)")
    ap.add_argument("--workers", type=int, default=8, help="--sections-only: parallel PDF readers (default 8)")
    a = ap.parse_args()
    utf8_console()
    if a.sections_only:
        return update_sections(connect(a.db), a.workers)
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
                con.executemany("INSERT INTO chunks(paper_key, seq, page_start, page_end, is_ref, text, section) "
                                "VALUES(?,?,?,?,?,?,?)", [(key, *c) for c in chunks])
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
