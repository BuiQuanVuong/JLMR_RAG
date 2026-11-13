"""BibTeX export for JMLR papers.

  python ask.py cite v13/gretton12a v24/21-1289                 print BibTeX
  python ask.py cite v13/gretton12a v24/21-1289 --out refs.bib  write a UTF-8 .bib file

Uses JMLR's own .bib file when it was downloaded (volumes 10 onwards); for older volumes the
entry is generated from the catalog (authors, title, year, volume, issue/month, pages, URL).
Use --out rather than PowerShell's '>' redirection, which writes UTF-16 that BibTeX can't read.
"""
import re
import sys

from common import JMLR_DIR, connect

MONTHS = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}


def generated_entry(p):
    """BibTeX built from catalog fields, in the same style as JMLR's own files."""
    cit = p["citation"] or ""
    pages = re.search(r"(\d+)\s*[-−–]+\s*(\d+)", cit)
    issue = re.search(r"\((\w+)\)", cit)
    fields = [
        ("author", " and ".join(a.strip() for a in (p["authors"] or "").split(",") if a.strip())),
        ("title", p["title"]),
        ("journal", "Journal of Machine Learning Research"),
        ("year", str(p["year"] or "")),
        ("volume", str(p["volume"])),
    ]
    if issue:
        fields.append(("month" if issue.group(1) in MONTHS else "number", issue.group(1)))
    if pages:
        fields.append(("pages", f"{pages.group(1)}--{pages.group(2)}"))
    fields.append(("url", f"http://jmlr.org/papers/v{p['volume']}/{p['pid']}.html"))
    body = ",\n".join(f"  {k:<7} = {{{v}}}" for k, v in fields if v)
    return f"@article{{JMLR:v{p['volume']}:{p['pid']},\n{body}\n}}"


def entry_for(p):
    bib = JMLR_DIR / f"v{p['volume']:02d}" / "bib" / f"{p['pid']}.bib"
    if bib.exists() and bib.stat().st_size > 0:
        return bib.read_text(encoding="utf-8", errors="replace").strip()
    return generated_entry(p)


def run(keys, out=None):
    con = connect()
    entries, missing = [], []
    for key in keys:
        p = con.execute("SELECT * FROM papers WHERE key=?", (key,)).fetchone()
        if p is None:
            missing.append(key)
            continue
        entries.append(entry_for(p))
    if missing:
        print(f"Unknown paper key(s): {', '.join(missing)} (format: v20/13-124)", file=sys.stderr)
    if not entries:
        sys.exit(1)
    text = "\n\n".join(entries) + "\n"
    if out:
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"Wrote {len(entries)} BibTeX entries to {out}")
    else:
        print(text, end="")
