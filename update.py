"""One-command update: fetch papers JMLR has published since the last run and add them everywhere.

  jmlr-rag update            (or: python update.py)
  jmlr-rag update --check    only report how many papers JMLR lists that aren't indexed yet (no downloads)

Steps (each one skips work that is already done, so a run with nothing new takes a few minutes):
  1. jmlr_dl.py        download new PDFs / abstracts / BibTeX politely (3 requests at a time, with pauses)
  2. build_index.py    extract text, passages, section labels for new PDFs; refresh the keyword index
  3. embed.py          embed the new passages (GPU; close the app first so they don't compete for it)
  4. topics.py assign  put new papers into their nearest existing topic-map subtopic
A summary of what changed is printed at the end and appended to data/update_log.txt.
"""
import argparse
import re
import subprocess
import sys
import time
from datetime import datetime

from common import DATA_DIR, JMLR_DIR, ROOT, connect, utf8_console

LOG = DATA_DIR / "update_log.txt"


def counts():
    con = connect()
    papers = con.execute("SELECT COUNT(*) FROM papers WHERE status='done'").fetchone()[0]
    try:
        topics = con.execute("SELECT COUNT(DISTINCT paper_key) FROM paper_topics").fetchone()[0]
    except Exception:
        topics = 0
    return papers, topics


def remote_count():
    """How many papers JMLR's volume pages list right now, parsed exactly as the downloader does
    (reads ~27 small pages, no PDFs). Returns (listed, volumes, keys listed but not indexed)."""
    import jmlr_dl
    idx = jmlr_dl.get(jmlr_dl.BASE).decode("utf-8", "replace")
    vols = sorted({int(n) for n in re.findall(r'href="v(\d+)"', idx)})
    keys = set()
    for v in vols:
        url = f"{jmlr_dl.BASE}v{v}/"
        page = jmlr_dl.get(url).decode("utf-8", "replace")
        keys |= {f"v{v:02d}/{p['id']}" for p in jmlr_dl.parse_volume(v, page, url)}
        time.sleep(0.5)
    have = {r[0] for r in connect().execute("SELECT key FROM papers WHERE status='done'")}
    return len(keys), len(vols), sorted(keys - have)


def step(title, args):
    print(f"\n=== {title} ===", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, *args], cwd=ROOT)
    print(f"--- {title}: {'done' if proc.returncode == 0 else f'FAILED (exit {proc.returncode})'} "
          f"in {time.time() - t0:.0f}s", flush=True)
    return proc.returncode == 0


def run(check_only=False):
    utf8_console()
    before, _ = counts()
    if check_only:
        listed, vols, missing = remote_count()
        print(f"JMLR lists {listed} papers in {vols} volumes; {before} are indexed here.")
        if missing:
            print(f"Not indexed yet ({len(missing)}): {', '.join(missing[:15])}{' ...' if len(missing) > 15 else ''}")
            print("(v24/22-1047 is expected: its PDF on JMLR's server is empty.)" if "v24/22-1047" in missing
                  else "")
        return
    t0 = time.time()
    ok = (step("1/4 Download new papers", [str(ROOT / "jmlr_dl.py"), str(JMLR_DIR)])
          and step("2/4 Extract text and update the keyword index", [str(ROOT / "build_index.py")])
          and step("3/4 Embed new passages", [str(ROOT / "embed.py")])
          and step("4/4 Place new papers in the topic map", [str(ROOT / "topics.py"), "assign"]))
    after, in_topics = counts()
    summary = (f"{datetime.now().isoformat(timespec='seconds')}  {'ok' if ok else 'FAILED'}  "
               f"papers {before} -> {after} (+{after - before}), in topic map {in_topics}, "
               f"{(time.time() - t0) / 60:.0f} min")
    print(f"\n{summary}")
    if after > before:
        con = connect()
        new = con.execute("SELECT key, title, year FROM papers WHERE status='done' "      # newest rows first
                          "ORDER BY rowid DESC LIMIT ?", (after - before,)).fetchall()
        print("New papers:")
        for r in new[:30]:
            print(f"  {r['year']}  {r['title'][:90]}  [{r['key']}]")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(summary + "\n")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only compare JMLR's listing with the local index")
    run(ap.parse_args().check)


if __name__ == "__main__":
    main()
