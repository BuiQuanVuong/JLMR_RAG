"""Download all JMLR papers (PDF + abstract page + BibTeX) and build a CSV catalog.
Resumable: existing non-empty files are skipped. Usage: python -I jmlr_dl.py OUTDIR
"""
import csv, html, os, re, sys, time, threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = "https://www.jmlr.org/papers/"
OUT = sys.argv[1] if __name__ == "__main__" else None   # importable (update.py reuses parse_volume)
UA = "Mozilla/5.0 (personal archive script)"
WORKERS = 3
lock = threading.Lock()
log_f = None


def log(msg):
    with lock:
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if getattr(e, "code", None) == 404 or i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def save(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "skip"
    data = get(url)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    time.sleep(0.3)
    return "ok"


def text(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s))).strip()


def parse_volume(v, page, vol_url):
    papers = []
    for block in re.split(r"<dt>", page, flags=re.I)[1:]:
        m = re.match(r"(.*?)(?:</dt>|<dd>)", block, re.S | re.I)
        title = text(m.group(1)) if m else ""
        dd = re.search(r"<dd>(.*)", block, re.S | re.I)
        dd = dd.group(1) if dd else ""
        am = re.search(r"<i>(.*?)</(?:i|b)>", dd, re.S | re.I)
        authors = text(am.group(1)) if am else ""
        after = re.split(r"</i>|</b>", dd, maxsplit=2)
        cite = ""
        cm = re.search(r";(.*?)(?:<br|<a)", dd, re.S | re.I)
        if cm:
            cite = text(cm.group(1))
        links = {}
        for href, label in re.findall(r"<a[^>]*href=['\"]?([^'\" >]+)['\"]?[^>]*>(.*?)</a>", dd, re.S | re.I):
            lab = text(label).strip("[]").lower()
            links.setdefault(lab, urljoin(vol_url, href).replace("http://", "https://"))
        pdf = links.get("pdf")
        if not pdf:
            continue
        pid = os.path.splitext(os.path.basename(pdf))[0]
        papers.append(dict(volume=v, id=pid, title=title, authors=authors, citation=cite,
                           pdf_url=pdf, abs_url=links.get("abs", ""), bib_url=links.get("bib", ""),
                           extra={k: u for k, u in links.items() if k not in ("pdf", "abs", "bib")}))
    return papers


def main():
    global log_f
    os.makedirs(OUT, exist_ok=True)
    log_f = open(os.path.join(OUT, "download.log"), "a", encoding="utf-8")
    save(BASE, os.path.join(OUT, "index.html"))
    idx = get(BASE).decode("utf-8", "replace")
    vols = sorted({int(n) for n in re.findall(r'href="v(\d+)"', idx)})
    all_papers = []
    for v in vols:
        vdir = os.path.join(OUT, f"v{v:02d}")
        os.makedirs(vdir, exist_ok=True)
        vol_url = f"{BASE}v{v}/"
        page = get(vol_url).decode("utf-8", "replace")
        with open(os.path.join(vdir, "_volume_index.html"), "w", encoding="utf-8") as f:
            f.write(page)
        ps = parse_volume(v, page, vol_url)
        log(f"Volume {v}: {len(ps)} papers")
        all_papers += ps

    fields = ["volume", "id", "title", "authors", "citation", "pdf_url", "abs_url", "bib_url", "local_pdf"]
    with open(os.path.join(OUT, "catalog.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for p in all_papers:
            p["local_pdf"] = f"v{p['volume']:02d}/{p['id']}.pdf"
            w.writerow(p)

    jobs = []
    for p in all_papers:
        vdir = os.path.join(OUT, f"v{p['volume']:02d}")
        jobs.append((p["pdf_url"], os.path.join(vdir, p["id"] + ".pdf")))
        if p["abs_url"]:
            jobs.append((p["abs_url"], os.path.join(vdir, "abs", p["id"] + ".html")))
        if p["bib_url"]:
            jobs.append((p["bib_url"], os.path.join(vdir, "bib", p["id"] + ".bib")))
    for _, path in jobs:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    log(f"Total papers: {len(all_papers)}; files to fetch: {len(jobs)}")
    done = fail = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(save, u, p): (u, p) for u, p in jobs}
        for fut in as_completed(futs):
            u, p = futs[fut]
            done += 1
            try:
                fut.result()
            except Exception as e:
                fail += 1
                log(f"FAIL {u} -> {e}")
            if done % 200 == 0:
                log(f"progress {done}/{len(jobs)} (failures {fail})")
    log(f"DONE {done}/{len(jobs)}, failures {fail}")


if __name__ == "__main__":
    main()
