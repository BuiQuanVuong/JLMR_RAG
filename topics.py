"""Topic map: a two-level hierarchy of research areas and subtopics over all JMLR papers.

Build (local CPU, plus ~$0.02 of Claude Haiku for the names):
  python topics.py build            sweep granularity, cluster, name, store, draw topic_map.png
  python topics.py build --no-names cluster and store without calling the API (names = top terms)
  python topics.py assign           put papers added later into their nearest existing subtopic
  python topics.py map              redraw topic_map.png (e.g. after renaming topics in topic_names.json)

Browse (via ask.py):
  python ask.py topics              areas with sizes and trend lines
  python ask.py topics 3            subtopics of area 3
  python ask.py topics 3.2          a subtopic: description, trend, top authors, central papers
  python ask.py topics --rising     subtopics that grew / shrank most (2020-25 vs 2005-19)
  python ask.py topics --for "..."  which subtopics a question falls into

How the granularity is chosen: every setting in the sweep is scored on (1) agreement (adjusted mutual
information) with JMLR's own topic / special-issue pages, (2) silhouette, (3) size balance. The best
setting inside a browsable range wins; scores alone favour few broad clusters, browsing favours more.
Names can be edited in data/topic_names.json (shown instead of the generated ones).
"""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import CHUNK_EMB, CHUNK_IDS, DATA_DIR, JMLR_DIR, PAPER_EMB, PAPER_KEYS, ROOT, connect, utf8_console

NAMES_FILE = DATA_DIR / "topic_names.json"
META_FILE = DATA_DIR / "topic_meta.json"
MAP_PNG = ROOT / "topic_map.png"
NAME_MODEL = "claude-haiku-5-5"

AREA_KS = [6, 8, 10, 12, 15, 18, 20, 25]       # level-1 sweep
BROWSABLE = (8, 25)                             # level-1 sizes we are willing to pick
SUB_SIZES = [30, 40, 60]                        # target papers per subtopic (level-2 sweep)
MIN_SIZE = 15                                   # smallest cluster worth showing
SECONDARY_MARGIN = 0.0                          # 2nd subtopic only if at least as close as the paper's own
                                                # (22% of papers; a 0.03 margin gave 82%, too many to mean much)
PERIODS = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2025)]

# JMLR topic pages used as an external check; event-based special issues are not topics
SKIP_GROUPS = {"COLT2005", "colt02", "icml01", "chervonenkis15"}
MERGE_GROUPS = {"learning_theory03": "learning_theory", "ilp03": "inductive_programming"}

STOPWORDS = set("""a an the of and or in on for to with by from is are was were be been being as at that this these
those it its we our us they their them which who whom how what when where while than then there here such any all
each some other more most less many much very can could should would will may might must also into over under
between both either neither not no nor only own same so too via using use used based new paper propose proposed
show shows shown present presents study studies approach approaches method methods problem problems result results
algorithm algorithms model models data learning machine framework provide provides general one two three first
second however well large small number given different several case cases set sets work recent thus""".split())


# ---------------------------------------------------------------- data

def paper_rows(con):
    rows = con.execute("SELECT key, title, authors, year, abstract FROM papers WHERE status='done'").fetchall()
    return {r["key"]: dict(r) for r in rows}


def representations(con):
    """Paper vectors: 'abstract' (title+abstract embedding), 'fulltext' (mean passage embedding), 'combined'."""
    keys = [str(k) for k in np.load(PAPER_KEYS)]
    abstract = np.load(PAPER_EMB).astype(np.float32)
    ids, emb = np.load(CHUNK_IDS), np.load(CHUNK_EMB)
    key_of = dict(con.execute("SELECT id, paper_key FROM chunks WHERE is_ref=0").fetchall())
    index = {k: i for i, k in enumerate(keys)}
    rows = np.array([index.get(key_of.get(int(c), ""), -1) for c in ids])
    keep = rows >= 0
    full = np.zeros_like(abstract)
    np.add.at(full, rows[keep], emb[keep].astype(np.float32))
    norm = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)  # noqa: E731
    full = norm(full)
    return keys, {"abstract": norm(abstract), "fulltext": full, "combined": norm(norm(abstract) + full)}


def reference_labels():
    """paper key -> JMLR topic group, from the downloaded topic / special-issue pages."""
    out = {}
    for f in sorted((JMLR_DIR / "topic_pages").glob("*.html")):
        group = f.stem
        if group in SKIP_GROUPS:
            continue
        group = MERGE_GROUPS.get(group, group)
        for vol, pid in re.findall(r"volume(\d+)/([^/\"']+)/", f.read_text(encoding="utf-8", errors="replace")):
            out.setdefault(f"v{int(vol):02d}/{pid}", group)
    return out


# ---------------------------------------------------------------- clustering

def score_partition(X, labels, ref_idx, ref_lab):
    from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]
    return {"ami": adjusted_mutual_info_score(ref_lab, labels[ref_idx]),
            "silhouette": silhouette_score(X, labels, metric="cosine"),
            "largest": sizes.max() / len(labels), "smallest": int(sizes.min()), "n": len(sizes)}


def sweep(keys, reps, ref):
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.cluster import KMeans
    ref_idx = np.array([i for i, k in enumerate(keys) if k in ref])
    groups = sorted({ref[keys[i]] for i in ref_idx})
    ref_lab = np.array([groups.index(ref[keys[i]]) for i in ref_idx])
    print(f"External check: {len(ref_idx)} papers from {len(groups)} JMLR topic groups\n")
    print(f"{'representation':<11}{'method':<8}{'k':>4}{'AMI':>7}{'silh.':>7}{'largest':>9}{'smallest':>9}")
    results = []
    for rep, X in reps.items():
        Z = linkage(X, method="ward")
        for k in AREA_KS:
            lab = fcluster(Z, k, criterion="maxclust") - 1
            s = score_partition(X, lab, ref_idx, ref_lab)
            results.append(dict(rep=rep, method="ward", k=k, labels=lab, **s))
        for k in (10, 15, 20):
            lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
            s = score_partition(X, lab, ref_idx, ref_lab)
            results.append(dict(rep=rep, method="kmeans", k=k, labels=lab, **s))
    for r in results:
        print(f"{r['rep']:<11}{r['method']:<8}{r['k']:>4}{r['ami']:>7.3f}{r['silhouette']:>7.3f}"
              f"{r['largest']:>8.0%}{r['smallest']:>9}")
    ok = [r for r in results if BROWSABLE[0] <= r["k"] <= BROWSABLE[1] and r["largest"] <= 0.25
          and r["smallest"] >= MIN_SIZE]
    best = max(ok or results, key=lambda r: (round(r["ami"], 2), r["silhouette"]))
    print(f"\nChosen areas: {best['rep']} / {best['method']} / k={best['k']} "
          f"(AMI {best['ami']:.3f}, silhouette {best['silhouette']:.3f})")
    return best, ref_idx, ref_lab


def subclusters(X, areas, target):
    """Split each area into clamp(round(n/target), 1..10) subtopics (Ward within the area)."""
    from sklearn.cluster import AgglomerativeClustering
    sub = np.zeros(len(areas), dtype=int)
    next_id = 0
    for a in np.unique(areas):
        idx = np.where(areas == a)[0]
        k = int(np.clip(round(len(idx) / target), 1, 10))
        if len(idx) < 2 * MIN_SIZE:
            k = 1
        lab = np.zeros(len(idx), dtype=int) if k == 1 else \
            AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X[idx])
        sub[idx] = lab + next_id
        next_id += k
    return sub


def order_ids(areas, subs):
    """Readable ids: areas 1..K by size, subtopics a.1..a.n by size within their area."""
    area_rank = {a: i + 1 for i, (a, _) in enumerate(Counter(areas).most_common())}
    sub_id = {}
    for a, rank in area_rank.items():
        within = Counter(subs[areas == a]).most_common()
        for j, (s, _) in enumerate(within, 1):
            sub_id[s] = f"{rank}.{j}"
    return [str(area_rank[a]) for a in areas], [sub_id[s] for s in subs]


# ---------------------------------------------------------------- naming

def distinctive_terms(texts_by_cluster, n=12):
    """c-TF-IDF over title+abstract words: frequent in the cluster, rare elsewhere."""
    counts = {c: Counter(w for t in texts for w in re.findall(r"[a-z][a-z\-]{2,}", t.lower())
                         if w not in STOPWORDS) for c, texts in texts_by_cluster.items()}
    total = Counter()
    for c in counts.values():
        total.update(c)
    avg = sum(total.values()) / max(len(counts), 1)
    out = {}
    for c, cnt in counts.items():
        size = sum(cnt.values()) or 1
        scored = {w: (f / size) * np.log(1 + avg / total[w]) for w, f in cnt.items() if f >= 3}
        out[c] = [w for w, _ in sorted(scored.items(), key=lambda x: -x[1])[:n]]
    return out


NAME_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
               "required": ["name", "description"], "additionalProperties": False}


def name_cluster(client, kind, terms, titles, area_terms=None, subtopic_names=None):
    """Subtopics are named first (from their own papers); an area is then named from ALL its subtopics, so
    its name covers everything inside it rather than just its most central papers."""
    if kind == "subtopic":
        where = f"a subtopic inside a broader area characterised by: {', '.join(area_terms or [])}"
        rule = "2-6 words, specific to these papers"
    else:
        where = "one broad research area"
        rule = "2-5 words, broad enough to cover every subtopic listed, e.g. Kernel Methods, Causal Inference"
    prompt = (f"In a clustering of all Journal of Machine Learning Research papers (2000-2025), these papers "
              f"form {where}.\n\nDistinctive terms: {', '.join(terms)}\n\n"
              + (f"Its subtopics:\n" + "\n".join(f"- {s}" for s in subtopic_names) + "\n\n" if subtopic_names else "")
              + "Most central papers:\n" + "\n".join(f"- {t}" for t in titles)
              + f"\n\nName this {kind} the way an ML researcher would ({rule}) and describe it in one sentence "
                "of at most 25 words.")
    msg = client.messages.create(model=NAME_MODEL, max_tokens=1000,
                                 output_config={"effort": "low",
                                                "format": {"type": "json_schema", "schema": NAME_SCHEMA}},
                                 messages=[{"role": "user", "content": prompt}])
    if msg.stop_reason != "end_turn":
        return None
    return json.loads(next(b.text for b in msg.content if b.type == "text"))


# ---------------------------------------------------------------- storage

TOPIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS topics(id TEXT PRIMARY KEY, parent TEXT, level INTEGER, name TEXT, description TEXT,
                                  size INTEGER, terms TEXT, centroid BLOB);
CREATE TABLE IF NOT EXISTS paper_topics(paper_key TEXT, topic_id TEXT, rank INTEGER, score REAL,
                                        PRIMARY KEY(paper_key, rank));
"""


def store(con, topics, assignments):
    con.executescript(TOPIC_SCHEMA)
    con.execute("DELETE FROM topics")
    con.execute("DELETE FROM paper_topics")
    con.executemany("INSERT INTO topics VALUES(?,?,?,?,?,?,?,?)",
                    [(t["id"], t["parent"], t["level"], t["name"], t["description"], t["size"],
                      json.dumps(t["terms"]), t["centroid"].astype(np.float32).tobytes()) for t in topics])
    con.executemany("INSERT INTO paper_topics VALUES(?,?,?,?)", assignments)
    con.commit()


def load_names():
    try:
        return json.loads(NAMES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def has_topics(con):
    return con.execute("SELECT 1 FROM sqlite_master WHERE name='paper_topics'").fetchone() is not None


# ---------------------------------------------------------------- build

def build(no_names=False):
    con = connect()
    papers = paper_rows(con)
    keys, reps = representations(con)
    keys_arr = np.array(keys)
    ref = reference_labels()
    best, ref_idx, ref_lab = sweep(keys, reps, ref)
    X, areas = reps[best["rep"]], best["labels"]

    print(f"\n{'papers per subtopic':<22}{'subtopics':>10}{'AMI':>7}{'smallest':>9}")
    sub_runs = []
    for target in SUB_SIZES:
        subs = subclusters(X, areas, target)
        s = score_partition(X, subs, ref_idx, ref_lab)
        sub_runs.append((target, subs, s))
        print(f"{target:<22}{s['n']:>10}{s['ami']:>7.3f}{s['smallest']:>9}")
    target, subs, s = max(sub_runs, key=lambda r: round(r[2]["ami"], 2) + (0.001 if r[0] == 40 else 0))
    print(f"Chosen subtopics: ~{target} papers each ({s['n']} subtopics, AMI {s['ami']:.3f})")

    area_ids, sub_ids = order_ids(areas, subs)
    area_ids, sub_ids = np.array(area_ids), np.array(sub_ids)
    norm = lambda v: v / max(np.linalg.norm(v), 1e-9)  # noqa: E731

    topics = []
    for level, ids in ((1, area_ids), (2, sub_ids)):
        for tid in sorted(set(ids), key=lambda t: [int(x) for x in t.split(".")]):
            members = np.where(ids == tid)[0]
            topics.append(dict(id=tid, parent=tid.split(".")[0] if level == 2 else None, level=level,
                               members=members, centroid=norm(X[members].mean(0)), size=len(members)))
    texts = {t["id"]: [f"{papers[keys[i]]['title']} {papers[keys[i]]['abstract'] or ''}" for i in t["members"]]
             for t in topics}
    terms = {**distinctive_terms({t: v for t, v in texts.items() if "." not in t}),
             **distinctive_terms({t: v for t, v in texts.items() if "." in t})}
    for t in topics:
        t["terms"] = terms[t["id"]]
        sims = X[t["members"]] @ t["centroid"]
        t["central"] = [keys[i] for i in t["members"][np.argsort(-sims)]]

    # names: areas first (subtopic prompts mention their area's name)
    for t in topics:
        t["name"], t["description"] = ", ".join(t["terms"][:3]).title(), ""
    if not no_names:
        import anthropic
        client = anthropic.Anthropic()
        by_id = {t["id"]: t for t in topics}

        def run(level):
            todo = [t for t in topics if t["level"] == level]

            def one(t):
                titles = [f"{papers[k]['title']} ({papers[k]['year']})" for k in t["central"][:10]]
                try:
                    if level == 2:
                        return t, name_cluster(client, "subtopic", t["terms"], titles,
                                               area_terms=by_id[t["parent"]]["terms"][:8])
                    subs = [s["name"] for s in topics if s["parent"] == t["id"]]
                    return t, name_cluster(client, "area", t["terms"], titles, subtopic_names=subs)
                except Exception as e:
                    print(f"  naming {t['id']} failed: {e}")
                    return t, None
            with ThreadPoolExecutor(max_workers=6) as pool:
                for t, res in pool.map(one, todo):
                    if res:
                        t["name"], t["description"] = res["name"].strip(), res["description"].strip()
        print("\nNaming subtopics, then areas from their subtopics, with Claude Haiku 5.5...")
        run(2)
        run(1)

    # primary + optional secondary subtopic per paper
    subs_only = [t for t in topics if t["level"] == 2]
    C = np.stack([t["centroid"] for t in subs_only])
    sims = X @ C.T
    col = {t["id"]: j for j, t in enumerate(subs_only)}
    assignments = []
    for i, k in enumerate(keys):
        own = col[sub_ids[i]]
        assignments.append((k, sub_ids[i], 1, float(sims[i, own])))
        order = np.argsort(-sims[i])
        other = next(j for j in order if j != own)
        if sims[i, other] >= sims[i, own] - SECONDARY_MARGIN:
            assignments.append((k, subs_only[other]["id"], 2, float(sims[i, other])))
    store(con, topics, assignments)
    META_FILE.write_text(json.dumps({"representation": best["rep"], "method": best["method"], "areas": best["k"],
                                     "papers_per_subtopic": target, "area_ami": round(best["ami"], 4),
                                     "subtopic_ami": round(s["ami"], 4)}, indent=1), encoding="utf-8")
    NAMES_FILE.write_text(json.dumps({t["id"]: {"name": t["name"], "description": t["description"]}
                                      for t in topics}, indent=1, ensure_ascii=False), encoding="utf-8")
    n_sec = sum(1 for a in assignments if a[2] == 2)
    print(f"\nStored {sum(t['level'] == 1 for t in topics)} areas, {len(subs_only)} subtopics; "
          f"{n_sec} papers also get a secondary subtopic. Names: {NAMES_FILE}")
    draw_map(X, area_ids, topics)
    report_reference(keys, area_ids, ref, topics)


def report_reference(keys, area_ids, ref, topics):
    """For each JMLR topic group: share of its papers in its most common area."""
    name = {t["id"]: t["name"] for t in topics}
    by_group = defaultdict(list)
    for i, k in enumerate(keys):
        if k in ref:
            by_group[ref[k]].append(area_ids[i])
    print("\nJMLR topic group -> where its papers landed (share in the most common area):")
    for g, ids in sorted(by_group.items(), key=lambda x: -len(x[1])):
        top, n = Counter(ids).most_common(1)[0]
        print(f"  {g:<38}{len(ids):>3} papers  {n / len(ids):>4.0%} in {top}. {name[top]}")


def draw_map(X, area_ids, topics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    t0 = time.time()
    Y = TSNE(n_components=2, init="pca", random_state=0, perplexity=40).fit_transform(
        PCA(n_components=50, random_state=0).fit_transform(X))
    areas = [t for t in topics if t["level"] == 1]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(14, 11), dpi=130)
    for j, t in enumerate(areas):
        m = area_ids == t["id"]
        ax.scatter(Y[m, 0], Y[m, 1], s=6, color=cmap(j % 20), alpha=0.7, linewidths=0)
    labels = []
    for j, t in enumerate(areas):
        m = area_ids == t["id"]
        cx, cy = np.median(Y[m, 0]), np.median(Y[m, 1])
        labels.append(ax.text(cx, cy, f"{t['id']}. {t['name']}", fontsize=9, weight="bold", ha="center",
                              va="center", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=cmap(j % 20),
                                                     alpha=0.85)))
    _separate_labels(fig, ax, labels)
    ax.set_title(f"JMLR 2000-2025: {len(X)} papers by research area (t-SNE of paper embeddings)", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(MAP_PNG)
    print(f"\nMap written to {MAP_PNG} ({time.time() - t0:.0f}s)")


def _separate_labels(fig, ax, labels, rounds=200, pad=4):
    """Nudge overlapping text labels apart vertically (display coordinates) until none overlap."""
    to_data = ax.transData.inverted()
    for _ in range(rounds):
        fig.canvas.draw()
        boxes = [l.get_window_extent().expanded(1.0, 1.0) for l in labels]
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = boxes[i], boxes[j]
                if a.x1 < b.x0 or b.x1 < a.x0 or a.y1 < b.y0 or b.y1 < a.y0:
                    continue
                overlap = min(a.y1, b.y1) - max(a.y0, b.y0) + pad
                step = overlap / 2
                up, down = (i, j) if (a.y0 + a.y1) >= (b.y0 + b.y1) else (j, i)
                for k, dy in ((up, step), (down, -step)):
                    x, y = ax.transData.transform(labels[k].get_position())
                    labels[k].set_position(to_data.transform((x, y + dy)))
                moved = True
        if not moved:
            return


def redraw():
    """Redraw topic_map.png from the stored map (picks up names edited in topic_names.json; no API calls)."""
    con = connect()
    if not has_topics(con):
        sys.exit("No topic map yet - run: python topics.py build")
    keys, reps = representations(con)
    try:
        rep = json.loads(META_FILE.read_text(encoding="utf-8"))["representation"]
    except Exception:
        rep = "combined"
    primary = dict(con.execute("SELECT paper_key, topic_id FROM paper_topics WHERE rank=1").fetchall())
    area_ids = np.array([primary.get(k, "0").split(".")[0] for k in keys])
    rows = _topic_rows(con)
    topics = [dict(id=t, name=r["name"], level=r["level"]) for t, r in
              sorted(rows.items(), key=lambda x: [int(p) for p in x[0].split(".")])]
    draw_map(reps[rep], area_ids, topics)


def assign_new():
    """Give papers indexed after the last build their nearest existing subtopic (keeps the map stable)."""
    con = connect()
    if not has_topics(con):
        sys.exit("No topic map yet - run: python topics.py build")
    keys, reps = representations(con)
    have = {r[0] for r in con.execute("SELECT DISTINCT paper_key FROM paper_topics")}
    rows = con.execute("SELECT id, centroid FROM topics WHERE level=2").fetchall()
    C = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    dim = C.shape[1]
    rep = next(X for X in reps.values() if X.shape[1] == dim)   # same width for every representation
    new = [(i, k) for i, k in enumerate(keys) if k not in have]
    for i, k in new:
        sims = C @ rep[i]
        con.execute("INSERT OR REPLACE INTO paper_topics VALUES(?,?,1,?)", (k, rows[int(np.argmax(sims))][0],
                                                                            float(sims.max())))
    con.commit()
    print(f"Assigned {len(new)} new papers to existing subtopics.")


# ---------------------------------------------------------------- browsing (used by ask.py)

SPARK = "▁▂▃▄▅▆▇█"


def _topic_rows(con):
    names = load_names()
    rows = {r["id"]: dict(r) for r in con.execute("SELECT id, parent, level, name, description, size, terms FROM topics")}
    for tid, r in rows.items():
        if tid in names:
            r["name"] = names[tid].get("name") or r["name"]
            r["description"] = names[tid].get("description") or r["description"]
    return rows


def _members(con, tid, primary_only=True):
    like = "" if "." in tid else "%"
    q = ("SELECT pt.paper_key, pt.score, pt.rank, p.title, p.year, p.authors FROM paper_topics pt "
         "JOIN papers p ON p.key = pt.paper_key WHERE ")
    q += "(pt.topic_id = ? OR pt.topic_id LIKE ?)" if like else "pt.topic_id = ?"
    args = (tid, f"{tid}.%") if like else (tid,)
    if primary_only:
        q += " AND pt.rank = 1"
    return [dict(r) for r in con.execute(q, args)]


def _period_counts(con):
    total = Counter()
    for (y,) in con.execute("SELECT year FROM papers WHERE status='done'"):
        total[_period(y)] += 1
    return total


def _period(y):
    return next((i for i, (a, b) in enumerate(PERIODS) if a <= (y or 0) <= b), len(PERIODS) - 1)


def _trend(members, total):
    """Share of each period's papers that belong to this topic, as a sparkline."""
    c = Counter(_period(m["year"]) for m in members)
    shares = [c[i] / max(total[i], 1) for i in range(len(PERIODS))]
    top = max(shares) or 1
    return "".join(SPARK[min(int(s / top * (len(SPARK) - 1) + 0.5), len(SPARK) - 1)] for s in shares), shares


def _authors(members, n=5):
    c = Counter(a.strip() for m in members for a in (m["authors"] or "").split(",") if a.strip())
    return ", ".join(f"{a} ({k})" for a, k in c.most_common(n))


def show(arg="", rising=False):
    utf8_console()
    con = connect()
    if not has_topics(con):
        sys.exit("No topic map yet - run: python topics.py build")
    rows, total = _topic_rows(con), _period_counts(con)
    periods = " ".join(f"{a % 100:02d}-{b % 100:02d}" for a, b in PERIODS)
    if rising:
        return show_rising(con, rows, total)
    if not arg:
        print(f"{len([r for r in rows.values() if r['level'] == 1])} research areas  "
              f"(trend = share of papers per period: {periods})\n")
        for tid in sorted((t for t in rows if rows[t]["level"] == 1), key=int):
            r = rows[tid]
            spark, _ = _trend(_members(con, tid), total)
            print(f"{tid:>3}. {r['name']:<46}{r['size']:>5} papers  {spark}")
            if r["description"]:
                print(f"      {r['description']}")
        print("\nOpen an area with:  python ask.py topics <number>")
        return
    if arg not in rows:
        sys.exit(f"Unknown topic {arg!r}; run 'python ask.py topics' for the list.")
    r = rows[arg]
    if r["level"] == 1:
        print(f"{arg}. {r['name']}  ({r['size']} papers)\n   {r['description']}\n")
        subs = sorted((t for t in rows if rows[t]["parent"] == arg), key=lambda t: int(t.split(".")[1]))
        for tid in subs:
            s = rows[tid]
            members = _members(con, tid)
            spark, _ = _trend(members, total)
            print(f"{tid:>6}  {s['name']:<52}{s['size']:>4}  {spark}")
            if s["description"]:
                print(f"        {s['description']}")
            print(f"        authors: {_authors(members, 3)}")
        print(f"\nTrend periods: {periods}.  Open a subtopic with:  python ask.py topics <a.b>")
        return
    members = sorted(_members(con, arg), key=lambda m: -m["score"])
    secondary = [m for m in _members(con, arg, primary_only=False) if m["rank"] == 2]
    spark, shares = _trend(members, total)
    parent = rows[r["parent"]]["name"]
    print(f"{arg}. {r['name']}   (in {r['parent']}. {parent})\n   {r['description']}\n")
    counts = Counter(_period(m["year"]) for m in members)
    print("   papers per period: " + "   ".join(f"{a}-{b % 100:02d}: {counts[i]}"
                                                for i, (a, b) in enumerate(PERIODS)) + f"   {spark}")
    print(f"   key terms: {', '.join(json.loads(r['terms'])[:10])}")
    print(f"   top authors: {_authors(members)}\n")
    print(f"   Most central papers ({len(members)} in total, +{len(secondary)} with this as second topic):")
    for m in members[:15]:
        print(f"     {m['year']}  {m['title'][:88]}  [{m['paper_key']}]")
    print("\nFind papers within it:  python ask.py search \"...\"   or open one:  python ask.py open <key>")


def show_rising(con, rows, total, n=10):
    subs = [t for t in rows if rows[t]["level"] == 2 and rows[t]["size"] >= 20]
    scored = []
    for tid in subs:
        members = _members(con, tid)
        c = Counter(_period(m["year"]) for m in members)
        recent = c[4] / max(total[4], 1)
        before = (c[1] + c[2] + c[3]) / max(total[1] + total[2] + total[3], 1)
        scored.append((tid, recent, before, (recent + 0.002) / (before + 0.002)))
    for title, items in (("Rising", sorted(scored, key=lambda x: -x[3])[:n]),
                         ("Fading", sorted(scored, key=lambda x: x[3])[:n])):
        print(f"\n{title} subtopics (share of papers 2020-25 vs 2005-19):")
        for tid, recent, before, ratio in items:
            print(f"  {tid:>6}  {rows[tid]['name']:<52}{before:>6.1%} -> {recent:>5.1%}  (x{ratio:.1f})")


def for_query(ix, q, n=5):
    """Which subtopics a question falls into: tally the subtopics of the best-matching passages."""
    con = ix.con
    if not has_topics(con):
        sys.exit("No topic map yet - run: python topics.py build")
    rows = _topic_rows(con)
    primary = dict(con.execute("SELECT paper_key, topic_id FROM paper_topics WHERE rank=1").fetchall())
    hits = ix.passages(q, k=40, per_paper=1, rerank=False)
    weight, examples = Counter(), defaultdict(list)
    for rank, h in enumerate(hits):
        tid = primary.get(h["paper_key"])
        if tid:
            weight[tid] += 1 / (rank + 5)
            if len(examples[tid]) < 3:
                examples[tid].append(h["paper_key"])
    tot = sum(weight.values()) or 1
    print(f"\"{q}\" falls mostly into:\n")
    for tid, w in weight.most_common(n):
        r = rows[tid]
        print(f"  {w / tot:>4.0%}  {tid:>6}  {r['name']}   (in {r['parent']}. {rows[r['parent']]['name']})")
        for k in examples[tid]:
            p = ix.papers[k]
            print(f"               e.g. {p['title'][:80]} ({p['year']})  [{k}]")
    print("\nBrowse one with:  python ask.py topics <a.b>")


def paper_topic_names(con):
    """paper key -> 'a.b Name' of its primary subtopic (empty if no topic map)."""
    if not has_topics(con):
        return {}
    rows = _topic_rows(con)
    return {k: f"{t} {rows[t]['name']}" for k, t in
            con.execute("SELECT paper_key, topic_id FROM paper_topics WHERE rank=1") if t in rows}


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "assign", "map"])
    ap.add_argument("--no-names", action="store_true", help="skip the Claude naming step")
    a = ap.parse_args()
    if a.command == "build":
        build(no_names=a.no_names)
    elif a.command == "map":
        redraw()
    else:
        assign_new()


if __name__ == "__main__":
    main()
