"""Data behind the interactive topic map (gui/topicmap.py): 2D coordinates, papers, topics, statistics.

No search index is needed: only the SQLite DB and the .npy embedding files.

The 2D map (t-SNE of the paper vectors, exactly as topics.draw_map: PCA 50 then t-SNE, perplexity 40,
random_state 0) is computed once (about a minute on CPU) and cached in data/topic_coords.json:
    {"version": 1, "representation": "combined", "coords": {key: [x, y]}, "topic": {key: "3.2"}}
Papers added later (topics.py assign) are placed at the mean position of their nearest cached neighbours,
so the cache never needs a full recompute after an update.
"""
import json
import threading
import time
from collections import Counter

import numpy as np

import topics
from common import DATA_DIR, PAPER_EMB, PAPER_KEYS, connect

COORDS_FILE = DATA_DIR / "topic_coords.json"
CACHE_VERSION = 1
RECENT, BEFORE = (2020, 2025), (2005, 2019)     # periods of the "rising subtopics" view (as topics.show_rising)

# 18 clearly separable categorical colours (tab20-like, greys removed), fixed order by area number.
PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f",
           "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#ff7f0e"]


def area_color(area_id):
    return PALETTE[(int(area_id) - 1) % len(PALETTE)]


def tid_sort_key(tid):
    return [int(p) for p in tid.split(".")]


# ---------------------------------------------------------------- coordinates

def _representation():
    try:
        return json.loads(topics.META_FILE.read_text(encoding="utf-8"))["representation"]
    except Exception:
        return "combined"


def compute_coords(progress=None):
    """Full t-SNE of all papers (blocking, CPU, ~20-60 s). Returns {key: [x, y]}."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    say = progress or (lambda msg: None)
    con = connect()
    say("loading paper vectors")
    keys, reps = topics.representations(con)
    X = reps[_representation()]
    say("PCA to 50 dimensions")
    P = PCA(n_components=50, random_state=0).fit_transform(X)
    say(f"t-SNE of {len(keys)} papers")
    Y = TSNE(n_components=2, init="pca", random_state=0, perplexity=40).fit_transform(P)
    Y = (Y - Y.mean(0)) / Y.std()          # normalised: comparable scale whatever the run
    return {k: [round(float(x), 4), round(float(y), 4)] for k, (x, y) in zip(keys, Y)}


def place_new(coords, new_keys, k=8):
    """Positions for papers missing from the cache: mean of their k nearest cached neighbours (abstract vectors)."""
    keys = [str(x) for x in np.load(PAPER_KEYS)]
    emb = np.load(PAPER_EMB).astype(np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    row = {key: i for i, key in enumerate(keys)}
    known = [key for key in coords if key in row]
    if not known:
        return {}
    K = emb[[row[key] for key in known]]
    C = np.array([coords[key] for key in known])
    rng = np.random.default_rng(0)
    out = {}
    for key in new_keys:
        if key not in row:
            continue
        sims = K @ emb[row[key]]
        nn = np.argsort(-sims)[:k]
        x, y = C[nn].mean(0) + rng.normal(0, 0.02, 2)
        out[key] = [round(float(x), 4), round(float(y), 4)]
    return out


def read_cache():
    try:
        data = json.loads(COORDS_FILE.read_text(encoding="utf-8"))
        return data if data.get("version") == CACHE_VERSION and data.get("coords") else None
    except Exception:
        return None


def write_cache(coords, primary):
    data = {"version": CACHE_VERSION, "representation": _representation(), "created": time.strftime("%Y-%m-%d %H:%M"),
            "coords": coords, "topic": {k: primary.get(k, "") for k in coords}}
    tmp = COORDS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    tmp.replace(COORDS_FILE)


def ensure_coords(progress=None):
    """{key: [x, y]} for every paper with a topic; computes / extends the cache when needed (blocking)."""
    con = connect()
    primary = dict(con.execute("SELECT paper_key, topic_id FROM paper_topics WHERE rank=1").fetchall())
    cache = read_cache()
    if cache is None:
        coords = compute_coords(progress)
        write_cache(coords, primary)
        return coords
    coords = cache["coords"]
    missing = [k for k in primary if k not in coords]
    if missing:
        coords.update(place_new(coords, missing))
        write_cache(coords, primary)
    return coords


# ---------------------------------------------------------------- the map data (loaded once per process)

class MapData:
    """Everything the page shows, loaded once: papers (index i), topics, member lists, statistics."""

    def __init__(self, coords):
        con = connect()
        self.rows = topics._topic_rows(con)
        self.areas = sorted((t for t, r in self.rows.items() if r["level"] == 1), key=int)
        self.subs = {a: sorted((t for t, r in self.rows.items() if r["parent"] == a), key=tid_sort_key)
                     for a in self.areas}
        primary = dict(con.execute("SELECT paper_key, topic_id FROM paper_topics WHERE rank=1").fetchall())
        self.score = dict(con.execute("SELECT paper_key, score FROM paper_topics WHERE rank=1").fetchall())
        secondary = dict(con.execute("SELECT paper_key, topic_id FROM paper_topics WHERE rank=2").fetchall())
        papers = {r["key"]: dict(r) for r in con.execute("SELECT key, title, authors, year FROM papers")}
        self.keys = sorted((k for k in primary if k in coords and k in papers and primary[k] in self.rows),
                           key=lambda k: (primary[k].split(".")[0], k))
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.xy = np.array([coords[k] for k in self.keys], dtype=float)
        self.title = [papers[k]["title"] or k for k in self.keys]
        self.authors = [papers[k]["authors"] or "" for k in self.keys]
        self.year = np.array([papers[k]["year"] or 0 for k in self.keys])
        self.sub = [primary[k] for k in self.keys]
        self.area = np.array([s.split(".")[0] for s in self.sub])
        self.sub_arr = np.array(self.sub)
        self.secondary = secondary
        self.years = (int(self.year[self.year > 0].min()), int(self.year.max()))
        self.total_by_year = Counter(int(y) for y in self.year)
        self.size_primary = Counter(self.sub) + Counter(self.area.tolist())

    # -- membership
    def members(self, tid):
        """Indices of papers whose primary topic is tid (an area '3' or a subtopic '3.2')."""
        return np.flatnonzero(self.sub_arr == tid) if "." in tid else np.flatnonzero(self.area == tid)

    def in_years(self, lo, hi):
        return (self.year >= lo) & (self.year <= hi)

    # -- statistics
    def per_year(self, tid):
        """(years, counts, share %) for a topic, over the whole corpus range."""
        c = Counter(int(self.year[i]) for i in self.members(tid))
        ys = list(range(self.years[0], self.years[1] + 1))
        return ys, [c[y] for y in ys], [round(100 * c[y] / max(self.total_by_year[y], 1), 2) for y in ys]

    def period_shares(self, tid):
        """Share of each topics.PERIODS period's papers that belong to tid (the trend line of `ask.py topics`)."""
        c, tot = Counter(), Counter()
        for y in self.year:
            tot[topics._period(int(y))] += 1
        for i in self.members(tid):
            c[topics._period(int(self.year[i]))] += 1
        return [c[p] / max(tot[p], 1) for p in range(len(topics.PERIODS))]

    def top_authors(self, tid, n=8):
        c = Counter(a.strip() for i in self.members(tid) for a in self.authors[i].split(",") if a.strip())
        return c.most_common(n)

    def central(self, tid, n=10):
        """Most central papers of a topic (highest assignment score)."""
        idx = self.members(tid)
        return sorted(idx, key=lambda i: -(self.score.get(self.keys[i]) or 0))[:n]

    def rising(self, n=8, min_size=20):
        """(rising, fading) lists of (tid, share before, share recent, ratio), as topics.show_rising."""
        def share(m, lo, hi):
            return m.sum() / max(((self.year >= lo) & (self.year <= hi)).sum(), 1)
        recent_m, before_m = self.in_years(*RECENT), self.in_years(*BEFORE)
        scored = []
        for a in self.areas:
            for tid in self.subs[a]:
                if self.rows[tid]["size"] < min_size:
                    continue
                mem = self.sub_arr == tid
                recent, before = share(mem & recent_m, *RECENT), share(mem & before_m, *BEFORE)
                scored.append((tid, before, recent, (recent + 0.002) / (before + 0.002)))
        return sorted(scored, key=lambda x: -x[3])[:n], sorted(scored, key=lambda x: x[3])[:n]

    def centre(self, tid):
        m = self.members(tid)
        return [float(np.median(self.xy[m, 0])), float(np.median(self.xy[m, 1]))] if len(m) else [0.0, 0.0]

    def anchor(self, tid, radius=0.25):
        """A label position for a topic: its member with the most same-topic neighbours within `radius`
        (areas are not always one blob on a t-SNE map, so the median can fall between them)."""
        m = self.members(tid)
        if not len(m):
            return [0.0, 0.0]
        P = self.xy[m]
        if len(P) > 600:
            P = P[np.random.default_rng(0).choice(len(P), 600, replace=False)]
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        dens = (d2 < radius ** 2).sum(1)
        top = int(dens.argmax())
        best = np.flatnonzero((dens >= dens.max() * 0.8) & (d2[top] < (2 * radius) ** 2))
        c = P[best].mean(0)                       # centre of the densest points around the peak
        return [float(c[0]), float(c[1])]

    def label_positions(self, texts, width_px=680, height_px=580, char_px=6.6, line_px=22, rounds=300):
        """Area label anchors nudged apart so the boxes do not overlap (approximate pixel sizes)."""
        lo, hi = self.bounds()
        ux, uy = (hi[0] - lo[0]) / width_px, (hi[1] - lo[1]) / height_px
        pos = np.array([self.anchor(a) for a in self.areas])
        w = np.array([(len(t) * char_px + 16) * ux for t in texts])
        h = line_px * uy
        for _ in range(rounds):
            moved = False
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    dx, dy = pos[j] - pos[i]
                    ox = (w[i] + w[j]) / 2 - abs(dx)
                    oy = h - abs(dy)
                    if ox <= 0 or oy <= 0:
                        continue
                    moved = True
                    # push apart along the cheaper axis (vertical moves are usually shorter)
                    if oy / h <= ox / ((w[i] + w[j]) / 2):
                        s = (oy / 2 + 1e-4) * (1 if dy >= 0 else -1)
                        pos[i, 1] -= s
                        pos[j, 1] += s
                    else:
                        s = (ox / 2 + 1e-4) * (1 if dx >= 0 else -1)
                        pos[i, 0] -= s
                        pos[j, 0] += s
            # keep every label inside the plot
            pos[:, 0] = np.clip(pos[:, 0], lo[0] + w / 2, hi[0] - w / 2)
            pos[:, 1] = np.clip(pos[:, 1], lo[1] + h, hi[1] - h)
            if not moved:
                break
        return pos.tolist()

    def bounds(self, pad=0.04):
        lo, hi = self.xy.min(0), self.xy.max(0)
        span = hi - lo
        return (lo - pad * span).tolist(), (hi + pad * span).tolist()

    def label(self, tid):
        return f"{tid}. {self.rows[tid]['name']}"


_DATA = {"map": None, "error": None, "status": ""}
_LOCK = threading.Lock()


def cached():
    return _DATA["map"]


def load(progress=None):
    """MapData, computing the coordinates if needed (blocking; call via run.io_bound). Thread-safe, done once."""
    with _LOCK:
        if _DATA["map"] is None:
            def say(msg):
                _DATA["status"] = msg
                if progress:
                    progress(msg)
            _DATA["map"] = MapData(ensure_coords(say))
        return _DATA["map"]


def status():
    return _DATA["status"]


_CON = {}


def abstract(key):
    if "con" not in _CON:
        _CON["con"] = connect()
    r = _CON["con"].execute("SELECT abstract FROM papers WHERE key=?", (key,)).fetchone()
    return (r["abstract"] if r else "") or ""
