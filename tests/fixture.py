r"""A small synthetic JMLR index for running the tests without the real (multi-GB) data, e.g. in CI.

    python tests/fixture.py OUTDIR          builds OUTDIR/data (index) and OUTDIR/papers (PDFs + catalog)

    from tests.fixture import build
    build(data_dir, papers_dir)

Deterministic (seeded) and fast (a few seconds). It writes everything the tests read through common.py:
  data/   jmlr.db (papers, chunks + FTS, topics, paper_topics), paper_emb.npy, paper_keys.npy, chunk_emb.npy,
          chunk_ids.npy, topic_coords.json, topic_meta.json, topic_names.json, sessions/
  papers/ catalog.csv, vNN/<id>.pdf (tiny valid PDFs for the named papers only), topic_pages/

Besides ~4,200 filler papers it holds a few real-looking ones that the tests look for, e.g. v13/gretton12a
"A Kernel Two-Sample Test" (subtopic 2.6) and papers that satisfy 4, 3 and 2 of the six terms of the
privacy query in tests/test_boolq.py (so no paper satisfies all six). Filler text is drawn from a
vocabulary that avoids every term those Boolean queries look for.

When the environment variable JMLR_TEST_FIXTURE=1 is set, conftest.py (pytest) and the script-style tests
call use_fixture() BEFORE importing common, which builds the fixture once into a temp folder and points
JMLR_RAG_DATA / JMLR_PAPERS_DIR at it.
"""
import csv
import json
import os
import pathlib
import sqlite3
import sys
import tempfile

import numpy as np

SEED = 0
N_FILLER = 4200
DIM = 384
FIRST_YEAR, LAST_YEAR = 2000, 2025

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers(
    key TEXT PRIMARY KEY, volume INTEGER, pid TEXT, title TEXT, authors TEXT, citation TEXT, year INTEGER,
    abstract TEXT, pdf_path TEXT, pdf_url TEXT, abs_url TEXT, n_pages INTEGER, n_chunks INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY, paper_key TEXT NOT NULL, seq INTEGER, page_start INTEGER, page_end INTEGER,
    is_ref INTEGER DEFAULT 0, text TEXT, section TEXT);
CREATE INDEX IF NOT EXISTS chunks_paper ON chunks(paper_key);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, authors, abstract, content='papers', content_rowid='rowid', tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS topics(id TEXT PRIMARY KEY, parent TEXT, level INTEGER, name TEXT, description TEXT,
                                  size INTEGER, terms TEXT, centroid BLOB);
CREATE TABLE IF NOT EXISTS paper_topics(paper_key TEXT, topic_id TEXT, rank INTEGER, score REAL,
                                        PRIMARY KEY(paper_key, rank));
"""
# (kept identical to common.SCHEMA + topics.TOPIC_SCHEMA; not imported, so building never needs common's env)

# ---------------------------------------------------------------- topics

AREAS = [
    ("Bayesian Inference and Probabilistic Modeling",
     ["Variational Inference", "Gaussian Process Models", "Markov Chain Monte Carlo", "Bayesian Nonparametrics"]),
    ("High-Dimensional Statistical Inference and Graphical Models",
     ["Gaussian Graphical Model Estimation", "Covariance Estimation", "Multiple Testing and False Discovery",
      "Structure Learning of Undirected Models", "Conformal and Selective Inference",
      "Kernel-based Two-Sample and Independence Testing"]),
    ("Low-Rank and Sparse Matrix Methods",
     ["Matrix Completion", "Sparse Penalized Linear Regression", "Robust Principal Component Analysis",
      "Tensor Decomposition"]),
    ("Neural Network Theory and Training",
     ["Approximation Theory of Deep Networks", "Training Dynamics and Overparameterization",
      "Generalization of Deep Networks"]),
    ("Classification and Support Vector Machines",
     ["Support Vector Machine Solvers", "Multiclass and Multilabel Classification", "Imbalanced Classification"]),
    ("Supervised and Semi-Supervised Learning",
     ["Semi-Supervised Learning", "Active Learning", "Learning with Noisy Labels"]),
    ("Online and Distributed Optimization",
     ["Online Convex Optimization", "Decentralized Optimization", "Communication-Efficient Training"]),
    ("Reinforcement Learning Theory and Methods",
     ["Policy Gradient Methods", "Offline Reinforcement Learning", "Exploration in Markov Decision Processes"]),
    ("Stochastic Gradient and Convex Optimization",
     ["Stochastic Gradient Descent Analysis", "Variance Reduction", "Proximal and Coordinate Methods"]),
    ("Ranking and Tree-Based Learning",
     ["Learning to Rank", "Random Forests", "Boosting"]),
    ("Graph Clustering and Network Analysis",
     ["Spectral Clustering", "Community Detection", "Graph Neural Networks"]),
    ("Kernel Methods and Reproducing Kernel Hilbert Spaces",
     ["Kernel Ridge Regression", "Random Features", "Operator-Valued Kernels"]),
    ("Statistical Learning Theory and Risk Bounds",
     ["PAC-Bayes Bounds", "Rademacher Complexity", "Minimax Rates"]),
    ("Causal Discovery and Inference",
     ["Causal Structure Learning", "Treatment Effect Estimation", "Instrumental Variables"]),
    ("Natural Language Processing",
     ["Topic Models", "Word Embeddings", "Sequence Labelling"]),
    ("Bandit Regret Analysis",
     ["Stochastic Multi-Armed Bandits", "Contextual Bandits", "Adversarial Bandits"]),
    ("Manifold Learning and Geometric Embeddings",
     ["Nonlinear Dimensionality Reduction", "Metric Learning", "Optimal Transport"]),
    ("Open-Source Machine Learning Software",
     ["Machine Learning Toolboxes", "Deep Learning Libraries", "Probabilistic Programming Software"]),
]

# ---------------------------------------------------------------- the named papers the tests look for
# (key, title, authors, subtopic, abstract, [(page_start, page_end, is_ref, section, text)])

GRETTON_CHUNKS = [
    (1, 1, 0, "Abstract", "We propose a framework for analyzing and comparing distributions, which we use to construct "
     "statistical tests to determine if two samples are drawn from different distributions. Our test statistic is "
     "the largest difference in expectations over functions in the unit ball of a reproducing kernel Hilbert space."),
    (1, 2, 0, "1 Introduction", "We address the problem of comparing samples from two probability distributions, by "
     "proposing statistical tests of the null hypothesis that these distributions are equal against the alternative "
     "that they are different. We call this the kernel two-sample test."),
    (3, 3, 0, "2 The Maximum Mean Discrepancy", "In this section we present the maximum mean discrepancy (MMD) and "
     "describe conditions under which it is a metric on the space of probability distributions."),
    (3, 4, 0, "2 The Maximum Mean Discrepancy / 2.1 Deﬁnition of the Maximum Mean Discrepancy", "Let p and q be "
     "Borel probability measures. The maximum mean discrepancy is the supremum over a function class of the "
     "difference of expectations under p and q."),
    (4, 5, 0, "2 The Maximum Mean Discrepancy / 2.2 The MMD in Reproducing Kernel Hilbert Spaces", "When the "
     "function class is the unit ball of a characteristic kernel space, the MMD is the distance between mean "
     "embeddings. The median heuristic sets the kernel bandwidth to the median distance between points."),
    (9, 10, 0, "3 Background Material / 3.1 Statistical Hypothesis Testing", "A statistical test distinguishes the "
     "null hypothesis from the alternative; the level bounds the Type I error and the power is one minus the "
     "Type II error."),
    (16, 17, 0, "6 A Linear Time Statistic and Test", "A linear time estimate of the squared MMD uses pairs of "
     "observations and has an asymptotically normal distribution, giving a cheap kernel test for streaming data."),
    (25, 26, 0, "8 Experiments", "We compare the kernel test with classical two-sample tests on toy data, data "
     "integration, attribute matching and graph data; the bandwidth is chosen by the median heuristic."),
    (32, 33, 0, "9 Conclusion", "The MMD gives consistent kernel two-sample tests with good power in high dimension."),
    (44, 45, 1, "References", "References Y. Altun and A. Smola. Unifying divergence minimization and statistical "
     "inference via convex duality. In Proceedings of the Annual Conference on Computational Learning Theory, "
     "2006. P. Anderson, P. Hall and D. Titterington. Journal of Multivariate Analysis, 50:41-54, 1994."),
]

NAMED = [
    ("v13/gretton12a", "A Kernel Two-Sample Test",
     "Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, Alexander Smola", "2.6",
     "We propose a framework for analyzing and comparing distributions, which we use to construct statistical "
     "tests to determine if two samples are drawn from different distributions. Our test statistic is the largest "
     "difference in expectations over functions in the unit ball of a reproducing kernel Hilbert space (RKHS), "
     "and is called the maximum mean discrepancy (MMD). We present two distribution-free kernel two-sample tests "
     "based on large deviation bounds for the MMD, and a third test based on its asymptotic distribution.",
     GRETTON_CHUNKS, 51),
    ("v24/22-0137", "Aggregated Kernel Two-Sample Tests with Adaptive Bandwidths",
     "Antonin Schrab, Ilmun Kim, Benjamin Guedj, Arthur Gretton", "2.6",
     "We propose aggregated kernel two-sample tests which combine several bandwidths of the maximum mean "
     "discrepancy test statistic, avoiding the median heuristic and data splitting.",
     [(1, 1, 0, "Abstract", "We propose aggregated kernel two-sample tests combining many kernel bandwidths."),
      (2, 3, 0, "1 Introduction", "The aggregated test controls the level while adapting to the unknown smoothness.")],
     38),
    ("v25/23-0512", "Privacy-Preserving Federated Learning for Multimodal Medical Imaging",
     "Maria Lindqvist, Tomas Ferreira", "7.2",
     "We study privacy-preserving training of multimodal medical imaging networks across hospitals.",
     [(1, 1, 0, "Abstract", "A privacy-preserving federated protocol for multimodal medical imaging."),
      (2, 3, 0, "1 Introduction", "Hospitals hold multimodal medical scans that cannot leave the site; federated "
       "training with differential privacy keeps them private while preserving accuracy.")], 24),
    ("v24/22-1101", "Federated Prototypes for Knowledge Sharing Across Clients",
     "Wen Zhao, Priya Natarajan", "7.3",
     "Clients exchange class prototypes instead of weights, enabling knowledge sharing in federated learning.",
     [(1, 1, 0, "Abstract", "Federated learning with prototypes for knowledge sharing between heterogeneous clients.")],
     19),
    ("v22/20-0899", "Learning Decision Rules for Medical Triage",
     "Helena Okafor", "10.3",
     "We learn short decision rules from medical triage records.",
     [(1, 1, 0, "Abstract", "Interpretable decision rules learned from medical triage records.")], 15),
]

# ---------------------------------------------------------------- filler text

# Generic ML vocabulary. It must NOT contain any term the Boolean-query tests look for (privacy, preserving,
# knowledge, sharing, multimodal, medical, federated, prototype(s), rule(s), aggregated, Gretton) nor the
# phrase "two sample".
ADJ = ["adaptive", "robust", "scalable", "sparse", "stochastic", "nonparametric", "efficient", "convex", "deep",
       "spectral", "random", "regularized", "hierarchical", "latent", "online", "distributed", "structured",
       "approximate", "consistent", "minimax", "linear", "nonlinear", "Bayesian", "variational", "kernel"]
NOUN = ["estimation", "regression", "classification", "clustering", "optimization", "inference", "embeddings",
        "networks", "models", "bandits", "policies", "graphs", "features", "representations", "matrices",
        "processes", "bounds", "rankings", "forests", "manifolds", "kernels", "gradients", "samplers", "trees"]
CONTEXT = ["for high-dimensional data", "with missing values", "under covariate shift", "in reproducing kernel spaces",
           "with theoretical guarantees", "at scale", "on graphs", "for time series", "with partial feedback",
           "via convex relaxation", "for sequential decisions", "with heavy tails", "in the streaming setting"]
FIRST = ["Anna", "Ben", "Carla", "David", "Elena", "Felix", "Grace", "Hugo", "Irene", "Jonas", "Kai", "Laura", "Marco",
         "Nina", "Oscar", "Paula", "Quentin", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wei", "Xin", "Yara", "Zoe"]
LAST = ["Abel", "Brandt", "Costa", "Dubois", "Eriksen", "Fischer", "Garcia", "Hansen", "Ito", "Jensen", "Kowalski",
        "Larsen", "Moreau", "Nakamura", "Olsen", "Petrov", "Quinn", "Rossi", "Silva", "Tanaka", "Ueda", "Vargas",
        "Weber", "Xu", "Yilmaz", "Zhang"]
SENT = ["We study {a} {n} {c}.", "Our method combines {a} {n} with {a2} {n2}.",
        "We prove that the estimator is consistent and attains near-optimal rates {c}.",
        "Experiments on benchmark datasets show that {a} {n} outperform existing baselines.",
        "The analysis relies on concentration inequalities for {a2} {n2}.",
        "We give an algorithm whose running time is linear in the number of observations.",
        "A lower bound shows that the dependence on the dimension cannot be improved.",
        "The approach extends to {a2} {n2} {c}."]
SECTIONS = ["1 Introduction", "2 Related Work", "3 Method", "3 Method / 3.1 Algorithm", "4 Theory",
            "5 Experiments", "6 Conclusion"]


def _sentence(rng):
    return rng.choice(SENT).format(a=rng.choice(ADJ), n=rng.choice(NOUN), a2=rng.choice(ADJ), n2=rng.choice(NOUN),
                                   c=rng.choice(CONTEXT))


def _text(rng, n):
    return " ".join(_sentence(rng) for _ in range(n))


def _title(rng):
    t = f"{rng.choice(ADJ)} {rng.choice(NOUN)} {rng.choice(CONTEXT)}"
    return t[0].upper() + t[1:]


def _authors(rng):
    return ", ".join(f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(int(rng.integers(1, 5))))


def _volume(year):
    return year - 1999          # JMLR: volume 1 = 2000 ... volume 26 = 2025


def _tiny_pdf(title):
    """A valid one-page PDF (correct xref offsets) showing the paper title."""
    text = title.encode("latin-1", "replace").replace(b"(", b"[").replace(b")", b"]")
    stream = b"BT /F1 14 Tf 72 720 Td (" + text + b") Tj ET"
    objs = [b"<</Type/Catalog/Pages 2 0 R>>", b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
            b"<</Length %d>>stream\n" % len(stream) + stream + b"\nendstream",
            b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj" % i + o + b"endobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


# ---------------------------------------------------------------- build

def build(data_dir, papers_dir, n_filler=N_FILLER, seed=SEED):
    """Write the synthetic dataset into data_dir and papers_dir (created; an existing jmlr.db is replaced)."""
    data_dir, papers_dir = pathlib.Path(data_dir), pathlib.Path(papers_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    papers_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sessions").mkdir(exist_ok=True)
    (papers_dir / "topic_pages").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)

    # topics: ids, per-subtopic popularity trend (so "rising" has winners and losers)
    subs = []                                    # (sub id, area id, name)
    for a, (_, names) in enumerate(AREAS, 1):
        subs += [(f"{a}.{s}", str(a), n) for s, n in enumerate(names, 1)]
    sub_ids = [s for s, _, _ in subs]
    trend = rng.normal(0, 1.2, len(subs))

    # filler papers: year distribution grows over time; subtopic drawn with a time-dependent preference
    years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
    year_w = np.linspace(1, 4, len(years))
    papers = []                                  # dicts
    for i in range(n_filler):
        year = int(rng.choice(years, p=year_w / year_w.sum()))
        t = (year - FIRST_YEAR) / (LAST_YEAR - FIRST_YEAR) - 0.5
        w = np.exp(trend * t)
        sub = sub_ids[int(rng.choice(len(subs), p=w / w.sum()))]
        vol = _volume(year)
        pid = f"{year % 100:02d}-{i:04d}"
        papers.append(dict(key=f"v{vol:02d}/{pid}", volume=vol, pid=pid, title=_title(rng), authors=_authors(rng),
                           year=year, abstract=_text(rng, 4), sub=sub, n_pages=int(rng.integers(10, 50)),
                           chunks=None, named=False))
    # named papers go first (stable dict order for the tests' fake indexes)
    named = []
    for key, title, authors, sub, abstract, chunks, n_pages in NAMED:
        vol, pid = int(key[1:3]), key.split("/")[1]
        named.append(dict(key=key, volume=vol, pid=pid, title=title, authors=authors, year=1999 + vol,
                          abstract=abstract, sub=sub, n_pages=n_pages, chunks=chunks, named=True))
    papers = named + papers

    # chunks
    chunk_rows = []
    for p in papers:
        if p["chunks"] is None:
            secs = [SECTIONS[0], SECTIONS[int(rng.integers(1, 5))], SECTIONS[5]]
            pages = sorted(int(x) for x in rng.integers(1, p["n_pages"], 3))
            p["chunks"] = [(1, 1, 0, "Abstract", p["abstract"])]
            p["chunks"] += [(pg, pg + 1, 0, s, _text(rng, 3)) for pg, s in zip(pages, secs)]
            p["chunks"].append((p["n_pages"] - 1, p["n_pages"], 1, "References",
                                "References A. Author. A study. Journal of Statistics, 12(3):45-67, 2004."))
        for seq, (ps, pe, ref, sec, text) in enumerate(p["chunks"]):
            chunk_rows.append((p["key"], seq, ps, pe, ref, text, sec))

    db = data_dir / "jmlr.db"
    for suffix in ("", "-wal", "-shm"):
        pathlib.Path(str(db) + suffix).unlink(missing_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO papers(key, volume, pid, title, authors, citation, year, abstract, pdf_path, pdf_url, abs_url, "
        "n_pages, n_chunks, status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?, 'done')",
        [(p["key"], p["volume"], p["pid"], p["title"], p["authors"], f"({p['volume']}):1-{p['n_pages']}, {p['year']}.",
          p["year"], p["abstract"], f"{p['key']}.pdf",
          f"https://www.jmlr.org/papers/volume{p['volume']}/{p['pid']}/{p['pid']}.pdf",
          f"https://www.jmlr.org/papers/v{p['volume']}/{p['pid']}.html", p["n_pages"], len(p["chunks"]))
         for p in papers])
    con.executemany("INSERT INTO chunks(paper_key, seq, page_start, page_end, is_ref, text, section) "
                    "VALUES(?,?,?,?,?,?,?)", chunk_rows)
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    con.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")

    # embeddings: a direction per area and per subtopic plus noise, so neighbours share topics
    area_dir = rng.normal(0, 1, (len(AREAS) + 1, DIM))
    sub_dir = {s: area_dir[int(a)] + 0.6 * rng.normal(0, 1, DIM) for s, a, _ in subs}
    norm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
    pemb = norm(np.stack([sub_dir[p["sub"]] + 0.8 * rng.normal(0, 1, DIM) for p in papers]))
    keys = [p["key"] for p in papers]
    np.save(data_dir / "paper_emb.npy", pemb.astype(np.float16))
    np.save(data_dir / "paper_keys.npy", np.array(keys))
    ids = [r[0] for r in con.execute("SELECT id FROM chunks WHERE is_ref=0 ORDER BY id")]
    row_of = {k: i for i, k in enumerate(keys)}
    owner = [row_of[k] for (k,) in con.execute("SELECT paper_key FROM chunks WHERE is_ref=0 ORDER BY id")]
    cemb = norm(pemb[owner] + 0.1 * rng.normal(0, 1, (len(ids), DIM)))
    np.save(data_dir / "chunk_emb.npy", cemb.astype(np.float16))
    np.save(data_dir / "chunk_ids.npy", np.array(ids, dtype=np.int64))

    # topics + assignments (primary = own subtopic; a secondary one for some papers)
    members = {s: [i for i, p in enumerate(papers) if p["sub"] == s] for s in sub_ids}
    topic_rows, names = [], {}
    for a, (aname, snames) in enumerate(AREAS, 1):
        area_members = [i for s, pa, _ in subs if pa == str(a) for i in members[s]]
        cen = norm(pemb[area_members].mean(0))
        desc = f"Research on {aname.lower()}."
        topic_rows.append((str(a), None, 1, aname, desc, len(area_members),
                           json.dumps(aname.lower().split()[:6]), cen.astype(np.float32).tobytes()))
        names[str(a)] = {"name": aname, "description": desc}
        for s, n in enumerate(snames, 1):
            tid = f"{a}.{s}"
            cen = norm(pemb[members[tid]].mean(0))
            desc = f"Papers on {n.lower()}."
            topic_rows.append((tid, str(a), 2, n, desc, len(members[tid]), json.dumps(n.lower().split()),
                               cen.astype(np.float32).tobytes()))
            names[tid] = {"name": n, "description": desc}
    centroids = {r[0]: np.frombuffer(r[7], dtype=np.float32) for r in topic_rows if r[2] == 2}
    assignments = []
    for i, p in enumerate(papers):
        own = float(pemb[i] @ centroids[p["sub"]])
        assignments.append((p["key"], p["sub"], 1, own))
        if i % 5 == 0:
            other = sub_ids[(sub_ids.index(p["sub"]) + 1) % len(sub_ids)]
            assignments.append((p["key"], other, 2, float(pemb[i] @ centroids[other])))
    con.executemany("INSERT INTO topics VALUES(?,?,?,?,?,?,?,?)", topic_rows)
    con.executemany("INSERT INTO paper_topics VALUES(?,?,?,?)", assignments)
    con.commit()
    con.close()

    # 2D map: areas on a circle, subtopics around their area, papers around their subtopic (normalised)
    centre = {}
    for a in range(1, len(AREAS) + 1):
        ang = 2 * np.pi * a / len(AREAS)
        centre[str(a)] = np.array([np.cos(ang), np.sin(ang)]) * 6
    xy = np.zeros((len(papers), 2))
    for s, a, _ in subs:
        c = centre[a] + rng.normal(0, 1.0, 2)
        idx = members[s]
        xy[idx] = c + rng.normal(0, 0.45, (len(idx), 2))
    xy = (xy - xy.mean(0)) / xy.std()
    coords = {k: [round(float(x), 4), round(float(y), 4)] for k, (x, y) in zip(keys, xy)}
    (data_dir / "topic_coords.json").write_text(json.dumps(
        {"version": 1, "representation": "combined", "created": "fixture", "coords": coords,
         "topic": {p["key"]: p["sub"] for p in papers}}, separators=(",", ":")), encoding="utf-8")
    (data_dir / "topic_meta.json").write_text(json.dumps(
        {"representation": "combined", "method": "fixture", "areas": len(AREAS), "papers_per_subtopic": 60}),
        encoding="utf-8")
    (data_dir / "topic_names.json").write_text(json.dumps(names, indent=1, ensure_ascii=False), encoding="utf-8")

    # papers folder: catalog + tiny PDFs for the named papers
    with open(papers_dir / "catalog.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["volume", "id", "title", "authors", "citation", "pdf_url", "abs_url", "bib_url", "local_pdf"])
        for p in papers:
            w.writerow([p["volume"], p["pid"], p["title"], p["authors"], f"({p['volume']}):1-{p['n_pages']}, {p['year']}.",
                        f"https://www.jmlr.org/papers/volume{p['volume']}/{p['pid']}/{p['pid']}.pdf",
                        f"https://www.jmlr.org/papers/v{p['volume']}/{p['pid']}.html", "", f"{p['key']}.pdf"])
    for p in papers:
        if p["named"]:
            pdf = papers_dir / f"{p['key']}.pdf"
            pdf.parent.mkdir(parents=True, exist_ok=True)
            pdf.write_bytes(_tiny_pdf(p["title"]))
    return data_dir, papers_dir


def _cleanup(root):
    """Remove the temp fixture at exit. On Windows open SQLite handles lock the files, so close them first."""
    import gc
    import shutil
    for obj in gc.get_objects():
        if isinstance(obj, sqlite3.Connection):
            try:
                obj.close()
            except Exception:
                pass
    shutil.rmtree(root, ignore_errors=True)


def use_fixture():
    """If JMLR_TEST_FIXTURE=1: build the fixture once per process into a temp folder and point JMLR_RAG_DATA /
    JMLR_PAPERS_DIR at it. Must run before `common` is imported (it reads the variables at import time)."""
    if os.environ.get("JMLR_TEST_FIXTURE") != "1" or os.environ.get("_JMLR_FIXTURE_BUILT"):
        return False
    if "common" in sys.modules:
        raise RuntimeError("tests/fixture.use_fixture() must run before common is imported")
    import atexit
    root = pathlib.Path(tempfile.mkdtemp(prefix="jmlr_fixture_"))
    atexit.register(_cleanup, root)
    data, papers = build(root / "data", root / "papers")
    os.environ["JMLR_RAG_DATA"] = str(data)
    os.environ["JMLR_PAPERS_DIR"] = str(papers)
    os.environ["JMLR_RAG_ENV_FILE"] = str(root / ".env")      # never read a developer's real .env
    os.environ["_JMLR_FIXTURE_BUILT"] = str(root)
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tests/fixture.py OUTDIR")
    import time
    t0 = time.time()
    out = pathlib.Path(sys.argv[1])
    build(out / "data", out / "papers")
    print(f"Fixture written to {out / 'data'} and {out / 'papers'} in {time.time() - t0:.1f}s")
