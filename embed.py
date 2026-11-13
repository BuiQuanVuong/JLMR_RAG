"""Step 2: compute semantic embeddings for papers (title + abstract) and passages.

Resumable and incremental: passages already embedded are skipped, progress is saved every
block, so you can stop it (Ctrl+C) and re-run later.  Usage:  python embed.py
"""
import time

import numpy as np

from common import (CHUNK_EMB, CHUNK_IDS, PAPER_EMB, PAPER_KEYS, connect, load_encoder,
                    utf8_console)

BLOCK = 8192


def save(ids, embs):
    # write to temp files then swap, so an interrupted save never corrupts the index
    for path, arr in ((CHUNK_IDS, ids), (CHUNK_EMB, embs)):
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(path)


def main():
    utf8_console()
    con = connect()
    model, device = load_encoder()
    print(f"Encoder on {device}")

    # papers: small, always recomputed
    rows = con.execute("SELECT key, title, abstract FROM papers WHERE status='done' ORDER BY key").fetchall()
    t0 = time.time()
    pe = model.encode([f"{r['title']}. {r['abstract'] or ''}" for r in rows], batch_size=64,
                      normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    np.save(PAPER_EMB, pe.astype(np.float16))
    np.save(PAPER_KEYS, np.array([r["key"] for r in rows]))
    print(f"Embedded {len(rows)} papers in {time.time() - t0:.0f}s")

    # passages: incremental
    if CHUNK_IDS.exists():
        ids, embs = np.load(CHUNK_IDS), np.load(CHUNK_EMB)
    else:
        ids, embs = np.zeros(0, dtype=np.int64), np.zeros((0, pe.shape[1]), dtype=np.float16)
    have = set(ids.tolist())
    todo = [r for r in con.execute("""SELECT c.id, c.text, p.title FROM chunks c
                                      JOIN papers p ON p.key = c.paper_key
                                      WHERE c.is_ref = 0 ORDER BY c.id""")
            if r["id"] not in have]
    print(f"{len(have)} passages already embedded, {len(todo)} to go")
    t0 = time.time()
    for b in range(0, len(todo), BLOCK):
        block = todo[b: b + BLOCK]
        e = model.encode([f"{r['title']}: {r['text']}" for r in block], batch_size=64,
                         normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        ids = np.concatenate([ids, np.array([r["id"] for r in block], dtype=np.int64)])
        embs = np.concatenate([embs, e.astype(np.float16)])
        save(ids, embs)
        done = b + len(block)
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(todo)} passages, {rate:.0f}/s, ~{(len(todo) - done) / rate / 60:.0f} min left",
              flush=True)
    print(f"Done: {len(ids)} passages embedded")


if __name__ == "__main__":
    main()
