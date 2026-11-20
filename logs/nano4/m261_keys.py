#!/usr/bin/env python
"""M261 helper (2026-11-20): the game keys of the rows of results/m260/rows.npz (same loader, same order) -> results/m260/rows_key.npy.

  python -I logs/nano4/m261_keys.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402


def main():
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    base.FULL, base.EXT = True, None; G, _ = base.load_games(False, log); key = np.asarray(G["key"]).astype("S"); q = np.load(base.CACHE / "queries.npz"); pidx = {p: i for i, p in enumerate(q["players"].astype(str))}
    z = np.load(ROOT / "results/m260/rows.npz"); owner = np.array([pidx[p] for p in G["pid"]]); assert len(key) == int(z["n_rows"]) and np.array_equal(owner, z["owner"]), "the rows are not in the order of rows.npz"
    np.save(ROOT / "results/m260/rows_key.npy", key); log(f"{len(key)} keys ({key.dtype}), unique {len(np.unique(key))}; first {key[0]!r}; written results/m260/rows_key.npy")


if __name__ == "__main__":
    main()
