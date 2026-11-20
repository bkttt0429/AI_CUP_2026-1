#!/usr/bin/env python
"""Section 4M M181 (third round; copy of m161q_tta.py with the paths and the width of m161r_prod.py): orientation-averaged queries (m124_tta_queries.py, unchanged) for the networks of m161r_prod.py, and the
learned centres as a second source.

  python -I logs/nano4/m161r_tta.py --seeds 40 --name S8t
      reads results/m161r and /work/u2507115/m161r_ckpt (width 768 / feed-forward 3,072), parts in results/m161r_tta, writes
      results/m84_cache/var_S7t.npz (prototypes; m84_variant's format) and var_S7tc.npz (protos = the learned centres,
      has = the players with a centre [members, 5, C], the same query vectors) for m164_ident.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m124_tta_queries as t  # noqa: E402

base = t.base
t.OUT, t.CK, t.TMP = ROOT / "results/m161r", Path("/work/u2507115/m161r_ckpt"), ROOT / "results/m161r_tta"
base.D, base.FFN = 768, 3072

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="0"); ap.add_argument("--name", default="S4t"); a_, _ = ap.parse_known_args(); seeds = [int(x) for x in a_.seeds.split(",")]
    t.main()
    V = np.load(base.CACHE / f"var_{a_.name}.npz"); C = V["protos"].shape[2]; cen = np.zeros_like(V["protos"]); has = np.zeros((len(seeds), 5, C), bool)
    for si, s in enumerate(seeds):
        for j, v in enumerate(base.NETS):
            z = np.load(t.OUT / f"out_{v}_s{s}.npz"); assert "centres" in z.files, f"net {v} seed {s} has no centres"; cen[si, j] = z["centres"]; has[si, j] = z["has_centre"]
    np.savez(base.CACHE / f"var_{a_.name}c.npz", protos=cen, qvec=V["qvec"], has=has)
    print(f"var_{a_.name}c written: centres of {len(seeds)} seed(s); players with a centre per network " + ", ".join(str(int(has[0, j].sum())) for j in range(5)), flush=True)
