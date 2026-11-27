#!/usr/bin/env python
"""Section 4M M216 (copy of m161r_tta.py with the paths, the width and the sequence limit of m216_prod.py): orientation-averaged
queries (m124_tta_queries.py, unchanged) for the networks that read games up to 384 plies, and the learned centres as a second
source. The loader is m216_prod's: base.T_MAX = 384 and the tail parts attached (with the same checks).

  python -I logs/nano4/m216_tta.py --seeds 30 --name S10t
      reads results/m216 and /work/u2507115/m216_ckpt (width 512 / feed-forward 2,048), parts in results/m216_tta, writes
      results/m84_cache/var_S10t.npz (prototypes; m84_variant's format) and var_S10tc.npz (protos = the learned centres,
      has = the players with a centre [members, 5, C], the same query vectors) for m164_ident.py.
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m124_tta_queries as t  # noqa: E402
import m216_prod as m216  # noqa: E402

base = t.base
t.OUT, t.CK, t.TMP = ROOT / "results/m216", Path("/work/u2507115/m216_ckpt"), ROOT / "results/m216_tta"
base.D, base.FFN = 512, 2048; base.T_MAX = m216.T_LONG  # base.build_model() then makes the 384-row position table the checkpoints hold
_load_games = base.load_games


def load_games(smoke, log):
    pool = mp.get_context("fork").Pool(12)  # before anything large exists (strict overcommit on the nodes)
    G, Tt = _load_games(smoke, log); m216.attach_tail(G, pool, log, smoke=smoke); pool.close(); pool.join()
    return G, Tt


base.load_games = load_games

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="30"); ap.add_argument("--name", default="S10t"); a_, _ = ap.parse_known_args(); seeds = [int(x) for x in a_.seeds.split(",")]
    t.main()
    V = np.load(base.CACHE / f"var_{a_.name}.npz"); C = V["protos"].shape[2]; cen = np.zeros_like(V["protos"]); has = np.zeros((len(seeds), 5, C), bool)
    for si, s in enumerate(seeds):
        for j, v in enumerate(base.NETS):
            z = np.load(t.OUT / f"out_{v}_s{s}.npz"); assert "centres" in z.files, f"net {v} seed {s} has no centres"; cen[si, j] = z["centres"]; has[si, j] = z["has_centre"]
    np.savez(base.CACHE / f"var_{a_.name}c.npz", protos=cen, qvec=V["qvec"], has=has)
    print(f"var_{a_.name}c written: centres of {len(seeds)} seed(s); players with a centre per network " + ", ".join(str(int(has[0, j].sum())) for j in range(5)), flush=True)
