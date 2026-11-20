#!/usr/bin/env python
"""Section 4M M218 (copy of m216_tta.py without the tail parts): orientation-averaged queries (m124_tta_queries.py, unchanged) for
one arm of m218_prod.py, and the learned centres as a second source.

  M218_ARM=g25o M218_CK=/dev/shm/m218/g25o python -I logs/nano4/m218_tta.py --seeds 30 --name S11t
      reads results/m218/<arm> (out_<net>_s<seed>.npz, written by m218_prod.py --save-out) and the checkpoints in M218_CK
      (default /work/u2507115/m218_ckpt/<arm>), temporary parts in M218_TMP (default /dev/shm/m218/tta_<arm>), writes
      results/m84_cache/var_S11t.npz (prototypes) and var_S11tc.npz (learned centres) for m164_ident.py.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m124_tta_queries as t  # noqa: E402

base = t.base; ARM = os.environ["M218_ARM"]
t.OUT, t.CK, t.TMP = ROOT / "results/m218" / ARM, Path(os.environ.get("M218_CK", f"/work/u2507115/m218_ckpt/{ARM}")), Path(os.environ.get("M218_TMP", f"/dev/shm/m218/tta_{ARM}"))
base.D, base.FFN = 512, 2048

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="30"); ap.add_argument("--name", default="S11t"); a_, _ = ap.parse_known_args(); seeds = [int(x) for x in a_.seeds.split(",")]
    t.main()
    V = np.load(base.CACHE / f"var_{a_.name}.npz"); C = V["protos"].shape[2]; cen = np.zeros_like(V["protos"]); has = np.zeros((len(seeds), 5, C), bool)
    for si, s in enumerate(seeds):
        for j, v in enumerate(base.NETS):
            z = np.load(t.OUT / f"out_{v}_s{s}.npz"); assert "centres" in z.files, f"net {v} seed {s} has no centres"; cen[si, j] = z["centres"]; has[si, j] = z["has_centre"]
    np.savez(base.CACHE / f"var_{a_.name}c.npz", protos=cen, qvec=V["qvec"], has=has)
    print(f"var_{a_.name}c written: centres of {len(seeds)} seed(s); players with a centre per network " + ", ".join(str(int(has[0, j].sum())) for j in range(5)), flush=True)
