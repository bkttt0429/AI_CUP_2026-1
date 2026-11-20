#!/usr/bin/env python
"""Section 4M M221 (copy of m219_tta.py): orientation-averaged queries (m124_tta_queries.py, unchanged) for the networks of
m221_prod.py (external strangers + the window input) and the learned centres as a second source.

  M221_ARM=wg25 M221_CK=results/m221/wg25_ckpt python -I logs/nano4/m221_tta.py --seeds 30,31 --name S13u
      reads results/m221/<arm> (out_<net>_s<seed>.npz, written by m221_prod.py --save-out) and the checkpoints in M221_CK
      (default results/m221/<arm>_ckpt), temporary parts in M221_TMP (default /dev/shm/m221/tta_<arm>), writes
      results/m84_cache/var_S13u.npz (prototypes) and var_S13uc.npz (learned centres) for m164_ident.py.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m124_tta_queries as t  # noqa: E402

base = t.base; ARM = os.environ["M221_ARM"]
t.OUT, t.CK, t.TMP = ROOT / "results/m221" / ARM, Path(os.environ.get("M221_CK", str(ROOT / "results/m221" / f"{ARM}_ckpt"))), Path(os.environ.get("M221_TMP", f"/dev/shm/m221/tta_{ARM}"))
base.D, base.FFN = 512, 2048
_plain = base.build_model


def build_model():
    """The production network + the single 9 x 9 window encoder of m221_prod.py (its checkpoints carry m218_prod's configuration, without the window entry)."""
    import pid_win
    return pid_win.wrap(_plain(), {"size": 9, "ch": 64})


base.build_model = build_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="30,31"); ap.add_argument("--name", default="S13u"); a_, _ = ap.parse_known_args(); seeds = [int(x) for x in a_.seeds.split(",")]
    t.main()
    V = np.load(base.CACHE / f"var_{a_.name}.npz"); C = V["protos"].shape[2]; cen = np.zeros_like(V["protos"]); has = np.zeros((len(seeds), 5, C), bool)
    for si, s in enumerate(seeds):
        for j, v in enumerate(base.NETS):
            z = np.load(t.OUT / f"out_{v}_s{s}.npz"); assert "centres" in z.files, f"net {v} seed {s} has no centres"; cen[si, j] = z["centres"]; has[si, j] = z["has_centre"]
    np.savez(base.CACHE / f"var_{a_.name}c.npz", protos=cen, qvec=V["qvec"], has=has)
    print(f"var_{a_.name}c written: centres of {len(seeds)} seed(s); players with a centre per network " + ", ".join(str(int(has[0, j].sum())) for j in range(5)), flush=True)
