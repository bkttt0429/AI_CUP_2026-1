#!/usr/bin/env python
"""Section 4M M219 (copy of m218_tta.py): orientation-averaged queries (m124_tta_queries.py, unchanged) for one arm of
m219_prod.py - the networks are built with the arm's window encoder (pid_win.wrap) - and the learned centres as a second source.

  M219_ARM=w9 M219_CK=/work/u2507115/m219_ckpt/w9 python -I logs/nano4/m219_tta.py --seeds 30,31 --name S12t
      reads results/m219/<arm> (out_<net>_s<seed>.npz, written by m219_prod.py --save-out) and the checkpoints in M219_CK,
      temporary parts in M219_TMP (default /dev/shm/m219/tta_<arm>), writes results/m84_cache/var_S12t.npz (prototypes) and
      var_S12tc.npz (learned centres) for m164_ident.py.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m124_tta_queries as t  # noqa: E402

base = t.base; ARM = os.environ["M219_ARM"]
t.OUT, t.CK, t.TMP = ROOT / "results/m219" / ARM, Path(os.environ.get("M219_CK", f"/work/u2507115/m219_ckpt/{ARM}")), Path(os.environ.get("M219_TMP", f"/dev/shm/m219/tta_{ARM}"))
base.D, base.FFN = 512, 2048
_plain = base.build_model


def build_model():
    """The production network + the window encoder of the arm (its configuration is read from a checkpoint of the folder)."""
    import torch
    import pid_win
    cfg = torch.load(sorted(t.CK.glob("ckpt_*_s*.pt"))[0], map_location="cpu", weights_only=False)["config"]["win"]
    return pid_win.wrap(_plain(), dict(cfg))


base.build_model = build_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="30,31"); ap.add_argument("--name", default="S12t"); a_, _ = ap.parse_known_args(); seeds = [int(x) for x in a_.seeds.split(",")]
    t.main()
    V = np.load(base.CACHE / f"var_{a_.name}.npz"); C = V["protos"].shape[2]; cen = np.zeros_like(V["protos"]); has = np.zeros((len(seeds), 5, C), bool)
    for si, s in enumerate(seeds):
        for j, v in enumerate(base.NETS):
            z = np.load(t.OUT / f"out_{v}_s{s}.npz"); assert "centres" in z.files, f"net {v} seed {s} has no centres"; cen[si, j] = z["centres"]; has[si, j] = z["has_centre"]
    np.savez(base.CACHE / f"var_{a_.name}c.npz", protos=cen, qvec=V["qvec"], has=has)
    print(f"var_{a_.name}c written: centres of {len(seeds)} seed(s); players with a centre per network " + ", ".join(str(int(has[0, j].sum())) for j in range(5)), flush=True)
