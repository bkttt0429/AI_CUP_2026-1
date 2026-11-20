#!/usr/bin/env python
"""Section 4M M169 step 2 (2026-11-15): an identifier's rows split by the outcome proxy (who made the last move), so that the
unchanged evidence code calibrates the two halves separately (design doc M169).

  python -I logs/nano4/m169_split.py SX4L      -> m84_SX4Lw~m15_{val,test}.npz (the target made the last move), m84_SX4Ll~m15_...
"""
import sys
from pathlib import Path

import numpy as np

RV = Path.cwd() / "results/r_variants"; tag = sys.argv[1]; lv = np.load(RV / "m169_last_val.npz"); lt = np.load(RV / "m169_last_test.npz")
v = dict(np.load(RV / f"m84_{tag}~m15_val.npz", allow_pickle=True)); t = dict(np.load(RV / f"m84_{tag}~m15_test.npz", allow_pickle=True)); assert np.array_equal(v["key"], lv["key"]) and np.array_equal(t["qid"], lt["qid"])
for suf, o in (("w", 1), ("l", 0)):
    mv_, mt_ = lv["last_is_target"] == o, lt["last_is_target"] == o
    np.savez_compressed(RV / f"m84_{tag}{suf}~m15_val.npz", **{k: x[mv_] for k, x in v.items()}); np.savez_compressed(RV / f"m84_{tag}{suf}~m15_test.npz", **{k: x[mt_] for k, x in t.items()})
    print(f"{tag}{suf}: validation rows {int(mv_.sum())}, test rows {int(mt_.sum())}")
