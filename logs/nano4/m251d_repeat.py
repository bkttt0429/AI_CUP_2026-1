#!/usr/bin/env python
"""M251d (2026-11-19), report only: repeated opponents inside one unit, test questions against validation (player, colour) units,
with the test rows' question ids taken from m82_test.npz (the alignment verified in M248 / M251). Re-reads the M186 statement
"repeated opponents are ten times more frequent in the validation questions than in the test questions", which was computed
from an unsorted games table (question ids of the test rows scrambled; see M251b).

  python -I logs/nano4/m251d_repeat.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()


def rate(unit, cand):
    df = pd.DataFrame({"u": unit, "c": cand}); c = df.groupby(["u", "c"]).size(); same = (c * (c - 1) / 2).groupby(level=0).sum(); n = df.groupby("u").size(); pairs = n * (n - 1) / 2; ok = pairs > 0
    return float(same[ok].sum() / pairs[ok].sum()), int(ok.sum()), float((c > 1).groupby(level=0).any()[ok].mean())


def main():
    dv = np.load(ROOT / "results/r_variants/m82_val.npz"); sv = np.load(ROOT / "results/r_variants/m84_SQ4~m15_val.npz"); assert np.array_equal(dv["key"], sv["key"])
    dt = np.load(ROOT / "results/r_variants/m82_test.npz"); st = np.load(ROOT / "results/r_variants/m84_SQ4~m15_test.npz"); assert np.array_equal(dt["qid"], st["qid"])
    if "f_opp_best" not in st.files:
        sys.exit(f"no candidate ids in the test file: {st.files}")
    pidv = dv["pid"].astype(str); tbv = dv["target_black"].astype(bool); hdv = dv["hd"].astype(int); omv = sv["f_opp_m"].astype(float); bv = sv["f_opp_best"].astype(str); qt = dt["qid"].astype(str); hdt = dt["hd"].astype(int); omt = st["f_opp_m"].astype(float); bt = st["f_opp_best"].astype(str)
    uv = np.char.add(pidv, np.where(tbv, ":B", ":W")); rng = np.random.default_rng(0)
    print("threshold (validation quantile of the margin, even games) | confident games: validation, test | pairs of confident anchors of one unit with the SAME candidate: validation (player, colour) units, test questions, test with shuffled question ids | units with any repeat: validation, test")
    for q_ in (0.80, 0.90, 0.95, 0.98):
        thr = float(np.quantile(omv[hdv == 0], q_)); mv = (hdv <= 1) & (omv >= thr); mt = (hdt <= 1) & (omt >= thr); rv, nv, av = rate(uv[mv], bv[mv]); rt, nt, at = rate(qt[mt], bt[mt]); rs, _, _ = rate(rng.permutation(qt[mt]), bt[mt])
        print(f"   q{int(q_ * 100)} ({thr:.2f}) | {float(mv.mean()):.3f}, {float(mt.mean()):.3f} | {rv:.4f} ({nv} units), {rt:.4f} ({nt} units), {rs:.4f} | {av:.3f}, {at:.3f}   ratio validation / test {rv / max(rt, 1e-9):.2f}")


if __name__ == "__main__":
    main()
