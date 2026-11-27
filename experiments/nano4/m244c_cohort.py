#!/usr/bin/env python
"""Section 4M M244c (2026-11-19), report only: where does the gain of the M244b stacker come from?
Reads results/r_variants/m244b_decisions.npz (seeds 0-2: truth, size, player, the stacker's and the linear pool's plain
decisions, the stacker's guarded decisions, the reference = the ct106 rule). Recomputes the ct112 rule = guard2(linear pool,
reference) and reports the stacker's gain (a) without guards against the linear pool and (b) with guards against the ct112
rule, separately for the validation players with fewer than 20 games in the data set and for the others.

  python -I logs/nano4/m244c_cohort.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
from aicup.qeval import SIZES, balanced, paired_diff  # noqa: E402

SM = mb7.SM; LEV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def main():
    z = np.load(ROOT / "results/r_variants/m244b_decisions.npz", allow_pickle=True); y, n, pid = z["y"], z["n"], z["pid"].astype(str); sp, lp, sg, ref = z["stack_plain"], z["linear_plain"], z["stack_guarded"], z["rule"]
    rule112 = mf7.guard2(lp, ref); act = pd.concat([pd.read_csv(p, usecols=["player_id"], dtype=str) for p in sorted((ROOT / "data/training").glob("train_*.csv"))]).player_id.value_counts(); ng = act.reindex(pid).to_numpy()
    S = lambda d: SM[d, y]  # noqa: E731
    print(f"questions {len(y)}; ct106 rule {np.mean([balanced(S(ref), y, n == s) for s in SIZES]):.4f}, ct112 rule {np.mean([balanced(S(rule112), y, n == s) for s in SIZES]):.4f}, stacker with guards {np.mean([balanced(S(sg), y, n == s) for s in SIZES]):.4f}; linear pool without guards {np.mean([balanced(S(lp), y, n == s) for s in SIZES]):.4f}, stacker without guards {np.mean([balanced(S(sp), y, n == s) for s in SIZES]):.4f}")
    for nm, a, b_ in (("WITHOUT guards: stacker minus the linear pool", S(sp), S(lp)), ("WITH guards: stacker minus the ct112 rule", S(sg), S(rule112))):
        d = paired_diff(a, b_, y, pid, n, sizes=SIZES, n=5000); print(f"\n{nm}: all validation players {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  by n " + " ".join(f"{k}: {v:+.4f}" for k, v in d["by_n"].items()))
        for lab, m in (("players with 20 or more games", ng >= 20), ("players with 15-19 games", (ng >= 15) & (ng < 20)), ("players with fewer than 15 games", ng < 15)):
            sizes = tuple(s for s in SIZES if (m & (n == s)).sum() >= 200)
            dm = paired_diff(a[m], b_[m], y[m], pid[m], n[m], sizes=sizes, n=3000)
            print(f"   {lab:32s} ({int(m.sum()):5d} questions): class-balanced {dm['diff']:+.4f} [{dm['ci95'][0]:+.4f}, {dm['ci95'][1]:+.4f}]  by n " + " ".join(f"{k}: {v:+.4f}" for k, v in dm["by_n"].items()) + f"; plain mean {float((a - b_)[m].mean()):+.4f}; answers changed {float((a != b_)[m].mean()):.3f}")
        ch = (sp != lp) if nm.startswith("WITHOUT") else (sg != rule112); da, db_ = (sp, lp) if nm.startswith("WITHOUT") else (sg, rule112)
        for lab, m in (("20 or more games", ng >= 20), ("fewer than 20 games", ng < 20)):
            t = pd.DataFrame({"k": [f"{LEV[x]}>{LEV[w]}" for x, w in zip(db_[ch & m], da[ch & m])], "g": (a - b_)[ch & m]}).groupby("k").g.agg(["size", "sum"]).sort_values("sum", ascending=False)
            print(f"   kinds of changes, players with {lab} (number, total points): best " + ", ".join(f"{k} {int(r['size'])} {r['sum']:+.1f}" for k, r in t.head(6).iterrows()) + " | worst " + ", ".join(f"{k} {int(r['size'])} {r['sum']:+.1f}" for k, r in t.tail(5).iterrows()))
    print("\nREPORT ONLY")


if __name__ == "__main__":
    sys.exit(main())
