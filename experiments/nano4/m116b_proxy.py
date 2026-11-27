"""Section 4M M116b (2026-11-14, report only, CPU): is "the less active training players, with their own level composition" a
better stand-in for the test set than the class-balanced validation? Check against facts already read on Public: eight one-way
pushes across a level boundary were uploaded as probes (rule of m53: on the questions whose base decision is i, the odds of j are
multiplied by r; the flipped questions are the push). Public precision (flipped questions truly j): 1D->A 8/8, B->A 6/13,
5D->6D 1/7, 3D->4D 5/12, 4D->5D 7/9, 5D->4D 1/8, B->C 6/9, C->D 4/9.
For each push: precision among the validation questions, (i) all players, (ii) players by their number of games in the data set.
Also each group's level composition, and the number of flips per 400 questions (Public had 8, 13, 7, 12, 9, 8, 9, 9).
  python -I logs/nano4/m116b_proxy.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402

b, s2, mb, ES, FOLDS, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.SM
PUSH = (("1D", "A", 2.0, 8, 8), ("B", "A", 2.0, 6, 13), ("5D", "6D", 2.6, 1, 7), ("3D", "4D", 2.0, 5, 12), ("4D", "5D", 2.0, 7, 9), ("5D", "4D", 2.0, 1, 8), ("B", "C", 2.0, 6, 9), ("C", "D", 2.0, 4, 9))


def main():
    pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); ngames = pf.n_games.astype(int); rows = []
    for v in FOLDS:
        y, n, pid, qd = m3.load_fold(v); y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid).astype(str); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); d0 = (P @ mb.S.T).argmax(1); fl = {}
        for i_, j_, r, _, _ in PUSH:
            i, j = LEVELS.index(i_), LEVELS.index(j_); P2 = P.copy(); P2[:, j] *= r; P2 /= P2.sum(1, keepdims=True); fl[f"{i_}->{j_}"] = (d0 == i) & ((P2 @ mb.S.T).argmax(1) == j)
        rows.append(pd.DataFrame({"y": y, "n": n, "ng": ngames.reindex(pid).to_numpy(), "d0": d0, **fl}))
    D = pd.concat(rows); groups = {"all players": D.ng > 0, "1-25 games": D.ng <= 25, "26-40": (D.ng > 25) & (D.ng <= 40), "41-70": (D.ng > 40) & (D.ng <= 70), "71-170": (D.ng > 70) & (D.ng <= 170), "171+": D.ng > 170}
    print("level composition of the players (20-game questions), by number of games in the data set:")
    for gname, s in groups.items():
        x = D[s & (D.n == 20)]; print(f"   {gname:12s} ({len(x):4d} players): " + "  ".join(f"{LEVELS[c]} {float((x.y == c).mean()):.3f}" for c in range(10)))
    sub = pd.read_csv(ROOT / "submission_rank_ct75u.csv", dtype=str); print("   test, ct75u's predicted levels:      " + "  ".join(f"{lv} {float((sub.pred_rank == lv).mean()):.3f}" for lv in LEVELS))
    print("\nprecision of each push (flipped questions that truly are the target level) and flips per 400 questions; Public in the first column:")
    print("   push      Public      " + "   ".join(f"{g:>14s}" for g in groups))
    for i_, j_, r, ok, tot in PUSH:
        k = f"{i_}->{j_}"; j = LEVELS.index(j_); cells = []
        for gname, s in groups.items():
            x = D[s]; f = x[k].to_numpy(); cells.append(f"{float((x.y[f] == j).mean()) if f.sum() >= 10 else float('nan'):.2f} ({400 * f.mean():4.1f})")
        print(f"   {k:8s}  {ok}/{tot} = {ok / tot:.2f}   " + "   ".join(f"{c:>14s}" for c in cells))
    pub = np.array([ok / tot for _, _, _, ok, tot in PUSH]); print("\ncorrelation of the eight Public precisions with each group's precisions (Pearson / Spearman):")
    for gname, s in groups.items():
        x = D[s]; pr = np.array([float((x.y[x[f"{i_}->{j_}"].to_numpy()] == LEVELS.index(j_)).mean()) for i_, j_, _, _, _ in PUSH]); ok_ = np.isfinite(pr)
        print(f"   {gname:12s}: {float(np.corrcoef(pub[ok_], pr[ok_])[0, 1]):+.2f} / {float(pd.Series(pub[ok_]).corr(pd.Series(pr[ok_]), method='spearman')):+.2f}")


if __name__ == "__main__":
    main()
