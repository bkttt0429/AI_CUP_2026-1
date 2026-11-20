import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd(); sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS
from ens_compare import ES, load_fold
from m43_gate import decide, qscore
Y, D, P = [], [], []
for v in (1, 2, 3, 4):
    y, n, pid, q = load_fold(v)
    p = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(float) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
    Y.append(y); D.append(decide(p)); P.append(p)
y, d, p = np.concatenate(Y), np.concatenate(D), np.concatenate(P)
C = pd.crosstab(pd.Series([LEVELS[i] for i in y], name="true"), pd.Series([LEVELS[i] for i in d], name="pred")).reindex(index=LEVELS, columns=LEVELS, fill_value=0)
print("confusion (rows true, columns predicted):"); print(C.to_string())
s = qscore(d, y); lost = 1 - s
print(f"\nper true class: questions, exact, official score, share of all lost points")
for i, lv in enumerate(LEVELS):
    m = y == i; print(f"  {lv:3s} {m.sum():6d}  exact {(d[m] == i).mean():.3f}  score {s[m].mean():.3f}  lost share {lost[m].sum() / lost.sum():.3f}")
print("\nadjacent boundaries: confusions between the two classes (both directions), share of all errors, ct9's pairwise AUC (log P(hi)/P(lo) on questions of the two classes)")
err = d != y
for i in range(9):
    pair = ((y == i) & (d == i + 1)) | ((y == i + 1) & (d == i))
    m = (y == i) | (y == i + 1)
    lo = np.log(np.clip(p[m, i + 1], 1e-9, 1)) - np.log(np.clip(p[m, i], 1e-9, 1)); lab = (y[m] == i + 1).astype(int)
    r = pd.Series(lo).rank().to_numpy(); n1 = lab.sum(); n0 = len(lab) - n1; auc = (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    print(f"  {LEVELS[i]:>3s} | {LEVELS[i + 1]:<3s}: {pair.sum():5d} questions ({pair.sum() / err.sum():.3f} of the errors), pairwise AUC {auc:.3f}")
