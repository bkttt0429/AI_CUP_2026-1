import sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS
from ens_compare import ES, load_fold
from m43_gate import decide, qscore
Y, D = [], []
for v in (1, 2, 3, 4):
    y, n, pid, q = load_fold(v)
    p = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(float) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
    Y.append(y); D.append(decide(p))
y, d = np.concatenate(Y), np.concatenate(D)
B, A = LEVELS.index("B"), LEVELS.index("A")
for name, c in (("B", B), ("A", A)):
    m = y == c; k = np.abs(d[m] - c)
    print(f"true {name}: {m.sum()} questions  exact {(k == 0).mean():.3f}  off by one {(k == 1).mean():.3f}  two or more {(k >= 2).mean():.3f}  official score {qscore(d[m], y[m]).mean():.3f}")
m = (y == B) | (y == A); both = m & ((d == B) | (d == A))
print(f"true B or A, ct9 predicted B or A: {both.sum()} of {m.sum()} ({both.sum() / m.sum():.3f}); of these right {(d[both] == y[both]).mean():.3f}")
