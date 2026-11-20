import sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd(); sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate
from ens_compare import ES, load_fold
for name, tags in (("ct7", ("C-f0-x40-oh", "T-f0-x40-oh")), ("ct9", ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh"))):
    bal, plain, n_q, singles, inner = [], [], 0, [], []
    for v in (1, 2, 3, 4):
        y, n, pid, q = load_fold(v)
        files = [ES / f"val{v}/{t}_s{s}.npz" for t in tags for s in (0, 1, 2)]
        P = np.mean([np.load(f)["proba"].astype(float) for f in files], axis=0)
        s = evaluate(P, y, q, sizes=SIZES)[1]
        bal.append(balanced_mean_over_n(s, y, n)); plain.append(s.mean()); n_q += len(s)
        for f in files:
            singles.append(balanced_mean_over_n(evaluate(np.load(f)["proba"].astype(float), y, q, sizes=SIZES)[1], y, n))
    print(f"{name}: CV ensemble class-balanced {np.mean(bal):.4f} (folds {' '.join(f'{x:.4f}' for x in bal)}), plain question average {np.mean(plain):.4f}; "
          f"{n_q} validation questions; single members {min(singles):.4f}-{max(singles):.4f}")
