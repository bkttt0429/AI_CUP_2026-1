"""Section 4M M47 follow-up: Black Box Shift Estimation (Lipton et al. 2018) of the test class shares. C[i, j] = P(ct9 predicts i | true j)
from the four validation folds (ct9's out-of-fold decisions); the test's predicted shares q = C w, w >= 0, sum 1, by non-negative least
squares; bootstrap over the 800 test questions and over the validation players (1,000). Also: the A and 1D shares the test predictions would
show under the validation prior and under a uniform prior.
  python -I logs/nano4/m47_bbse.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, decide  # noqa: E402


def simplex_ls(C, q, iters=20000, lr=0.5):
    """argmin ||C w - q||^2 over the probability simplex (projected gradient)."""
    w = np.full(C.shape[1], 1 / C.shape[1])
    for _ in range(iters):
        w = w - lr * C.T @ (C @ w - q)
        u = np.sort(w)[::-1]; css = np.cumsum(u); rho = np.nonzero(u * np.arange(1, len(u) + 1) > (css - 1))[0][-1]
        w = np.maximum(w - (css[rho] - 1) / (rho + 1), 0)
    return w


Y, Dd, PID = [], [], []
for v in FOLDS:
    y, n, pid, q = load_fold(v)
    P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(float) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
    Y.append(y); Dd.append(decide(P)); PID.append(pid)
y, d, pid = np.concatenate(Y), np.concatenate(Dd), np.concatenate(PID)


def conf(y, d):
    C = np.zeros((10, 10))
    for j in range(10):
        m = y == j; C[:, j] = np.bincount(d[m], minlength=10) / max(m.sum(), 1)
    return C


C = conf(y, d)
sub = pd.read_csv(ROOT / "submission_rank_ct9.csv"); dt = sub["pred_rank"].map({lv: i for i, lv in enumerate(LEVELS)}).to_numpy()
pub = sub["question_id"].str.startswith("pub").to_numpy()
val_prior = np.bincount(y, minlength=10) / len(y)
for name, w in (("validation prior", val_prior), ("uniform prior", np.full(10, 0.1))):
    qq = C @ w; print(f"expected test prediction shares under the {name}: " + " ".join(f"{lv} {x:.3f}" for lv, x in zip(LEVELS, qq)))
for name, m in (("all 800", np.ones(len(dt), bool)), ("public", pub), ("private", ~pub)):
    qt = np.bincount(dt[m], minlength=10) / m.sum()
    print(f"observed test prediction shares ({name}): " + " ".join(f"{lv} {x:.3f}" for lv, x in zip(LEVELS, qt)))
qt = np.bincount(dt, minlength=10) / len(dt)
w = simplex_ls(C, qt)
rng = np.random.default_rng(0); players = np.unique(pid); idx = {p: np.flatnonzero(pid == p) for p in players}
bs = []
for _ in range(1000):
    it = rng.integers(0, len(dt), len(dt)); qb = np.bincount(dt[it], minlength=10) / len(dt)
    ii = np.concatenate([idx[p] for p in rng.choice(players, len(players))]); Cb = conf(y[ii], d[ii])
    bs.append(simplex_ls(Cb, qb, iters=4000))
bs = np.array(bs); lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
print("BBSE test class shares (95% CI): " + "  ".join(f"{lv} {x:.3f} [{a:.3f}, {b:.3f}]" for lv, x, a, b in zip(LEVELS, w, lo, hi)))
print(f"validation prior for comparison: " + " ".join(f"{lv} {x:.3f}" for lv, x in zip(LEVELS, val_prior)))
A, D1 = LEVELS.index("A"), LEVELS.index("1D")
r = bs[:, A] / np.maximum(bs[:, D1], 1e-9)
print(f"A share {w[A]:.3f} [{lo[A]:.3f}, {hi[A]:.3f}] (validation {val_prior[A]:.3f}); 1D share {w[D1]:.3f} [{lo[D1]:.3f}, {hi[D1]:.3f}] (validation {val_prior[D1]:.3f}); "
      f"A/1D ratio median {np.median(r):.2f} [{np.percentile(r, 2.5):.2f}, {np.percentile(r, 97.5):.2f}] (validation {val_prior[A] / val_prior[D1]:.2f}, uniform 1.00)")
