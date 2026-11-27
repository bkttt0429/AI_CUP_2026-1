"""M26 follow-up (report only): does replacing the Chinese-rule Human SL evidence with the Japanese-rule one change any decisions?
On the 600 pilot players (20 games each): per player the mean posterior-mean index over the 18 profiles under cn75 and under jp; a simple
5-fold OOF mapping to the label (linear fit, then the expected-score decision with a Gaussian residual), the official per-player score,
the paired player bootstrap of jp - cn75, and how many decisions change.
  python -I logs/nano4/rules_pilot_decide.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.humansl import PROFILES  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}; K = len(LEVELS)
T = pd.concat([pd.read_csv(f, dtype={"player_id": str, "game_key": str}) for f in sorted(glob.glob(str(ROOT / "results/rules_pilot/fold1/shard*.csv")))], ignore_index=True)
for arm in ("cn75", "jp"):
    T[f"pmi_{arm}"] = posterior_mean_index(T[[f"ll_{arm}_{p}" for p in PROFILES]].to_numpy(float))
g = T.groupby("player_id").agg(y=("rank", "first"), cn75=("pmi_cn75", "mean"), jp=("pmi_jp", "mean")); g["y"] = g["y"].map(RANK)
y = g.y.to_numpy(int); n = len(g); fold = np.random.RandomState(0).randint(0, 5, n)
d = (g.jp - g.cn75).to_numpy()
print(f"{n} players; player-level pmi change jp - cn75 (profile index units, 18 profiles): mean {d.mean():+.4f}, sd {d.std():.4f}, max |change| {np.abs(d).max():.4f}; "
      f"corr(cn75, jp) {np.corrcoef(g.cn75, g.jp)[0, 1]:.5f}; spread of pmi across players (sd) {g.cn75.std():.3f}")


def decide(x):
    pred = np.zeros(n, int)
    for k in range(5):
        tr, te = fold != k, fold == k
        A = np.column_stack([np.ones(tr.sum()), x[tr]]); w = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        sig = np.std(y[tr] - A @ w)
        mu = w[0] + w[1] * x[te]
        P = np.exp(-0.5 * ((np.arange(K)[None, :] - mu[:, None]) / sig) ** 2); P /= P.sum(1, keepdims=True)
        pred[te] = [expected_score_decision(p) for p in P]
    return pred


pc, pj = decide(g.cn75.to_numpy()), decide(g.jp.to_numpy())
sc = np.array([score(int(a), int(b)) for a, b in zip(pc, y)]); sj = np.array([score(int(a), int(b)) for a, b in zip(pj, y)])
rng = np.random.default_rng(0); bs = [np.mean((sj - sc)[idx]) for idx in (rng.integers(0, n, n) for _ in range(5000))]
print(f"simple OOF decision (player mean pmi -> level): score cn75 {sc.mean():.4f}, jp {sj.mean():.4f}, diff {sj.mean() - sc.mean():+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}]")
ch = pc != pj
print(f"decisions changed: {ch.sum()} of {n} players ({ch.mean():.1%}); among them jp better {(sj > sc)[ch].sum()}, worse {(sj < sc)[ch].sum()}, same score {(sj == sc)[ch].sum()}")
