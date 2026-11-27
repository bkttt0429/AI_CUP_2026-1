#!/usr/bin/env python
"""M132b (report only): the kinds of changes of M129's bundle WITHOUT the two guards, by the player's number of games in the data
set (natural level mix of each group, no class balancing). Question: would the low-activity players (the group the test set
resembles in its share of one-stone games) have predicted what the Public decode showed - that leaving A and 5D->6D lose?"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m110_fusion as mf  # noqa: E402
import m123_lookalike_sym as la  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES  # noqa: E402

b, s2, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.ES, m3.FOLDS, m3.REC, m3.SM
CACHE = ROOT / "results/m84_cache"; FIVE = ("v3", "v5", "s2", "s3c", "s3"); W0 = {5: 0.5, 10: 0.5, 20: 0.1}
rep = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = rep["density_ratio"]
ids = [REC[k][0] for k in FIVE]; V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~m15" for x in ids]]); rec = dict(REC)
for k in FIVE:
    idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec[k] = (f"{idn}~m15", enc, cc, temp, pi0, bmin, scale[idn])
z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); players = q["players"].astype(str); plab = q["plab"]
tabf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); pfold = tabf.fold.astype(int).reindex(players).to_numpy(); ngames = tabf.n_games.astype(int)
v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]; rows = []
for v in FOLDS:
    y, n, pid, qd = m3.load_fold(v); y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {0: {}, 2: {}}
    for A, R in ((0, REC), (2, rec)):
        for name, (idn, enc, cc, temp, pi0, bmin, sc) in R.items():
            ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[A][name] = L * Tn if temp else L
    qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
    for m in range(NM):
        np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
    tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE); RD = lr_ - lr_.mean(0, keepdims=True); wq = np.array([W0[int(m)] for m in n]); g = ngames.reindex(pid).to_numpy()
    for seeds in ((0, 1, 2), (6, 7, 8)):
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); _, d_old = mf.rule75u(P, Ls[0]); _, d_new = mf.rule75u(la.pooled(P, RD, wq), Ls[2])
        for i in np.flatnonzero(d_new != d_old):
            rows.append((seeds[0], int(n[i]), int(y[i]), int(d_old[i]), int(d_new[i]), float(SM[d_new[i], y[i]] - SM[d_old[i], y[i]]), int(g[i])))
df = pd.DataFrame(rows, columns=["seeds", "n", "y", "old", "new", "gain", "games"]); df["kind"] = [f"{LEVELS[o]}->{LEVELS[w_]}" for o, w_ in zip(df.old, df.new)]
df["grp"] = np.where(df.games <= 25, "1-25", np.where(df.games <= 40, "26-40", np.where(df.games <= 170, "41-170", "171+")))
print("unguarded bundle, changes by kind and by the player's games in the data set: changes, mean gain per change (natural level mix)")
print(f"{'kind':9s} " + "  ".join(f"{g_:>16s}" for g_ in ("all", "1-25", "26-40", "41-170", "171+")))
for k, r in df.groupby("kind").gain.agg(["size"]).sort_values("size", ascending=False).iterrows():
    if r["size"] < 60:
        continue
    line = f"{k:9s} "
    for g_ in ("all", "1-25", "26-40", "41-170", "171+"):
        s_ = df[(df.kind == k)] if g_ == "all" else df[(df.kind == k) & (df.grp == g_)]; line += f"  {len(s_):5d} {s_.gain.mean() if len(s_) else float('nan'):+.3f}   "
    print(line)
for name, m in (("leaving A (A->B, A->1D, A->others)", df.old == LEVELS.index("A")), ("5D->6D", (df.old == 8) & (df.new == 9)), ("everything else", ~((df.old == 3) | ((df.old == 8) & (df.new == 9))))):
    print(f"{name:36s} " + "  ".join(f"{g_}: {int((m & (df.grp == g_)).sum())} changes {df[m & (df.grp == g_)].gain.mean():+.3f}" for g_ in ("1-25", "26-40", "41-170", "171+")))
print("level mix of the questions' players by group (share of A / 1D / 5D / 6D among the changed questions' truths): " + "  ".join(f"{g_}: A {float((df[df.grp == g_].y == 3).mean()):.2f} 1D {float((df[df.grp == g_].y == 4).mean()):.2f} 5D {float((df[df.grp == g_].y == 8).mean()):.2f} 6D {float((df[df.grp == g_].y == 9).mean()):.2f}" for g_ in ("1-25", "26-40", "41-170", "171+")))
