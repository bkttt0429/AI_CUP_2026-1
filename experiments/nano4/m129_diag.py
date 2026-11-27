#!/usr/bin/env python
"""M129 diagnostics for the candidate ct104 (report only): (1) in CV, the changes the final choice makes relative to the ct75u
rule, by kind (size of the jump, direction, the A / 1D pair, question size, players' activity) with their mean gain; (2) the
look-alike reading on the test against validation (top similarity, entropy of the level distribution, by question size) -
the check that the signal exists on the test; (3) the changed test questions by kind."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m105_voters as m5  # noqa: E402
import m110_fusion as mf  # noqa: E402
import m123_lookalike_sym as la  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM
CACHE = ROOT / "results/m84_cache"; FIVE = ("v3", "v5", "s2", "s3c", "s3"); W = {5: 0.5, 10: 0.5, 20: 0.1}
rep = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = rep["density_ratio"]
ids = [REC[k][0] for k in FIVE]; V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~m15" for x in ids]])
rec = dict(REC)
for k in FIVE:
    idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec[k] = (f"{idn}~m15", enc, cc, temp, pi0, bmin, scale[idn])
z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); players = q["players"].astype(str); plab = q["plab"]
pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy(); v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]
ngames = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(int)
rows = []; vstats = []
for v in FOLDS:
    y, n, pid, qd = m3.load_fold(v); y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {0: {}, 2: {}}
    for A, R in ((0, REC), (2, rec)):
        for name, (idn, enc, cc, temp, pi0, bmin, sc) in R.items():
            ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[A][name] = L * Tn if temp else L
    qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
    for m in range(NM):
        np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
    tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); lr_, smax, ent = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE); RD = lr_ - lr_.mean(0, keepdims=True); vstats.append(np.stack([n, smax, ent], 1))
    wq = np.array([W[int(m)] for m in n])
    for seeds in ((0, 1, 2), (6, 7, 8)):
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); _, d_old = mf.rule75u(P, Ls[0]); _, d_new = mf.rule75u(la.pooled(P, RD, wq), Ls[2]); _, d_b = mf.rule75u(la.pooled(P, RD, wq), Ls[0]); _, d_a = mf.rule75u(P, Ls[2])
        for i in np.flatnonzero(d_new != d_old):
            rows.append({"seeds": seeds[0], "n": int(n[i]), "y": int(y[i]), "old": int(d_old[i]), "new": int(d_new[i]), "gain": float(SM[d_new[i], y[i]] - SM[d_old[i], y[i]]), "games": int(ngames.get(pid[i], 0)), "by": "A" if d_b[i] == d_old[i] else ("B" if d_a[i] == d_old[i] else "both")})
df = pd.DataFrame(rows); nq = sum(len(m3.load_fold(v)[0]) for v in FOLDS)
print(f"CV, final choice (m15 for the five recipes; look-alike weights {W}); {len(df)} changed decisions over the two seed sets of {nq} questions each")


def tab(title, key):
    print(f"  {title}:")
    for k, g in df.groupby(key):
        print(f"      {str(k):22s} {len(g):5d} ({len(g) / (2 * nq):.4f} of the questions)  better {float((g.gain > 0).mean()):.3f}  worse {float((g.gain < 0).mean()):.3f}  mean gain {g.gain.mean():+.3f}  total {g.gain.sum():+.1f}")


df["jump"] = np.where(np.abs(df.new - df.old) >= 2, "two levels or more", "one level"); df["dir"] = np.where(df.new > df.old, "up", "down"); tab("by the size of the change", "jump"); tab("by direction", "dir"); tab("by question size", "n"); tab("by what caused it (A = statistic, B = look-alike)", "by")
df["pair"] = [f"{LEVELS[o]}->{LEVELS[w_]}" for o, w_ in zip(df.old, df.new)]; top = df.groupby("pair").gain.agg(["size", "mean", "sum"]).sort_values("size", ascending=False).head(14)
print("  the most frequent changes (count, mean gain, total):"); print("      " + "   ".join(f"{k} {int(r['size'])} {r['mean']:+.2f} {r['sum']:+.1f}" for k, r in top.iterrows()))
df["act"] = np.where(df.games <= 40, "1-40 games", np.where(df.games <= 170, "41-170", "171+")); tab("by the player's games in the data set", "act")
big = df[df.jump != "one level"]; print(f"  changes of two levels or more: {len(big)}; by question size: " + ", ".join(f"n={k}: {len(g)} (mean gain {g.gain.mean():+.2f})" for k, g in big.groupby("n")))
# ---- the reading on the test against validation
pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); tqid = T["5e"]["qid"].astype(str); nq_t = pd.Series(tqid).value_counts().reindex(qids).to_numpy()
ti = np.flatnonzero(q["qtest"] >= 0); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in tqid]); tq = np.zeros((NM, 800, 128))
for m in range(NM):
    np.add.at(tq[m], gq, QV[m, tgt].astype(np.float64))
tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); lr, smax_t, ent_t = la.reading(tq, [PR[m, 4].astype(np.float64) for m in range(NM)], plab, la.BE); vs = np.concatenate(vstats)
print("the look-alike reading on the test against validation, by question size: mean top similarity | mean entropy of the level distribution | share of questions whose reading puts > 0.5 on one level")
lrv = None
for m in (5, 10, 15, 20):
    s_t = nq_t == m; s_v = vs[:, 0] == m; print(f"    n={m}: test {smax_t[s_t].mean():.3f} | {ent_t[s_t].mean():.3f} | {float((np.exp(lr[s_t]).max(1) > 0.5).mean()):.3f}" + (f"   validation {vs[s_v, 1].mean():.3f} | {vs[s_v, 2].mean():.3f}" if s_v.any() else "   (no validation questions of this size)"))
sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
c75u, c104 = sub("ct75u"), sub("ct104"); rk = np.vectorize(b.RANK.get); ch = c104 != c75u; pub = np.char.startswith(qids.astype(str), "pub_"); jj = rk(c104) - rk(c75u)
print(f"test: {int(ch.sum())} changed; by question size: " + ", ".join(f"n={m}: {int((ch & (nq_t == m)).sum())} of {int((nq_t == m).sum())}" for m in (5, 10, 15, 20)) + f"; up {int((jj > 0).sum())}, down {int((jj < 0).sum())}; two levels or more: " + ", ".join(f"{qids[i]} ({int(nq_t[i])} games) {c75u[i]}->{c104[i]}" for i in np.flatnonzero(np.abs(jj) >= 2)))
pairs = pd.Series([f"{a}->{c}" for a, c in zip(c75u[ch], c104[ch])]).value_counts(); print("test, changes by pair: " + "  ".join(f"{k} {v}" for k, v in pairs.items()))
