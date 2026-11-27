#!/usr/bin/env python
"""M129 diagnostic 2 (report only): distribution of the look-alike reading's top similarity on the test against validation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m123_lookalike_sym as la  # noqa: E402

b, FOLDS = m3.b, m3.FOLDS
CACHE = ROOT / "results/m84_cache"
V, T, key_pid, used = b.sources(["5e"]); z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); players = q["players"].astype(str); plab = q["plab"]
pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
QS = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
for var in ("S4e", "S2y"):
    v_ = np.load(CACHE / f"var_{var}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]; val = []
    for v in FOLDS:
        y, n, pid, qd = m3.load_fold(v); n = np.asarray(n); gk = list(qd["game_keys"]); qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); tq = np.zeros((NM, len(n), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)
        for cset, name in ((np.flatnonzero(pfold != v), "seen candidates only"), (np.arange(len(players)), "all candidates (incl. the fold's own players)")):
            S = sum(tq[m] @ PR[m, v - 1][cset].astype(np.float64).T for m in range(NM)) / NM; top = S.max(1); own = None
            if name.startswith("all"):
                pi = {p: i for i, p in enumerate(players)}; me = np.array([pi[str(p)] for p in pid]); own = (S.argmax(1) == me)
            val.append((name, n, top, own))
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); tqid = T["5e"]["qid"].astype(str); nq_t = pd.Series(tqid).value_counts().reindex(qids).to_numpy()
    ti = np.flatnonzero(q["qtest"] >= 0); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in tqid]); tq = np.zeros((NM, 800, 128))
    for m in range(NM):
        np.add.at(tq[m], gq, QV[m, tgt].astype(np.float64))
    tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); St = sum(tq[m] @ PR[m, 4].astype(np.float64).T for m in range(NM)) / NM; top_t = St.max(1); srt = np.sort(St, axis=1)
    print(f"\n=== vectors {var}: top similarity of a question's mean target vector to the candidates' prototypes, quantiles {QS}")
    for m in (5, 10, 20):
        for name in ("seen candidates only", "all candidates (incl. the fold's own players)"):
            n_ = np.concatenate([x[1] for x in val if x[0] == name]); t_ = np.concatenate([x[2] for x in val if x[0] == name]); s_ = n_ == m; line = f"  n={m:2d} validation, {name:46s} " + " ".join(f"{np.quantile(t_[s_], u):.3f}" for u in QS)
            if name.startswith("all"):
                o_ = np.concatenate([x[3] for x in val if x[0] == name]); line += f"   (the question's own player is the top candidate in {float(o_[s_].mean()):.3f}; top similarity when it is: median {np.median(t_[s_ & o_]):.3f}, when not: {np.median(t_[s_ & ~o_]):.3f})"
            print(line)
        s_ = nq_t == m; print(f"  n={m:2d} TEST (all 8,007 candidates)                                    " + " ".join(f"{np.quantile(top_t[s_], u):.3f}" for u in QS) + f"   gap to the second candidate: median {np.median((srt[:, -1] - srt[:, -2])[s_]):.3f}")
    s_ = nq_t == 15; print(f"  n=15 TEST                                                             " + " ".join(f"{np.quantile(top_t[s_], u):.3f}" for u in QS))
