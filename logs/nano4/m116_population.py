"""Section 4M M116 (2026-11-14, report only, CPU): do the LESS ACTIVE training players look like the test players?
Known oddities of the test set: 22.8% of its games are HD1 (training without HD >= 2: 17.7%), its A players look like 1D to the
model (ct13: eight of eight 1D->A flips right; validation precision 0.47), fewer opponents are recognised (density ratio 0.85).
Validation players by their number of games in the data set: share of HD1 games, level composition, base and final score, the two
A / 1D confusions, share of recognised opponents - next to the same quantities of the test set where they exist.
  python -I logs/nano4/m116_population.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m87_pi0 as m87  # noqa: E402
import m111_label_readings as m1  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402

b, s2, mb, ES, FOLDS, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.SM


def main():
    V, T, key_pid, used = b.sources(["5e", "SP2e"]); z = V["SP2e"]; t = T["SP2e"]; row = {k: i for i, k in enumerate(z["key"])}; ones = np.ones(len(z["y"]), bool); edges, cal, om = m87.corrected(z, ones); _, _, band, g = s2.fit(z, ones)
    pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); ngames = pf.n_games.astype(int); rows = []
    for v in FOLDS:
        tr = z["fold"] != v; edg, cl, _ = m87.corrected(z, tr); _, _, bnd, gg = s2.fit(z, tr); Kc, P0c, R = m1.full_tables(z, tr, cl, bnd, gg); y, n, pid, qd = m3.load_fold(v); y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid).astype(str); Q = len(y)
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(qd["game_keys"])]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in qd["game_keys"]]).astype(int); hd1 = np.bincount(qi, weights=(z["hd"][gi] == 1), minlength=Q) / np.maximum(np.bincount(qi, minlength=Q), 1); rec = np.bincount(qi, weights=(band[gi] >= 3) & (g[gi] >= 0), minlength=Q) / np.maximum(np.bincount(qi, minlength=Q), 1)
        ok = g[gi] >= 0; kk, bb, lab = g[gi][ok], band[gi][ok], z["opp_lab"][gi][ok]; a = np.array([cl[k]["a"][x] for k, x in zip(kk, bb)]); L = np.zeros((Q, 10)); np.add.at(L, qi[ok], np.log((1 - a)[:, None] + a[:, None] * R[kk, :, lab])); L *= np.array([b.TN[int(m)] for m in n])[:, None]
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); d0, dA, _, _, _ = mb.decisions(P, L, 0.5, 3.0, 0.6)
        rows.append(pd.DataFrame({"pid": pid, "y": y, "n": n, "hd1": hd1, "rec": rec, "d0": d0, "dA": dA, "s0": SM[d0, y], "sA": SM[dA, y], "ng": ngames.reindex(pid).to_numpy()}))
    D = pd.concat(rows); D = D[D.n == 20]; qs = [0, 40, 70, 110, 170, 260, 10 ** 6]; D["grp"] = pd.cut(D.ng, qs, labels=[f"{qs[i] + 1}-{qs[i + 1] if qs[i + 1] < 10 ** 6 else '..'}" for i in range(len(qs) - 1)])
    print("validation players (their 20-game question) by the number of games they have in the data set:")
    print("  games in data | players | HD1 share of the games | recognised opponents (>= q90) | base score -> with evidence | true A called 1D (base -> evidence) | true 1D called A | share of A / of 1D among the group's players")
    for gname, x in D.groupby("grp", observed=True):
        A, O = x[x.y == 3], x[x.y == 4]
        print(f"  {gname:>9s} | {len(x):5d} | {x.hd1.mean():.3f} | {x.rec.mean():.3f} | {x.s0.mean():.3f} -> {x.sA.mean():.3f} | {float((A.d0 == 4).mean()) if len(A) else float('nan'):.3f} -> {float((A.dA == 4).mean()) if len(A) else float('nan'):.3f} ({len(A)}) | {float((O.d0 == 3).mean()) if len(O) else float('nan'):.3f} -> {float((O.dA == 3).mean()) if len(O) else float('nan'):.3f} ({len(O)}) | {len(A) / len(x):.3f} / {len(O) / len(x):.3f}")
    print("  by level, mean number of games in the data set and HD1 share (20-game questions): " + "  ".join(f"{LEVELS[c]} {D[D.y == c].ng.mean():.0f}/{D[D.y == c].hd1.mean():.3f}" for c in range(10)))
    tb = np.where(~t["target_black"], np.searchsorted(edges[True], t["opp_m"]), np.searchsorted(edges[False], t["opp_m"])); tg = s2.group_of(t["hd"], t["target_black"]); sub = pd.read_csv(ROOT / "submission_rank_ct75u.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); pl = sub.reindex(t["qid"]).to_numpy()
    print(f"test set: HD1 share of the games {float((t['hd'] == 1).mean()):.3f}; recognised opponents (>= q90) {float(((tb >= 3) & (tg >= 0)).mean()):.3f}; HD1 share by ct75u's level: " + "  ".join(f"{LEVELS[c]} {float((t['hd'][pl == c] == 1).mean()):.3f}" for c in range(10)))
    print("validation, all 20-game questions: HD1 share by TRUE level: " + "  ".join(f"{LEVELS[c]} {D[D.y == c].hd1.mean():.3f}" for c in range(10)) + " | by the final decision: " + "  ".join(f"{LEVELS[c]} {D[D.dA == c].hd1.mean():.3f}" for c in range(10)))


if __name__ == "__main__":
    main()
