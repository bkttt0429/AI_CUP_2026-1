#!/usr/bin/env python
"""Section 4M M185 pilot (2026-11-16, report only): games BETWEEN two test players - do they exist, and can they be read?

A game is filed under one owner only. If the opponent of a game in question B is the target of another question A, the two
unknown levels are tied by the same matchmaking kernel that ties a player to a recognised training opponent - one more
reading for both questions, from data the anchors never touch (the opponent is not a training player).
Pilot, label-free on the test side: prototype of every test question = normalised sum of the target-side vectors of its
games (identity networks of the second production set S7t, two seeds, and S6at; all-players networks); queries = the
opponent-side vectors of the 10,000 test games, scored against the prototypes of the OTHER 799 questions; null = the
target-side vectors of the same games against the other questions' prototypes (a different person for certain). Scores are
standardised over the candidates per query and averaged over the three networks; statistic = first minus the mean of the
2nd-5th. Reported: excess of the opponent side over the null (a lower bound of the share of test games played against
another test player), and - label-free check - for the matched pairs, how close ct112's answers of the two questions are
(random pairs for reference; handicap games: the direction of the one-level gap).
The same on the validation side with labels: the players of a fold as each other's candidates (their 20-game question,
cross-fit networks), where the label differences of the matched pairs can be read directly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402


def zs(S):
    mu = S.mean(1, keepdims=True); sd = S.std(1, keepdims=True); return (S - mu) / np.maximum(sd, 1e-9)


def stat(F):
    top = -np.partition(-F, 5, axis=1)[:, :5]; top.sort(axis=1); top = top[:, ::-1]; return top[:, 0] - top[:, 1:5].mean(1), F.argmax(1)


def match(vecs, q_opp, q_tgt, owner, n_owner):
    """vecs: list of [nq, 128] arrays (one per network); q_opp / q_tgt: query indices of the games; owner: question index of each game."""
    Fo = Fn = 0
    for V in vecs:
        T = np.zeros((n_owner, V.shape[1]), np.float64); np.add.at(T, owner, V[q_tgt].astype(np.float64)); T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
        So = V[q_opp].astype(np.float64) @ T.T; Sn = V[q_tgt].astype(np.float64) @ T.T; big = -1e9; So[np.arange(len(owner)), owner] = np.nan; Sn[np.arange(len(owner)), owner] = np.nan
        f = lambda S: np.nan_to_num((S - np.nanmean(S, 1, keepdims=True)) / np.maximum(np.nanstd(S, 1, keepdims=True), 1e-9), nan=big)  # noqa: E731
        Fo = Fo + f(So); Fn = Fn + f(Sn)
    Fo /= len(vecs); Fn /= len(vecs); Fo[np.arange(len(owner)), owner] = big; Fn[np.arange(len(owner)), owner] = big
    return stat(Fo), stat(Fn)


def main():
    q = np.load(mb7.CACHE / "queries.npz"); Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); vecs = []
    for name in ("S7t", "S6at"):
        V = np.load(mb7.CACHE / f"var_{name}.npz")["qvec"]; vecs += [V[m] for m in range(V.shape[0])]
    print(f"identity vectors: {len(vecs)} networks (S7t x2, S6at)", flush=True)
    # ---------------------------------------------------------------- test side
    tg = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((ROOT / "results/ct_submission/vec").glob("games_s*_p*.csv"))], ignore_index=True); qo = np.arange(2 * Q + H, 2 * Q + H + T); qt = np.arange(2 * Q + H + T, 2 * Q + H + 2 * T); assert (q["side"][qo] == 1).all() and (q["side"][qt] == 0).all() and np.array_equal(q["qtest"][qo], q["qtest"][qt])
    rows = q["qtest"][qo].astype(int); qid = tg.question_id.to_numpy()[rows]; gkey = tg.game_key.to_numpy()[rows]; uq, owner = np.unique(qid, return_inverse=True)
    meta = pd.read_csv(ROOT / "results/data/test_games.csv", dtype=str); meta["key"] = meta.question_id + "#" + meta.i; tg["i"] = tg.groupby("question_id").cumcount().astype(str)
    (so, bo), (sn, bn) = match(vecs, qo, qt, owner, len(uq)); ans = pd.read_csv(ROOT / "submission_rank_ct112.csv", dtype=str).set_index("question_id").pred_rank.map({l: i for i, l in enumerate(LV)}).reindex(uq).to_numpy()
    rng = np.random.default_rng(0); rp = rng.integers(0, len(uq), (200000, 2)); rp = rp[rp[:, 0] != rp[:, 1]]; rd = np.abs(ans[rp[:, 0]] - ans[rp[:, 1]]); print(f"test side: {T} games, {len(uq)} questions; random pairs of questions: ct112's answers equal {float((rd == 0).mean()):.3f}, within one level {float((rd <= 1).mean()):.3f}")
    grid = np.quantile(sn, np.linspace(0.5, 0.9995, 800)); sup = max(float((so > t).mean() - (sn > t).mean()) for t in grid); print(f"  largest excess of the opponent side over the null: {sup:.4f} of the games (= about {sup * T:.0f} games)")
    for qq in (0.90, 0.99, 0.999, 0.9999):
        thr = float(np.quantile(sn, qq)); m = so > thr; d = np.abs(ans[owner[m]] - ans[bo[m]]); mn = sn > thr; dn = np.abs(ans[owner[mn]] - ans[bn[mn]])
        print(f"  above the null's q{qq * 100:g} ({thr:.2f}): opponent side {int(m.sum())} games ({float(m.mean()):.4f}; nominal {1 - qq:.4f}) -> excess {float(m.mean()) - (1 - qq):+.4f}; answers of the two questions equal {float((d == 0).mean()) if m.any() else float('nan'):.3f}, within one {float((d <= 1).mean()) if m.any() else float('nan'):.3f} (null matches above the same line: {int(mn.sum())}, equal {float((dn == 0).mean()) if mn.any() else float('nan'):.3f}, within one {float((dn <= 1).mean()) if mn.any() else float('nan'):.3f}); questions touched {len(set(owner[m].tolist()) | set(bo[m].tolist()))}")
    thr = float(np.quantile(sn, 0.999)); m = np.flatnonzero(so > thr); print("  strongest matches (game's question -> matched question, statistic, ct112's answers):")
    for i in m[np.argsort(-so[m])][:12]:
        print(f"    {qid[i]} ({LV[ans[owner[i]]]}) -> {uq[bo[i]]} ({LV[ans[bo[i]]]}), {so[i]:.2f}")
    np.savez(ROOT / "results/r_variants/m185_tt_test.npz", stat_opp=so, best_opp=bo, stat_null=sn, best_null=bn, owner=owner, qid=uq, game_qid=qid)
    # ---------------------------------------------------------------- validation side (labels)
    row = {k: i for i, k in enumerate(q["qkey"][:Q].astype(str))}; hd = pd.read_csv(ROOT / "results/data/hd_table.csv", usecols=["game_key", "hd"], dtype={"game_key": str}).set_index("game_key").hd.astype(float); agg = {}
    for v in mb7.FOLDS:
        y, n, pid, qd = mb7.m3.load_fold(v); gk = list(qd["game_keys"]); sel = np.flatnonzero(np.asarray(n) == 20); r, ow = [], []
        for j, qi in enumerate(sel):
            ks = [k for k in gk[qi] if k in row]; r += [row[k] for k in ks]; ow += [j] * len(ks)
        r, ow = np.array(r, dtype=np.int64), np.array(ow, dtype=np.int64); yl = np.asarray(y)[sel]; (so_, bo_), (sn_, bn_) = match(vecs, r, Q + r, ow, len(sel)); even = hd.reindex(q["qkey"][:Q].astype(str)[r]).to_numpy() == 0
        for nm, val in (("so", so_), ("sn", sn_), ("d", yl[bo_] - yl[ow]), ("dn", yl[bn_] - yl[ow]), ("even", even), ("nplayers", np.full(len(r), len(sel)))):
            agg.setdefault(nm, []).append(val)
    A = {k: np.concatenate(v) for k, v in agg.items()}; so_, sn_, d, dn, even = A["so"], A["sn"], A["d"], A["dn"], A["even"]
    grid = np.quantile(sn_, np.linspace(0.5, 0.9995, 800)); sup = max(float((so_ > t).mean() - (sn_ > t).mean()) for t in grid); rdv = float(np.mean(np.abs(dn) <= 1))
    print(f"\nvalidation side: {len(so_)} games of the folds' 20-game questions (candidates = the other players of the same fold, about {int(np.median(A['nplayers']))}); largest excess of the opponent side over the null {sup:.4f}")
    for qq in (0.90, 0.99, 0.999, 0.9999):
        thr = float(np.quantile(sn_, qq)); m = so_ > thr; me = m & even; mn = sn_ > thr
        print(f"  above the null's q{qq * 100:g} ({thr:.2f}): opponent side {int(m.sum())} games ({float(m.mean()):.4f}) -> excess {float(m.mean()) - (1 - qq):+.4f}; even games: matched player's label minus the owner's = -1 / 0 / +1: {float((d[me] == -1).mean()):.3f} / {float((d[me] == 0).mean()):.3f} / {float((d[me] == 1).mean()):.3f}, within one {float((np.abs(d[me]) <= 1).mean()):.3f} (null matches: same level {float((dn[mn] == 0).mean()) if mn.any() else float('nan'):.3f}, within one {float((np.abs(dn[mn]) <= 1).mean()) if mn.any() else float('nan'):.3f}; all null first candidates within one {rdv:.3f})")


if __name__ == "__main__":
    main()
