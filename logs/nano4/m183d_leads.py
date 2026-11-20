#!/usr/bin/env python
"""Section 4M M183d (2026-11-16, report only): the two leads of M183c.

(1) 20-game questions: which feature predicts (label - pooled expected level), and is it the questions with a repeated
    recognised opponent (a validation-only structure, M183 G)? Standardised coefficients; balanced out-of-fold R^2 with and
    without the questions that hold a repeated opponent; mean miss by the number of anchor games.
(2) handicap-heavy questions (>= 80 % of the games with a handicap): who are they - the target gives (White) or receives
    (Black) the stones, true level, anchor games per game, base / final score, mean signed miss of the base and of the pooled
    posterior - against the other questions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); E1 = float(np.exp(-1.0)); LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402


def main():
    z = np.load(ROOT / "results/r_variants/m183_dump.npz", allow_pickle=True); y, n, pid = z["y"].astype(int), z["n"].astype(int), z["pid"].astype(str); post = z["anat_post"]; logp = z["anat_logp"]; R = z["anat_R"]; Lall = z["anat_Lall"]; L1 = z["anat_L"]; AN = z["anat_AN"].astype(int); fin = z["SQ4"].astype(int); base = z["anat_base"].astype(int)
    lev = np.arange(10.0); sm = lambda a: (e := np.exp(a - a.max(1, keepdims=True))) / e.sum(1, keepdims=True)  # noqa: E731
    ep = post @ lev; eb = np.exp(logp) @ lev; ea = sm(Lall) @ lev; er = sm(R) @ lev; ent = -(post * np.log(np.maximum(post, 1e-12))).sum(1); sc = lambda d: np.where(d == y, 1.0, np.where(np.abs(d - y) == 1, E1, 0.0))  # noqa: E731
    w = np.zeros(len(y))
    for s in sorted(set(n)):
        for c in range(10):
            mm = (n == s) & (y == c); w[mm] = 1.0 / max(int(mm.sum()), 1)
    wm = lambda x, m: float((w * x)[m].sum() / w[m].sum())  # noqa: E731
    # per question: repeated recognised opponents, handicap share, the target's role in the handicap games
    q = np.load(mb7.CACHE / "queries.npz"); Q = int(q["Q"]); row = {k: i for i, k in enumerate(q["qkey"][:Q].astype(str))}; zz = np.load(ROOT / "results/r_variants/m127_SQ4.npz"); st = zz["TV"][:, 0] - zz["TV"][:, 1:5].mean(1); best = zz["best"]; thr = float(np.quantile(st[Q:2 * Q], 0.95))
    hdt = pd.read_csv(ROOT / "results/data/hd_table.csv", usecols=["game_key", "color", "hd"], dtype={"game_key": str, "color": str}).set_index("game_key"); hdv = hdt.hd.astype(float); col = hdt.color
    rep, hs, gives = [], [], []
    for v in mb7.FOLDS:
        _, _, _, qd = mb7.m3.load_fold(v)
        for g in qd["game_keys"]:
            g = list(g); r = np.array([row[k] for k in g if k in row], dtype=np.int64); ok = st[r] >= thr; b_ = best[r][ok]; rep.append(len(b_) - len(set(b_.tolist()))); h = hdv.reindex(g).to_numpy() > 0; hs.append(float(h.mean())); c = col.reindex(g).to_numpy(); gives.append(float((c[h] == "W").mean()) if h.any() else np.nan)
    rep, hs, gives = np.array(rep), np.array(hs), np.array(gives); assert len(rep) == len(y)
    print("(1) 20-game questions: what predicts (label - pooled expected level)")
    m = n == 20; names = ["base - pooled", "anchors - pooled", "look-alike - pooled", "(base - pooled)^2", "|anchors - pooled|", "entropy", "anchor games (x2)", "anchor games (x4)"]; X = np.column_stack([eb - ep, ea - ep, er - ep, (eb - ep) ** 2, np.abs(ea - ep), ent, AN[:, 1], AN[:, 2]])[m]; t = (y - ep)[m]; ww = w[m] / w[m].sum()
    Xs = np.column_stack([np.ones(len(X)), (X - X.mean(0)) / X.std(0)]); A = Xs * ww[:, None]; beta = np.linalg.solve(Xs.T @ A, A.T @ t); print("  standardised coefficients (levels per sd; balanced fit): " + ", ".join(f"{nm} {b:+.3f}" for nm, b in zip(names, beta[1:])))
    for j, nm in enumerate(names[:3]):
        xj = X[:, j]; qs = np.quantile(xj, [0.1, 0.3, 0.7, 0.9]); bq = np.digitize(xj, qs); print(f"  mean miss (label - pooled) by {nm} (five bands, low to high; band mean of the gap | miss): " + "  ".join(f"{float(xj[bq == k].mean()):+.2f} | {float((ww * t)[bq == k].sum() / ww[bq == k].sum()):+.3f}" for k in range(5)))
    up = np.unique(pid); rng = np.random.default_rng(0); fold = dict(zip(up, rng.integers(0, 5, len(up)))); fo = np.array([fold[p] for p in pid])[m]; pred = np.zeros(len(t))
    for k in range(5):
        tr = fo != k; A = Xs[tr] * ww[tr][:, None]; pred[~tr] = Xs[~tr] @ np.linalg.solve(Xs[tr].T @ A + 1e-6 * np.eye(Xs.shape[1]), A.T @ t[tr])
    r2 = lambda sel: 1 - float((ww * (t - pred) ** 2)[sel].sum() / (ww * (t - float((ww * t)[sel].sum() / ww[sel].sum())) ** 2)[sel].sum())  # noqa: E731
    rp = rep[m]; print(f"  questions with a repeated recognised opponent: {float((rp > 0).mean()):.3f} of the 20-game questions; balanced out-of-fold R^2 - all {r2(np.ones(len(t), bool)):+.4f}, with a repeat {r2(rp > 0):+.4f}, without {r2(rp == 0):+.4f}; score of the final answers with / without a repeat {wm(sc(fin), m & (rep > 0)):.3f} / {wm(sc(fin), m & (rep == 0)):.3f}")
    k2 = AN[:, 1]; print("  by anchor games (x2) - mean |miss| and mean signed (label - pooled), base - pooled: " + "  ".join(f"{lo}-{hi}: {wm(np.abs(y - ep), m & (k2 >= lo) & (k2 <= hi)):.3f} / {wm(y - ep, m & (k2 >= lo) & (k2 <= hi)):+.3f} / {wm(eb - ep, m & (k2 >= lo) & (k2 <= hi)):+.3f}" for lo, hi in ((0, 4), (5, 7), (8, 10), (11, 13), (14, 20))))
    # ---------------------------------------------------------------- (2)
    print("\n(2) handicap-heavy questions (>= 80 % of the games with a handicap) against the rest")
    hh = hs >= 0.8; role = np.where(gives >= 0.8, "gives", np.where(gives <= 0.2, "receives", "mixed"))
    for lab, sel in (("the rest", ~hh), ("handicap-heavy", hh), ("  target gives the stones", hh & (role == "gives")), ("  target receives them", hh & (role == "receives")), ("  mixed", hh & (role == "mixed"))):
        if sel.sum() < 100:
            print(f"  {lab}: {int(sel.sum())} questions (too few)"); continue
        print(f"  {lab}: {int(sel.sum())} questions ({float(sel.mean()):.3f}); score base {wm(sc(base), sel):.3f}, final {wm(sc(fin), sel):.3f}; exact base {wm(base == y, sel):.3f} final {wm(fin == y, sel):.3f}; two or more levels off (final) {wm(np.abs(fin - y) >= 2, sel):.3f}; anchor games per game {float((AN[sel, 1] / n[sel]).mean()):.2f}; signed miss of the expected level, base {wm(eb - y, sel):+.2f} pooled {wm(ep - y, sel):+.2f}; sd of the pooled miss {np.sqrt(wm((ep - y) ** 2, sel) - wm(ep - y, sel) ** 2):.2f}; true levels " + " ".join(f"{LV[c]} {float((y[sel] == c).mean()):.2f}" for c in range(10)))
    for lab, sel in (("target gives", hh & (role == "gives")), ("target receives", hh & (role == "receives"))):
        if sel.sum() >= 300:
            print(f"  {lab}, by true level (questions | final score | pooled signed miss): " + "  ".join(f"{LV[c]} {int((sel & (y == c)).sum())}|{wm(sc(fin), sel & (y == c)):.2f}|{wm(ep - y, sel & (y == c)):+.2f}" for c in range(10) if (sel & (y == c)).sum() >= 30))
    tg = pd.read_csv(ROOT / "results/data/test_games.csv", dtype=str); tg["h"] = tg.hd.astype(float) > 0; g = tg.groupby("question_id"); th = g.h.mean(); tgv = tg[tg.h].groupby("question_id").color.agg(lambda c: float((c == "W").mean())).reindex(th.index); heavy = th >= 0.8
    print(f"  test questions: handicap-heavy {int(heavy.sum())} of {len(th)} ({float(heavy.mean()):.3f}); among them the target gives {int((tgv[heavy] >= 0.8).sum())}, receives {int((tgv[heavy] <= 0.2).sum())}, mixed {int(((tgv[heavy] > 0.2) & (tgv[heavy] < 0.8)).sum())}")
    sub = pd.read_csv(ROOT / "submission_rank_ct112.csv", dtype=str).set_index("question_id").pred_rank.reindex(th.index)
    for lab, sel in (("target gives", heavy & (tgv >= 0.8)), ("target receives", heavy & (tgv <= 0.2)), ("the rest", ~heavy)):
        print(f"    ct112's answers, {lab} ({int(sel.sum())}): " + " ".join(f"{k} {v}" for k, v in sub[sel].value_counts().reindex(LV).fillna(0).astype(int).items()))


if __name__ == "__main__":
    main()
