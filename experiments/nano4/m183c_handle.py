#!/usr/bin/env python
"""Section 4M M183c (2026-11-16, report only): two follow-ups of M183b.

(1) M183b (c) found an out-of-fold R^2 of 0.08 for "label minus pooled expected level" - but unweighted, and with the pooled
    level itself among the features: the pooled posterior is fitted for equal class weights while the validation players are
    not equally many per level, so part of that is the class composition again. Redo with the class-balanced weights of the
    score (weighted fit, weighted R^2, weighted score change), for the channel gaps alone and with the pooled level added;
    all three sizes.
(2) M183b (b): players with >= 50 % handicap games carry a much larger player part (sd 0.86 against 0.53). How do questions
    split by their own share of handicap games, in validation and in the test questions (label-free count), and what is the
    score by that share?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); E1 = float(np.exp(-1.0))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]


def main():
    z = np.load(ROOT / "results/r_variants/m183_dump.npz", allow_pickle=True); y, n, pid = z["y"].astype(int), z["n"].astype(int), z["pid"].astype(str); post = z["anat_post"]; logp = z["anat_logp"]; R = z["anat_R"]; Lall = z["anat_Lall"]; AN = z["anat_AN"].astype(int); fin = z["SQ4"].astype(int); base = z["anat_base"].astype(int)
    lev = np.arange(10.0); sm = lambda a: (e := np.exp(a - a.max(1, keepdims=True))) / e.sum(1, keepdims=True)  # noqa: E731
    ep = post @ lev; eb = np.exp(logp) @ lev; ea = sm(Lall) @ lev; er = sm(R) @ lev; ent = -(post * np.log(np.maximum(post, 1e-12))).sum(1); SMx = np.eye(10) + E1 * (np.eye(10, k=1) + np.eye(10, k=-1)); sc = lambda d, yy: np.where(d == yy, 1.0, np.where(np.abs(d - yy) == 1, E1, 0.0))  # noqa: E731
    up = np.unique(pid); rng = np.random.default_rng(0); fold = dict(zip(up, rng.integers(0, 5, len(up)))); fo = np.array([fold[p] for p in pid])
    print("(1) does anything a question carries predict (label - pooled expected level)? class-balanced weights, out of fold over players")
    gaps = np.column_stack([eb - ep, ea - ep, er - ep, (eb - ep) ** 2, np.abs(ea - ep), ent, AN[:, 1], AN[:, 2]]); withp = np.column_stack([gaps, ep, ep ** 2])
    for s in sorted(set(n)):
        m = n == s; w = np.zeros(m.sum()); yy = y[m]
        for c in range(10):
            w[yy == c] = 1.0 / max(int((yy == c).sum()), 1)
        w /= w.sum(); t = (y - ep)[m]; P = post[m]; d0 = (P @ SMx).argmax(1); s0 = sc(d0, yy)
        for lab, X in (("channel gaps, entropy, anchor counts", gaps[m]), ("the same + the pooled level", withp[m])):
            Xs = np.column_stack([np.ones(len(X)), (X - X.mean(0)) / X.std(0)]); pred = np.zeros(len(t)); pu = np.zeros(len(t))
            for k in range(5):
                tr = fo[m] != k; A = Xs[tr] * w[tr][:, None]; beta = np.linalg.solve(Xs[tr].T @ A + 1e-6 * np.eye(Xs.shape[1]), A.T @ t[tr]); pred[~tr] = Xs[~tr] @ beta
                beta_u = np.linalg.lstsq(Xs[tr], t[tr], rcond=None)[0]; pu[~tr] = Xs[~tr] @ beta_u
            tm = float((w * t).sum()); r2 = 1 - float((w * (t - pred) ** 2).sum() / (w * (t - tm) ** 2).sum()); r2u = 1 - float(((t - pu) ** 2).mean() / ((t - t.mean()) ** 2).mean()); out = []
            for lam in (0.5, 1.0, 2.0):
                Ps = P * np.exp(lam * pred[:, None] * (lev[None] - (P @ lev)[:, None])); Ps /= Ps.sum(1, keepdims=True); d1 = (Ps @ SMx).argmax(1); out.append(f"{lam}: {float((w * (sc(d1, yy) - s0)).sum()):+.4f} ({float((d1 != d0).mean()):.3f} changed)")
            print(f"  {s:2d} games, {lab}: balanced R^2 {r2:+.4f} (unweighted fit and R^2 {r2u:+.4f}); balanced score change when the posterior is tilted by the prediction, by strength - " + "  ".join(out))
    # ---------------------------------------------------------------- (2)
    print("\n(2) questions by their own share of handicap games")
    import m137b_build as mb7
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", usecols=["game_key", "hd"], dtype={"game_key": str}).set_index("game_key").hd.astype(float); hs = []
    for v in mb7.FOLDS:
        yv, nv, pv, qd = mb7.m3.load_fold(v); hs += [float((hd.reindex(list(g)).to_numpy() > 0).mean()) for g in qd["game_keys"]]
    hs = np.array(hs); assert len(hs) == len(y); edges = [0, 1e-9, 0.2, 0.5, 0.8, 1.01]; names = ["none", "(0, 0.2)", "[0.2, 0.5)", "[0.5, 0.8)", "[0.8, 1]"]
    tg = pd.read_csv(ROOT / "results/data/test_games.csv", dtype=str); tg["h"] = tg.hd.astype(float) > 0; th = tg.groupby("question_id").h.mean(); tn = tg.groupby("question_id").size()
    w = np.zeros(len(y))
    for s in sorted(set(n)):
        for c in range(10):
            mm = (n == s) & (y == c); w[mm] = 1.0 / max(int(mm.sum()), 1)
    wm = lambda x, m: float((w * x)[m].sum() / w[m].sum())  # noqa: E731
    print(f"  share of games with a handicap: validation {float((hd.reindex(np.concatenate([list(g) for v in mb7.FOLDS for g in mb7.m3.load_fold(v)[3]['game_keys']])).to_numpy() > 0).mean()):.3f}, test {float(tg.h.mean()):.3f}")
    for s in sorted(set(n)):
        m = n == s; row = []
        for i in range(len(names)):
            b_ = m & (hs >= edges[i]) & (hs < edges[i + 1]); tsel = (tn == s) & (th >= edges[i]) & (th < edges[i + 1])
            row.append(f"{names[i]}: validation {float(b_.sum() / m.sum()):.3f} / test {float(tsel.sum() / max(int((tn == s).sum()), 1)):.3f}" + (f", score base {wm(sc(base, y), b_):.3f} final {wm(sc(fin, y), b_):.3f}" if b_.sum() >= 150 else ""))
        print(f"  {s:2d} games - " + " | ".join(row))
    t15 = tn == 15; print("  15 games (test only) - " + " | ".join(f"{names[i]}: {float(((th >= edges[i]) & (th < edges[i + 1]) & t15).sum() / max(int(t15.sum()), 1)):.3f}" for i in range(len(names))))


if __name__ == "__main__":
    main()
