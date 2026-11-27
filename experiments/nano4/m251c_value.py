#!/usr/bin/env python
"""M251c (2026-11-19), report only: twins inside one validation fold (both unseen by the fold's networks, neither a candidate
of the other's anchors or look-alike reading - the situation of two test questions of one person): their label relation and
what knowing the twin's label is worth on top of the ct112 rule. (design doc M251c, registered before any reading)

Twins = two players of a fold whose games (the 20 scored games, own side) vote for each other (M251b's game-by-game votes),
by the smaller of the two vote shares. For every twin's validation questions (5 / 10 / 20 games), against the ct112 rule:
  T0  answer the twin's label
  T1  pooled posterior x K(level - twin's label), K from the pairs of the other folds, expected-score decision
  T2  pooled posterior x sum_a twin's posterior(a) K(level - a)   (the twin's truth unknown)

  python -I logs/nano4/m251c_value.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402
import m251b_votes as mv  # noqa: E402

CACHE, SM, b = mb7.CACHE, mb7.SM, mb7.b; RNG = np.random.default_rng(2513); BANDS = (0.5, 0.7, 0.85)


def ktab(d, w=0.5):
    """Distribution of the label difference (twin - own) on -9..9 with a small floor."""
    c = np.full(19, w * 0.02); np.add.at(c, np.clip(np.asarray(d, int), -9, 9) + 9, 1.0); return c / c.sum()


def tilt(P, K, centre):
    """P [10] x K(level - centre[a]) summed over the weights centre [10] (a one-hot for a known label)."""
    L = np.array([sum(centre[a] * K[(lv - a) + 9] for a in range(10)) for lv in range(10)]); out = P * L; return out / out.sum()


def main():
    q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); vecs = []
    for name in mv.NETS:
        V = np.load(CACHE / f"var_{name}.npz")["qvec"]; vecs += [V[m] for m in range(V.shape[0])]
    dv = np.load(ROOT / "results/r_variants/m82_val.npz"); pidv = dv["pid"].astype(str); foldv = dv["fold"].astype(int); yv = dv["y"].astype(int); assert np.array_equal(q["qkey"][:Q].astype(str), dv["key"].astype(str))
    z = np.load(ROOT / "results/r_variants/m251_val_post.npz"); zy, zn, zp, zpost, zdec = z["y"].astype(int), z["n"].astype(int), z["pid"].astype(str), z["post"].astype(np.float64), z["dec"].astype(int)
    qrows = {}
    for i, p_ in enumerate(zp):
        qrows.setdefault(p_, []).append(i)
    print(f"validation questions with a pooled posterior: {len(zy)}; ct112 rule, plain mean score {float(SM[zdec, zy].mean()):.4f}", flush=True)
    pairs = []  # (fold, a, b, min share, label a, label b)
    for f in sorted(set(foldv.tolist())):
        idx = np.flatnonzero(foldv == f); u, names = pd.factorize(pidv[idx]); k = np.bincount(u); keep = k[u] >= 10; idx, (u, names) = idx[keep], pd.factorize(pidv[idx][keep]); nu = len(names); lab = np.zeros(nu, int); lab[u] = yv[idx]
        F = mv.scores(vecs, Q + idx, Q + idx, u, nu, u); W, fav, share, mut, _, _, kk = mv.votes(F, u, nu)
        for a in range(nu):
            if mut[a] and a < fav[a]:
                pairs.append((f, str(names[a]), str(names[fav[a]]), float(min(share[a], share[fav[a]])), int(lab[a]), int(lab[fav[a]])))
        print(f"fold {f}: {nu} players with >= 10 scored games; mutual pairs {sum(1 for x in pairs if x[0] == f)}; share of a player's games for the most voted player 50 / 90 / 99 %: " + " / ".join(f"{float(np.quantile(share, q_)):.2f}" for q_ in (0.5, 0.9, 0.99)), flush=True)
    rep = {}
    print("\n== label relation of the mutual pairs by the smaller vote share (pairs | same level | one apart | two or more | mean |difference|); random pairs of players: same 0.115, one apart 0.183")
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
        d = np.array([x[5] - x[4] for x in pairs if lo <= x[3] < hi])
        if len(d):
            print(f"   [{lo:.2f}, {hi:.2f}): {len(d):4d} | {float((d == 0).mean()):.3f} | {float((np.abs(d) == 1).mean()):.3f} | {float((np.abs(d) >= 2).mean()):.3f} | {float(np.abs(d).mean()):.2f}")
            rep[f"band_{lo:.2f}"] = {"pairs": int(len(d)), "same": float((d == 0).mean()), "adjacent": float((np.abs(d) == 1).mean())}
    print("\n== value of the twin's label for the twin's validation questions, against the ct112 rule (mean score; paired difference [95 %, bootstrap over pairs])")
    for lo in BANDS:
        sel = [x for x in pairs if x[3] >= lo]; rows = []  # (pair index, size, current, T0, T1, T2, base exact, T1 exact)
        for j, (f, a, b_, sh, la, lb) in enumerate(sel):
            dk = [x[5] - x[4] for x in sel if x[0] != f]; K = ktab(dk + [-x for x in dk])
            for me, tw, lt in ((a, b_, lb), (b_, a, la)):
                tq = [i for i in qrows.get(tw, []) if zn[i] == 20] or qrows.get(tw, []); Ptw = zpost[tq].mean(0) if tq else np.full(10, 0.1)
                for i in qrows.get(me, []):
                    one = np.zeros(10); one[lt] = 1.0; d1 = int(np.asarray(b.decide(tilt(zpost[i], K, one)[None, :]), int)[0]); d2 = int(np.asarray(b.decide(tilt(zpost[i], K, Ptw)[None, :]), int)[0])
                    rows.append((j, zn[i], SM[zdec[i], zy[i]], SM[lt, zy[i]], SM[d1, zy[i]], SM[d2, zy[i]], float(zdec[i] == zy[i]), float(d1 == zy[i]), float(d1 != zdec[i]), float(d2 != zdec[i])))
        R = np.array(rows)
        if not len(R):
            continue
        dk = [x[5] - x[4] for x in sel]; print(f"   smaller vote share >= {lo}: {len(sel)} pairs, {len(R)} questions; twins' labels same {float(np.mean(np.array(dk) == 0)):.3f}, one apart {float(np.mean(np.abs(dk) == 1)):.3f}")
        out = {}
        for nm, m_ in (("all sizes", np.ones(len(R), bool)), ("5 games", R[:, 1] == 5), ("10 games", R[:, 1] == 10), ("20 games", R[:, 1] == 20)):
            S = R[m_]; pj = S[:, 0].astype(int); up = np.unique(pj); line = f"      {nm:9s} ({len(S)} questions): ct112 rule {S[:, 2].mean():.4f} (exact {S[:, 6].mean():.3f})"
            for col, lab_ in ((3, "T0 twin's label"), (4, "T1 posterior x K"), (5, "T2 twin unknown")):
                diff = S[:, col] - S[:, 2]; bs = []
                for _ in range(1000):
                    pick = RNG.choice(up, len(up)); w = np.bincount(np.searchsorted(up, pick), minlength=len(up))[np.searchsorted(up, pj)]; bs.append(float((diff * w).sum() / max(w.sum(), 1)))
                lo_, hi_ = np.percentile(bs, [2.5, 97.5]); line += f" | {lab_} {S[:, col].mean():.4f} ({diff.mean():+.4f} [{lo_:+.4f}, {hi_:+.4f}])"; out[lab_] = {"score": float(S[:, col].mean()), "diff": float(diff.mean()), "ci": [float(lo_), float(hi_)]}
            print(line + f" | T1 exact {S[:, 7].mean():.3f}, answers changed by T1 {S[:, 8].mean():.3f}, by T2 {S[:, 9].mean():.3f}", flush=True); rep[f"value_{lo}_{nm}"] = {"n": int(len(S)), "rule": float(S[:, 2].mean()), **out}
    (ROOT / "results/r_variants/m251c_value.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m251c_value.json")


if __name__ == "__main__":
    main()
