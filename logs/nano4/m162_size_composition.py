#!/usr/bin/env python
"""Section 4M M162 (2026-11-15, report only): does the level composition of the test questions depend on the question size?
(design doc M162)

The test holds 200 questions of each size 5 / 10 / 15 / 20. If the size reflects the player's total number of games, every size
has its own composition (training players with <= 25 games: 74% kyu, 1.3% 1D; all players: 54% kyu), while the pipeline uses
one set of pushes for all sizes.
(1) training players: level composition by the total number of games;
(2) calibration: Saerens' EM on class-balanced posteriors (the base ensemble's validation posteriors), questions resampled to a
    known composition per size;
(3) test: EM composition by size (base posterior; the learned pool's posterior as a second reading), question bootstrap;
(4) homogeneity: G = log-likelihood gain of one EM composition per size over a common one; null from validation questions
    resampled with a common composition (the test's overall EM estimate), 200 per size (size 15 = half 10, half 20); power under
    "composition = the activity group of the size";
(5) log-likelihood of fixed hypotheses per size: balanced, by players, players with <= 25 games, total games in [n, n + 4],
    total games >= n.

  python -I logs/nano4/m162_size_composition.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402

m3, s2, ES, FOLDS, b = mb7.m3, mb7.s2, mb7.ES, mb7.FOLDS, mb7.b; RV = ROOT / "results/r_variants"; SZ = (5, 10, 15, 20)


def em_prior(P, iters=500):
    pi = np.full(10, 0.1)
    for _ in range(iters):
        r = P * pi; r /= r.sum(1, keepdims=True); new = r.mean(0)
        if np.abs(new - pi).max() < 1e-8:
            return new
        pi = new
    return pi


def loglik(P, pi):
    return float(np.log((P * (pi / 0.1)).sum(1)).sum())


def desc(pi):
    return f"kyu {pi[:4].sum():.3f} (A {pi[3]:.3f}), 1D {pi[4]:.3f}, 2D-6D {pi[5:].sum():.3f}, mean level {float((pi * np.arange(10)).sum()):.2f}"


def gstat(groups):
    """Gain of one EM composition per group over a common one."""
    allp = np.concatenate(groups); pc = em_prior(allp); return float(sum(loglik(g, em_prior(g)) - loglik(g, pc) for g in groups))


def main():
    rng = np.random.default_rng(162); rep = {}
    # ---------------------------------------------------------------- (1) training players by activity
    pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str); lv = pf["rank"].map(b.RANK).to_numpy().astype(int); ng = pf.n_games.astype(int).to_numpy()
    comp = lambda m: np.bincount(lv[m], minlength=10) / max(int(m.sum()), 1)  # noqa: E731
    print(f"training players {len(lv)}; total games: min {ng.min()}, quartiles " + " / ".join(f"{x:.0f}" for x in np.quantile(ng, [0.1, 0.25, 0.5, 0.75, 0.9])))
    H = {"balanced": np.full(10, 0.1), "by players": comp(ng > 0), "players with <= 25 games": comp(ng <= 25)}
    for lo, hi in ((5, 9), (10, 14), (15, 19), (20, 24), (25, 39), (40, 99), (100, 10 ** 9)):
        m = (ng >= lo) & (ng <= hi); print(f"   games {lo:3d}-{'' if hi > 10 ** 8 else hi:<4}: {int(m.sum()):5d} players; {desc(comp(m))}")
    for n in SZ:
        H[f"exact {n}"] = comp((ng >= n) & (ng <= n + 4)); H[f"min {n}"] = comp(ng >= n); print(f"   games >= {n:2d}: {int((ng >= n).sum()):5d} players; {desc(H[f'min {n}'])}")
    rep["hyp"] = {k: v.tolist() for k, v in H.items()}
    # ---------------------------------------------------------------- validation posteriors (base ensemble, class-balanced)
    PV, YV, NV = [], [], []
    for v in FOLDS:
        y, n, pid, qd = m3.load_fold(v); PV.append(np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0)); YV.append(np.asarray(y)); NV.append(np.asarray(n))
    PV, YV, NV = np.concatenate(PV), np.concatenate(YV), np.concatenate(NV); PV /= PV.sum(1, keepdims=True); pool = {(s, c): np.flatnonzero((NV == s) & (YV == c)) for s in (5, 10, 20) for c in range(10)}
    print(f"\nvalidation questions {len(YV)} (sizes " + ", ".join(f"{s}: {int((NV == s).sum())}" for s in (5, 10, 20)) + ")")

    def draw(size, pi, k):
        """k validation questions of this size with composition pi (size 15: half from 10, half from 20)."""
        cls = rng.choice(10, k, p=pi / pi.sum()); src = np.where(rng.random(k) < 0.5, 10, 20) if size == 15 else np.full(k, size)
        return PV[[pool[(int(s), int(c))][rng.integers(len(pool[(int(s), int(c))]))] for s, c in zip(src, cls)]]
    # (2) calibration of the estimator
    print("(2) calibration: EM estimate from 200 resampled validation questions (mean of 100 draws), per size, against the composition drawn")
    for hn in ("balanced", "by players", "players with <= 25 games"):
        for s in SZ:
            est = np.mean([em_prior(draw(s, H[hn], 200)) for _ in range(100)], axis=0); sd = np.std([em_prior(draw(s, H[hn], 200))[:4].sum() for _ in range(100)])
            print(f"   {hn:26s} size {s:2d}: true kyu {H[hn][:4].sum():.3f}, A {H[hn][3]:.3f}, 1D {H[hn][4]:.3f} | estimated kyu {est[:4].sum():.3f} (sd of one estimate {sd:.3f}), A {est[3]:.3f}, 1D {est[4]:.3f}")
    # ---------------------------------------------------------------- (3) the test set
    z = np.load(RV / "m137_test_posterior.npz", allow_pickle=True); qid = z["qid"].astype(str); tn, tpool = {}, {}
    for nm in ("public", "private"):
        t = pd.read_csv(ROOT / f"data/tests/rank_prediction_test_{nm}.csv", usecols=["question_id", "num_games"], dtype=str)
        tn.update(zip(t.question_id, t.num_games.astype(int))); tpool.update({k: nm for k in t.question_id})
    n_t = np.array([tn[k] for k in qid]); pl_t = np.array([tpool[k] for k in qid]); print(f"\ntest questions {len(qid)}: " + ", ".join(f"size {s}: {int((n_t == s).sum())} (Public {int(((n_t == s) & (pl_t == 'public')).sum())})" for s in SZ))
    sub = pd.read_csv(ROOT / "submission_rank_ct106.csv", dtype=str); ans = sub.set_index(sub.columns[0])[sub.columns[1]].reindex(qid).map(b.RANK).to_numpy().astype(int)
    for nm in ("base", "learned"):
        P = z[nm].astype(np.float64); P /= P.sum(1, keepdims=True); print(f"(3) EM composition from the {nm} posterior")
        alle = em_prior(P); print(f"   all sizes        : {desc(alle)}"); rep[f"{nm}_all"] = alle.tolist()
        for s in SZ:
            m = n_t == s; e = em_prior(P[m]); bs = np.array([em_prior(P[m][rng.integers(0, m.sum(), m.sum())]) for _ in range(300)]); ky = bs[:, :4].sum(1); ml = (bs * np.arange(10)).sum(1)
            print(f"   size {s:2d} ({int(m.sum())} q): {desc(e)}; kyu 90% interval [{np.quantile(ky, 0.05):.3f}, {np.quantile(ky, 0.95):.3f}], mean level [{np.quantile(ml, 0.05):.2f}, {np.quantile(ml, 0.95):.2f}] | Public {desc(em_prior(P[m & (pl_t == 'public')]))} | Private {desc(em_prior(P[m & (pl_t == 'private')]))}")
            rep[f"{nm}_size{s}"] = e.tolist()
        print("   ct106 answers by size (share of kyu answers | A | 1D | mean level): " + "  ".join(f"{s}: {float((ans[n_t == s] < 4).mean()):.3f} | {float((ans[n_t == s] == 3).mean()):.3f} | {float((ans[n_t == s] == 4).mean()):.3f} | {float(ans[n_t == s].mean()):.2f}" for s in SZ))
    # ---------------------------------------------------------------- (4) homogeneity
    P = z["base"].astype(np.float64); P /= P.sum(1, keepdims=True); groups = [P[n_t == s] for s in SZ]; G = gstat(groups); common = em_prior(P)
    null = np.array([gstat([draw(s, common, 200) for s in SZ]) for _ in range(300)]); alt = np.array([gstat([draw(s, H[f"exact {s}"], 200) for s in SZ]) for _ in range(100)]); alt2 = np.array([gstat([draw(s, H[f"min {s}"], 200) for s in SZ]) for _ in range(100)])
    print(f"\n(4) homogeneity (base posterior): G = {G:.2f}; null from validation resampling: median {np.median(null):.2f}, q95 {np.quantile(null, 0.95):.2f}, q99 {np.quantile(null, 0.99):.2f}; share of the null above G {float((null >= G).mean()):.3f}")
    print(f"    power: G under 'total games in [n, n + 4]' median {np.median(alt):.2f} (above the null's q99 in {float((alt > np.quantile(null, 0.99)).mean()):.2f}); under 'total games >= n' median {np.median(alt2):.2f} (above q99 in {float((alt2 > np.quantile(null, 0.99)).mean()):.2f})")
    rep["G"] = G; rep["null_q95"] = float(np.quantile(null, 0.95)); rep["null_q99"] = float(np.quantile(null, 0.99)); rep["p"] = float((null >= G).mean())
    # the same within each pool
    for nm in ("public", "private"):
        g_ = [P[(n_t == s) & (pl_t == nm)] for s in SZ]; Gp = gstat(g_); nullp = np.array([gstat([draw(s, common, 100) for s in SZ]) for _ in range(200)]); print(f"    {nm}: G = {Gp:.2f}; null (100 per size) q95 {np.quantile(nullp, 0.95):.2f}, q99 {np.quantile(nullp, 0.99):.2f}; share above {float((nullp >= Gp).mean()):.3f}")
    # ---------------------------------------------------------------- (5) fixed hypotheses per size
    print("\n(5) log-likelihood against 'balanced' (base posterior; positive = fits better), per size and in total")
    tot = {}
    for hn in ("by players", "players with <= 25 games", "exact", "min", "EM of all sizes", "EM per size"):
        row = []
        for s in SZ:
            m = n_t == s; pi = {"exact": H[f"exact {s}"], "min": H[f"min {s}"], "EM of all sizes": common, "EM per size": em_prior(P[m])}.get(hn, H.get(hn)); row.append(loglik(P[m], pi) - loglik(P[m], H["balanced"]))
        tot[hn] = float(sum(row)); print(f"   {hn:26s}: " + "  ".join(f"size {s}: {x:+7.2f}" for s, x in zip(SZ, row)) + f"  | total {sum(row):+7.2f}")
    rep["loglik_vs_balanced"] = tot
    (RV / "m162_size_composition.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m162_size_composition.json")


if __name__ == "__main__":
    main()
