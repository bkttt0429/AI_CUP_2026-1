#!/usr/bin/env python
"""Section 4M M145 (2026-11-15, pilot, report only): is a recognised opponent's label a sharper reading when that opponent's own
level is stable? (design doc M145)

The anchor evidence uses one table per game type: P(label of the recognised opponent | label of the target). Its spread comes
from both players' levels moving over their periods (a label is one value per player; an even game pairs the two players' levels
at the time of the game). The spread of a candidate X can be measured without the target: in X's OWN validation games, the
confidently recognised opponents' labels scatter around X's label. If that scatter differs between candidates and carries over
to the games where X is the recognised opponent, per-candidate tables sharpen the evidence of stable candidates.

Measured on the production identifier (SP2e, statistic m15), confident even-game recognitions (band >= 5):
  per player P (as the target of his own games): n_P readings d = opponent label - own label; stability s_P = share with d = 0,
  spread m_P = mean |d|;
  for every confident recognition (target T, candidate X): X's stability computed from X's own readings whose recognised
  opponent is not in T's fold (so T's label never enters), X needs >= MIN readings; the same-level rate P(label X = label T),
  the within-one rate and mean |d| by tercile of X's stability, overall, by the target's level and after removing level effects
  (logistic regression with level dummies); the split-half reliability of s_P (odd / even readings) = is stability a trait?
Reading (registered): a same-level rate that differs by >= 0.10 between the top and the bottom tercile, with the level-adjusted
coefficient's CI excluding 0 -> build the split tables and test them (M146); otherwise closed.

  python -I logs/nano4/m145_candidate_stability.py [--id SP2e~m15] [--min 3]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m86_bundle as b  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402

s2 = b.s2; RV = ROOT / "results/r_variants"


def logit_fit(X, y, l2=1e-3, iters=100):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + l2 * w; H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1]); st = np.linalg.solve(H, g); w -= st
        if np.abs(st).max() < 1e-9:
            break
    p = 1 / (1 + np.exp(-X @ w)); cov = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1]))
    return w, np.sqrt(np.diag(cov))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--id", default="SP2e~m15"); ap.add_argument("--min", type=int, default=3); ap.add_argument("--band", type=int, default=5); a_ = ap.parse_args()
    V, T, key_pid, used = b.sources([a_.id]); z = V[a_.id]; best = np.load(RV / f"m84_{a_.id}_val.npz")["f_opp_best"].astype(str)
    ones = np.ones(len(z["y"]), bool); edges, cal, band, g = s2.fit(z, ones); pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); fold_of = pf.fold.astype(int).to_dict(); lab_of = {p: LEVELS.index(r) for p, r in pf["rank"].items()}; ng_of = pf.n_games.astype(int).to_dict()
    assert (np.array([lab_of[p] for p in best[:2000]]) == z["opp_lab"][:2000]).all(), "candidate ids and labels do not match"
    conf = (band >= a_.band) & (g == 0); print(f"identifier {a_.id}: {len(conf)} validation games, even {int((g == 0).sum())}, confident even recognitions (band >= {a_.band}) {int(conf.sum())} = {float(conf.sum() / (g == 0).sum()):.3f} of the even games")
    D = pd.DataFrame({"T": z["pid"][conf].astype(str), "X": best[conf], "yT": z["y"][conf].astype(int), "yX": z["opp_lab"][conf].astype(int), "fT": z["fold"][conf].astype(int)}); D["d"] = D.yX - D.yT; D["fX"] = D.X.map(fold_of).astype(int)
    print(f"confident readings: same level {float((D.d == 0).mean()):.3f}, within one {float((D.d.abs() <= 1).mean()):.3f}, mean |d| {float(D.d.abs().mean()):.3f}; targets {D['T'].nunique()}, distinct candidates {D.X.nunique()}; candidate's fold: " + " ".join(f"{k}: {v}" for k, v in D.fX.value_counts().sort_index().items()))
    per = D.groupby("T").agg(n=("d", "size"), s=("d", lambda x: float((x == 0).mean())), m=("d", lambda x: float(np.abs(x).mean())), bias=("d", "mean"))
    print(f"players with readings as the target: {len(per)}; readings per player: quartiles " + " / ".join(f"{x:.0f}" for x in per.n.quantile([0.25, 0.5, 0.75, 0.9]).to_numpy()) + f"; with >= {a_.min}: {int((per.n >= a_.min).sum())}")
    # split-half reliability of the per-player stability (alternate readings)
    D["par"] = D.groupby("T").cumcount() % 2; h = D.groupby(["T", "par"]).d.agg(s=lambda x: float((x == 0).mean()), m=lambda x: float(np.abs(x).mean()), n="size").unstack()
    ok = (h[("n", 0)] >= 2) & (h[("n", 1)] >= 2); hh = h[ok]; yT = D.groupby("T").yT.first().reindex(hh.index)
    res_s = [hh[("s", k)] - hh[("s", k)].groupby(yT).transform("mean") for k in (0, 1)]; res_m = [hh[("m", k)] - hh[("m", k)].groupby(yT).transform("mean") for k in (0, 1)]
    print(f"split-half (players with >= 2 + 2 readings: {int(ok.sum())}): same-level share r = {float(np.corrcoef(hh[('s', 0)], hh[('s', 1)])[0, 1]):+.3f}, within the player's level r = {float(np.corrcoef(res_s[0], res_s[1])[0, 1]):+.3f}; mean |d| r = {float(np.corrcoef(hh[('m', 0)], hh[('m', 1)])[0, 1]):+.3f}, within level {float(np.corrcoef(res_m[0], res_m[1])[0, 1]):+.3f}")
    # the candidate's stability from his own readings, leaving out readings whose recognised opponent is in the target's fold
    own = {x: grp for x, grp in D.groupby("T")}; sX = np.full(len(D), np.nan); mX = np.full(len(D), np.nan); nX = np.zeros(len(D), int)
    for i, (x, ft, t) in enumerate(zip(D.X.to_numpy(), D.fT.to_numpy(), D["T"].to_numpy())):
        grp = own.get(x)
        if grp is None:
            continue
        u = grp[(grp.fX != ft) & (grp.X != t)]
        if len(u) >= a_.min:
            sX[i] = float((u.d == 0).mean()); mX[i] = float(u.d.abs().mean()); nX[i] = len(u)
    D["sX"], D["mX"], D["nX"] = sX, mX, nX; has = np.isfinite(sX); print(f"\nreadings whose candidate has >= {a_.min} own readings (target's fold left out): {int(has.sum())} of {len(D)} ({float(has.mean()):.3f}); median own readings {np.median(nX[has]):.0f}")
    E = D[has].copy(); rep = {"n": int(has.sum())}
    for stat, hi_is_stable in (("sX", True), ("mX", False)):
        qs = E[stat].quantile([1 / 3, 2 / 3]).to_numpy(); E["bin"] = np.searchsorted(qs, E[stat].to_numpy(), side="right"); print(f"by tercile of the candidate's {'same-level share' if stat == 'sX' else 'mean |d|'} (cuts {qs[0]:.2f}, {qs[1]:.2f}):")
        rows = {}
        for k in (0, 1, 2):
            s = E[E.bin == k]; rows[k] = {"n": len(s), "same": float((s.d == 0).mean()), "within1": float((s.d.abs() <= 1).mean()), "mean_abs": float(s.d.abs().mean())}
            print(f"    tercile {k}: {len(s):6d} readings; same level {rows[k]['same']:.3f}, within one {rows[k]['within1']:.3f}, mean |d| {rows[k]['mean_abs']:.3f}")
        rep[stat] = rows
        print("    same-level rate by the target's level (bottom | middle | top tercile): " + "  ".join(f"{LEVELS[c]} " + "|".join(f"{float((E[(E.bin == k) & (E.yT == c)].d == 0).mean()):.2f}" for k in (0, 1, 2)) for c in range(10)))
    # level-adjusted logistic regression: same level ~ candidate's stability (standardised) + target-level dummies + candidate-level dummies
    X = np.column_stack([(E.sX - E.sX.mean()) / E.sX.std(), np.eye(10)[E.yT.to_numpy()], np.eye(10)[E.yX.to_numpy()][:, 1:]]); w, se = logit_fit(X, (E.d == 0).to_numpy(float))
    print(f"level-adjusted logistic regression (same level ~ candidate's same-level share, per sd; dummies of both levels): {w[0]:+.3f} +/- {se[0]:.3f} (z {w[0] / se[0]:+.1f})"); rep["logit"] = {"coef": float(w[0]), "se": float(se[0])}
    # the same with the candidate's level removed from the outcome side: does the stability predict |d| beyond the levels?
    grp_mean = E.groupby(["yT", "yX"]).d.transform(lambda x: float(np.abs(x).mean())); print(f"correlation of the candidate's same-level share with the reading's |d|: raw {float(np.corrcoef(E.sX, E.d.abs())[0, 1]):+.3f}; (the pair of levels fixes |d| exactly, so the usable part is the share of sharp candidates per level)")
    lvl = E.groupby("yX").agg(share_same=("d", lambda x: float((x == 0).mean())), sX=("sX", "mean"), n=("d", "size")); print("by the candidate's level (share of readings with the same level | mean stability of the candidates | n): " + "  ".join(f"{LEVELS[c]} {r.share_same:.2f}|{r.sX:.2f}|{int(r.n)}" for c, r in lvl.iterrows()))
    # how much sharper could the tables be: the entropy of d given the target's level, pooled against per tercile (lower = sharper)
    def ent(s):
        tot = 0.0
        for c in range(10):
            x = s[s.yT == c].d.to_numpy()
            if len(x) < 30:
                continue
            p = np.bincount(x + 9, minlength=19) / len(x); p = p[p > 0]; tot += len(x) * float(-(p * np.log(p)).sum())
        return tot / max(len(s), 1)
    E["bin"] = np.searchsorted(E.sX.quantile([1 / 3, 2 / 3]).to_numpy(), E.sX.to_numpy(), side="right")
    print(f"entropy of (candidate label - target label) given the target's level: pooled {ent(E):.3f} nats; by tercile of the candidate's stability " + " / ".join(f"{ent(E[E.bin == k]):.3f}" for k in (0, 1, 2)) + f"; weighted mean {sum(ent(E[E.bin == k]) * (E.bin == k).mean() for k in (0, 1, 2)):.3f}")
    rep["entropy"] = {"pooled": ent(E), "by_tercile": [ent(E[E.bin == k]) for k in (0, 1, 2)]}
    (RV / "m145_candidate_stability.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m145_candidate_stability.json")


if __name__ == "__main__":
    main()
