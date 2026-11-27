#!/usr/bin/env python
"""Section 4M M245b (2026-11-19), report only, no test labels: does the level of the IDENTIFIED OPPONENTS change with the
question size on the test set? Even games pair equal ranks, so the labels of confidently identified opponents (identifier SQ4,
statistic m15, calibrated true-pair share >= 0.9, even games) are a model-free reading of the level mix of the questions.
Validation questions of the three sizes come from the same players, so there the reading is flat by construction (printed as the
reference, players with 20 or more games); the test set is read per size (5 / 10 / 15 / 20), Public and Private.

  python -I logs/nano4/m245b_anchor_size.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402

m3, b, s2, FOLDS, STAT = mb7.m3, mb7.b, mb7.s2, mb7.FOLDS, mb7.STAT
m87 = m3.m87


def main():
    rng = np.random.default_rng(2452); V, T, _, _ = b.sources([f"SQ4~{STAT}"]); z, t = V[f"SQ4~{STAT}"], T[f"SQ4~{STAT}"]; ones = np.ones(len(z["y"]), bool)
    edges, cal, _ = m87.corrected(z, ones); tb = t["target_black"]; band = np.where(~tb, np.searchsorted(edges[True], t["opp_m"]), np.searchsorted(edges[False], t["opp_m"])); g = s2.group_of(t["hd"], tb)
    a = np.zeros(len(band))
    for k in (0, 1, 2):
        a[g == k] = cal[k]["a"][band[g == k]]
    qid = t["qid"].astype(str); nq = pd.Series(qid).map(pd.Series(qid).value_counts()).to_numpy(); sel = (g == 0) & (a >= 0.9); lab = t["opp_lab"].astype(int); pubg = np.char.startswith(qid, "pub")
    print(f"test games {len(qid)}; even games with a confident identification: {int(sel.sum())} ({float(sel.mean()):.3f}); by question size: " + ", ".join(f"n={s}: {float(sel[nq == s].mean()):.3f}" for s in (5, 10, 15, 20)))
    df = pd.DataFrame({"q": qid[sel], "lab": lab[sel], "n": nq[sel], "pub": pubg[sel]}); qm = df.groupby("q").agg(lab=("lab", "mean"), kyu=("lab", lambda x: float((x <= 3).mean())), n=("n", "first"), pub=("pub", "first"), k=("lab", "size"))
    def boot(x):
        bs = [float(x[rng.integers(0, len(x), len(x))].mean()) for _ in range(2000)]; return float(x.mean()), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    for nm, m in (("both halves", np.ones(len(qm), bool)), ("Public", qm["pub"].to_numpy()), ("Private", ~qm["pub"].to_numpy())):
        print(f"\ntest, {nm}: per question size - questions with an anchor, mean level of the identified opponents (question means), share of kyu opponents")
        for s in (5, 10, 15, 20):
            x = qm[m & (qm["n"].to_numpy() == s)]; ml, lo, hi = boot(x["lab"].to_numpy()); mk, lo2, hi2 = boot(x["kyu"].to_numpy())
            print(f"   n={s:2d}: {len(x):3d} questions, mean level {ml:.3f} [{lo:.3f}, {hi:.3f}], kyu share {mk:.3f} [{lo2:.3f}, {hi2:.3f}]")
        x5 = qm[m & (qm["n"].to_numpy() == 5)]; x20 = qm[m & (qm["n"].to_numpy() == 20)]; d = [float(x20["kyu"].to_numpy()[rng.integers(0, len(x20), len(x20))].mean() - x5["kyu"].to_numpy()[rng.integers(0, len(x5), len(x5))].mean()) for _ in range(4000)]
        print(f"   kyu share, 20-game minus 5-game questions: {float(x20['kyu'].mean() - x5['kyu'].mean()):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]; mean level 20 minus 5: {float(x20['lab'].mean() - x5['lab'].mean()):+.3f}")
    # validation reference: the 20-game questions' players, games grouped as their 5 / 10 / 20-game questions
    act = pd.concat([pd.read_csv(p, usecols=["player_id"], dtype=str) for p in sorted((ROOT / "data/training").glob("train_*.csv"))]).player_id.value_counts()
    vb = np.zeros(len(z["y"]), int); va = np.zeros(len(z["y"]))
    for v in FOLDS:
        tr = z["fold"] != v; e_, c_, _ = m87.corrected(z, tr); te = np.flatnonzero(z["fold"] == v); tbv = z["target_black"][te]; bd = np.where(tbv, np.searchsorted(e_[False], z["opp_m"][te]), np.searchsorted(e_[True], z["opp_m"][te])); gv = s2.group_of(z["hd"][te], tbv)
        for k in (0, 1, 2):
            va[te[gv == k]] = c_[k]["a"][bd[gv == k]]
        vb[te] = gv
    selv = (vb == 0) & (va >= 0.9); pidv = z["pid"].astype(str); ngv = act.reindex(pidv).to_numpy(); dv = pd.DataFrame({"p": pidv[selv], "lab": z["opp_lab"][selv].astype(int), "y": z["y"][selv].astype(int), "ng": ngv[selv]})
    pm = dv.groupby("p").agg(kyu=("lab", lambda x: float((x <= 3).mean())), lab=("lab", "mean"), y=("y", "first"), ng=("ng", "first"))
    comp = {"D": 51, "C": 44, "B": 49, "A": 56, "1D": 24, "2D": 34, "3D": 35, "4D": 36, "5D": 36, "6D": 35}; pi = np.array(list(comp.values()), float) / 400
    for nm, m in (("validation players with 20 or more games", pm["ng"].to_numpy() >= 20), ("validation players with fewer than 20 games", pm["ng"].to_numpy() < 20)):
        x = pm[m]; by = x.groupby("y").agg(kyu=("kyu", "mean"), lab=("lab", "mean"), k=("kyu", "size")); w = pi[by.index.to_numpy()] / pi[by.index.to_numpy()].sum()
        print(f"\n{nm}: {len(x)} players with an anchor; under the Public level mix the expected kyu share of the identified opponents is {float((by['kyu'].to_numpy() * w).sum()):.3f}, mean level {float((by['lab'].to_numpy() * w).sum()):.3f}; by true level (kyu share of the opponents): " + " ".join(f"{['D','C','B','A','1D','2D','3D','4D','5D','6D'][c]} {r['kyu']:.2f}" for c, r in by.iterrows()))
    print("REPORT ONLY")


if __name__ == "__main__":
    sys.exit(main())
