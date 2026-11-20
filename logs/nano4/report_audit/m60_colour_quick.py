# M60 quick look: test questions are single-colour (400/400 per half) while 98% of training players mix colours.
# (1) colours of the 8 known A, of ct13's 18 flips, and of ct26's / ct29's sets; (2) test mean predicted level by colour;
# (3) validation: ct9 residual (expected level - truth) and A|1D first-band precision by the question's white share.
import json, sys
import numpy as np, pandas as pd
sys.path[:0] = ["src", "logs/nano4"]
from m43_gate import FOLDS, decide
from ens_compare import ES, load_fold
LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]; A, D1 = 3, 4
g = pd.read_csv("results/trunk_seq/es/m51_test_games.csv"); colq = g.groupby("question_id")["color"].first()
d9, d13 = (pd.read_csv(f"submission_rank_{k}.csv", dtype=str).set_index("question_id")["pred_rank"] for k in ("ct9", "ct13"))
fl = d13.index[d13 != d9.reindex(d13.index)]
for name, qs in (("ct13 public flips (known A)", [q for q in fl if q.startswith("pub")]), ("ct13 private flips", [q for q in fl if q.startswith("priv")])):
    print(f"{name}: colours {colq.reindex(qs).value_counts().to_dict()}")
pr = pd.read_csv("submission_rank_ct9_proba.csv").set_index("question_id"); P = pr[[f"p_{l}" for l in LV]].to_numpy(); E = pd.Series((P * np.arange(10)).sum(1), index=pr.index)
dt = pd.Series(decide(P), index=pr.index); odds = pr["p_A"] / pr["p_1D"]
for half in ("pub", "priv"):
    qs = [q for q in pr.index if q.startswith(half)]
    for c in ("B", "W"):
        s = [q for q in qs if colq[q] == c]
        band = [q for q in s if dt[q] == D1 and odds[q] >= 0.5]
        print(f"test {half} colour {c}: {len(s)} questions, mean expected level {E[s].mean():.2f}, routed A/1D {sum(dt[q] in (A, D1) for q in s)}, first-band 1D->A candidates {len(band)}")
hd = pd.read_csv("results/data/hd_table.csv", dtype={"game_key": str}, usecols=["game_key", "color"]).drop_duplicates("game_key").set_index("game_key")["color"].to_dict()
rows = []
for v in FOLDS:
    y, n, pid, q = load_fold(v)
    Pv = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
    wsh = np.array([np.mean([hd.get(k, "B") == "W" for k in ks]) for ks in q["game_keys"]])
    dv = decide(Pv); ov = Pv[:, A] / Pv[:, D1]
    rows.append(pd.DataFrame({"y": y, "n": n, "w": wsh, "E": (Pv * np.arange(10)).sum(1), "d": dv, "odds": ov}))
V = pd.concat(rows); V["res"] = V.E - V.y
V["wbin"] = pd.cut(V.w, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
print("validation: residual (expected - truth) by white share; and A|1D first-band precision (decided 1D, odds >= 1/2: share truly A)")
for b, s in V.groupby("wbin", observed=True):
    band = s[(s.d == D1) & (s.odds >= 0.5)]
    print(f"  white share {b}: {len(s):6d} q, residual {s.res.mean():+.3f}, by n " + " ".join(f"{m}:{s[s.n == m].res.mean():+.3f}" for m in (5, 10, 20))
          + f"; first band {len(band)} q, precision A {((band.y == A).mean() if len(band) else float('nan')):.3f}")
for c in range(10):
    s = V[V.y == c]
    print(f"  true {LV[c]:3s}: corr(white share, residual) {np.corrcoef(s.w, s.res)[0, 1]:+.3f}, slope {np.polyfit(s.w, s.res, 1)[0]:+.3f} levels per unit white share")
