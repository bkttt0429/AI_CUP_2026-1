"""M71: what the anchors did on the three known-truth Public questions ct40 broke, and how often one question's anchors are the SAME candidate."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
import m71_step2 as s2  # noqa: E402

v = dict(np.load(ROOT / "results/r_variants/m71_val_regime.npz")); t = dict(np.load(ROOT / "results/r_variants/m71_test_queries.npz"))
edges, cal, vband, vg = s2.fit(v, np.ones(len(v["y"]), bool))
band = np.where(t["target_black"], np.searchsorted(edges[False], t["opp_m"]), np.searchsorted(edges[True], t["opp_m"])); g = s2.group_of(t["hd"], t["target_black"])
a = np.array([0.74 * cal[k]["a"][b] if k >= 0 else 0.0 for k, b in zip(g, band)])
p9 = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}).set_index("question_id"); p39 = pd.read_csv(ROOT / "submission_rank_ct39_proba.csv", dtype={"question_id": str}).set_index("question_id")
for q in ("pub_q_0070", "pub_q_0239", "pub_q_0341"):
    s = t["qid"] == q
    print(f"{q} (truth A): {int(s.sum())} games; ct9 posterior " + " ".join(f"{lv} {x:.2f}" for lv, x in zip(LEVELS, p9.loc[q].to_numpy()) if x >= 0.02) + " | with anchors " + " ".join(f"{lv} {x:.2f}" for lv, x in zip(LEVELS, p39.loc[q].to_numpy()) if x >= 0.02))
    for i in np.flatnonzero(s):
        if a[i] >= 0.1:
            print(f"    game HD{t['hd'][i]} target {'B' if t['target_black'][i] else 'W'}: matched {t['opp_best'][i]} label {LEVELS[t['opp_lab'][i]]}, margin band {band[i]}, weight {a[i]:.2f}")
# repetition of the same candidate among a question's anchors (weight >= 0.2)
T = pd.DataFrame({"q": t["qid"], "best": t["opp_best"], "a": a}); A = T[T.a >= 0.2]
rep_t = A.groupby("q").best.agg(lambda x: x.value_counts().iloc[0]); n_t = A.groupby("q").size()
vb = np.where(v["target_black"], np.searchsorted(edges[False], v["opp_m"]), np.searchsorted(edges[True], v["opp_m"])); av = np.array([cal[k]["a"][b] if k >= 0 else 0.0 for k, b in zip(vg, vb)])
V = pd.DataFrame({"q": v["pid"], "best": v["opp_best"], "a": av}); B = V[V.a >= 0.2]; rep_v = B.groupby("q").best.agg(lambda x: x.value_counts().iloc[0]); n_v = B.groupby("q").size()
print(f"test: {len(n_t)} questions with >= 1 anchor (weight >= 0.2); with >= 2 anchors {int((n_t >= 2).sum())}; the same candidate twice or more in {int((rep_t >= 2).sum())} questions ({(rep_t >= 2).sum() / max((n_t >= 2).sum(), 1):.2f} of those with >= 2)")
print(f"validation (20 games each): {len(n_v)} players with >= 1; with >= 2 anchors {int((n_v >= 2).sum())}; same candidate twice or more in {int((rep_v >= 2).sum())} ({(rep_v >= 2).sum() / max((n_v >= 2).sum(), 1):.2f})")
# label spread of a question's anchors: how often two confident anchors of one question disagree by >= 3 levels
def spread(df, lab):
    df = df.assign(lab=lab[df.index]); s = df.groupby("q").lab.agg(lambda x: x.max() - x.min()); return float((s[df.groupby("q").size() >= 2] >= 3).mean())
print(f"questions with >= 2 anchors whose anchor labels span >= 3 levels: test {spread(A, t['opp_lab']):.2f}, validation {spread(B, v['opp_lab']):.2f}")
