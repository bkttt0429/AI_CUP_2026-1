"""Section 4M M71 corrected composition (2026-11-10): submission_rank_ct41.csv.
 - questions the two validated pushes flipped (ct28 != ct9) keep ct28's answer (locked);
 - every other question: ct9 posterior x opponent-anchor evidence (margin bands >= --min-band, a = --a-scale x validation a), expected-score decision;
 - no push is re-computed on the anchored posterior.
Uses the saved per-query tables (m71_val_regime.npz for the calibration, m71_test_queries.npz).
  python -I logs/nano4/m71_build_locked.py --min-band 3 --a-scale 0.74 --out 41
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.submission import write_submission  # noqa: E402
import m71_step2 as s2  # noqa: E402
from m43_gate import decide  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--min-band", type=int, default=3); ap.add_argument("--a-scale", type=float, default=0.74); ap.add_argument("--out", default="41")
a_ = ap.parse_args()
v = dict(np.load(ROOT / "results/r_variants/m71_val_regime.npz")); t = dict(np.load(ROOT / "results/r_variants/m71_test_queries.npz"))
edges, cal, _, _ = s2.fit(v, np.ones(len(v["y"]), bool))
band = np.where(t["target_black"], np.searchsorted(edges[False], t["opp_m"]), np.searchsorted(edges[True], t["opp_m"])); g = s2.group_of(t["hd"], t["target_black"])
logE = np.zeros((len(band), 10)); used = np.zeros(len(band), bool)
for k in (0, 1, 2):
    s = g == k; a = a_.a_scale * cal[k]["a"] * (np.arange(len(cal[k]["a"])) >= a_.min_band); dd = t["opp_lab"][s][:, None] - np.arange(10)[None, :]
    logE[s] = np.log((1 - a[band[s]])[:, None] + a[band[s]][:, None] * cal[k]["ratio"][dd + 9]); used[s] = a[band[s]] > 0
pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); cols = [f"p_{lv}" for lv in LEVELS]
L = pd.DataFrame(logE).groupby(t["qid"]).sum().reindex(pr.question_id).fillna(0.0).to_numpy()
P = pr[cols].to_numpy(float); Pn = P * np.exp(L - L.max(axis=1, keepdims=True)); Pn /= Pn.sum(axis=1, keepdims=True)
sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(pr.question_id).to_numpy()  # noqa: E731
c9, c28, c40 = sub("ct9"), sub("ct28"), sub("ct40"); d_anchor = np.array([LEVELS[i] for i in np.asarray(decide(Pn), int)])
assert (np.array([LEVELS[i] for i in np.asarray(decide(P), int)]) == c9).all(), "ct9 decisions do not reproduce from its probabilities"
locked = c28 != c9; final = np.where(locked, c28, d_anchor)
out = pr.copy(); out[cols] = Pn; out.to_csv(ROOT / f"submission_rank_ct{a_.out}_proba.csv", index=False)
write_submission(list(zip(pr.question_id, final)), ROOT / f"submission_rank_ct{a_.out}.csv")
pub = pr.question_id.str.startswith("pub_").to_numpy()
for name, ref in (("ct28", c28), ("ct40", c40), ("ct9", c9)):
    ch = final != ref
    print(f"ct{a_.out} vs {name}: {int(ch.sum())} differ (public {int(ch[pub].sum())}, private {int(ch[~pub].sum())}); moves " + str(pd.Series([f"{a}->{b}" for a, b in zip(ref[ch], final[ch])]).value_counts().head(16).to_dict()))
nq = pd.Series(used).groupby(t["qid"]).sum().reindex(pr.question_id).fillna(0).to_numpy()
print(f"locked questions {int(locked.sum())} (public {int(locked[pub].sum())}, private {int(locked[~pub].sum())}); questions with >= 1 used anchor {int((nq > 0).sum())}; anchors would have changed a locked question in {int((locked & (d_anchor != c9)).sum())} cases (ignored)")
jump = np.abs(np.array([LEVELS.index(x) for x in final]) - np.array([LEVELS.index(x) for x in c28]))
print(f"size of the changes vs ct28: one level {int((jump == 1).sum())}, two {int((jump == 2).sum())}, three or more {int((jump >= 3).sum())}")
print("md5", hashlib.md5((ROOT / f"submission_rank_ct{a_.out}.csv").read_bytes()).hexdigest()[:12], f"submission_rank_ct{a_.out}.csv; rows", len(final))
