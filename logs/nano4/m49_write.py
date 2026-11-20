"""Section 4M M49: write submission_rank_ct12.csv = ct9 with the 'bal' specialists at 2D|3D and A|1D (trained on all four folds' validation
questions, seeds 0-2 = ct9's members), applied to submission_rank_ct9_proba.csv. Report only; the rule decision is in the design doc.
  python -I logs/nano4/m49_write.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, decide  # noqa: E402
from m46_a1d import xprobs  # noqa: E402
from m49_select import FIXED, fold_in, specialist  # noqa: E402

D = {}
for v in FOLDS:
    y, n, pid, q = load_fold(v)
    P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
    D[v] = {"y": y, "X": xprobs(P, n)}
pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv"); cols = [f"p_{lv}" for lv in LEVELS]; P = pr[cols].to_numpy(float)
ng = pd.concat([pd.read_csv(ROOT / f"data/tests/rank_prediction_test_{s}.csv", usecols=["question_id", "num_games"]) for s in ("public", "private")]).set_index("question_id")["num_games"]
Xt = xprobs(P, ng.reindex(pr["question_id"]).to_numpy(float))
Q = fold_in(P, {k: specialist(D, list(FOLDS), k, Xt) for k in FIXED})
qid = pr["question_id"]; pub = qid.str.startswith("pub").to_numpy()
d9 = pd.read_csv(ROOT / "submission_rank_ct9.csv").set_index("question_id")["pred_rank"].reindex(qid).to_numpy()
d10 = pd.read_csv(ROOT / "submission_rank_ct10.csv").set_index("question_id")["pred_rank"].reindex(qid).to_numpy()
assert (np.array([LEVELS[i] for i in decide(P)]) == d9).all()
d12 = np.array([LEVELS[i] for i in decide(Q)])
order = pd.read_csv(ROOT / "submission_rank_ct9.csv")["question_id"]
pd.DataFrame({"question_id": qid, "pred_rank": d12}).set_index("question_id").loc[order].reset_index().to_csv(ROOT / "submission_rank_ct12.csv", index=False)
o = pd.DataFrame(Q, columns=cols); o["question_id"] = qid; o.to_csv(ROOT / "submission_rank_ct12_proba.csv", index=False)
for nm, dd in (("ct9", d9), ("ct10", d10)):
    ch = d12 != dd; print(f"ct12 vs {nm}: {ch.sum()} of 800 differ (public {ch[pub].sum()}, private {ch[~pub].sum()})")
ch = d12 != d9
print("changes vs ct9: " + ", ".join(f"{a}->{b} {c}" for (a, b), c in pd.Series(list(zip(d9[ch], d12[ch]))).value_counts().items()))
print("predicted counts ct9 ", pd.Series(d9).value_counts().reindex(LEVELS).to_dict())
print("predicted counts ct12", pd.Series(d12).value_counts().reindex(LEVELS).to_dict())
