# Research idea check: the "kyu-block" form of the A|1D shift (all kyu classes D, C, B, A x r = one cutpoint shift in a cumulative-logit
# model) vs ct13 (A x 2 on A/1D-decided only). Which questions differ?
import numpy as np, pandas as pd, sys
sys.path[:0] = ["src", "logs/nano4"]
from m43_gate import decide
LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
pr = pd.read_csv("submission_rank_ct9_proba.csv"); P = pr[[f"p_{l}" for l in LV]].to_numpy(float); qid = pr.question_id.to_numpy(); pub = pr.question_id.str.startswith("pub").to_numpy()
d13 = pd.read_csv("submission_rank_ct13.csv", dtype=str).set_index("question_id")["pred_rank"].reindex(qid).map({l: i for i, l in enumerate(LV)}).to_numpy()
for r in (1.5, 2.0, 2.6):
    Q = P.copy(); Q[:, :4] *= r; Q /= Q.sum(1, keepdims=True); dk = decide(Q)
    diff = dk != d13
    moves = pd.Series([f"{LV[a]}->{LV[b]}" for a, b in zip(d13[diff], dk[diff])]).value_counts().to_dict()
    print(f"kyu-block r={r}: differs from ct13 on {int(diff.sum())} (public {int((diff & pub).sum())}, private {int((diff & ~pub).sum())}); moves ct13->kyu-block {moves}")
