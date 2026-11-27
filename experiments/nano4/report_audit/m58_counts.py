# Sizes for the next-experiment discussion: A|1D sub-bands below r=2, "A hidden in 2D" candidates, and routed A/1D counts per half.
import numpy as np, pandas as pd, sys
sys.path[:0] = ["src", "logs/nano4"]
from m43_gate import decide
LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]; A, D1, D2 = 3, 4, 5
pr = pd.read_csv("submission_rank_ct9_proba.csv"); P = pr[[f"p_{l}" for l in LV]].to_numpy(float); pub = pr.question_id.str.startswith("pub").to_numpy()
d = decide(P); odds = P[:, A] / P[:, D1]
for lo, hi in ((1 / 2.6, 1 / 2), (1 / 3.5, 1 / 2.6), (1 / 5, 1 / 3.5)):
    m = (d == D1) & (odds >= lo) & (odds < hi)
    print(f"1D-decided, A/1D odds [{lo:.3f}, {hi:.3f}): public {int((m & pub).sum())}, private {int((m & ~pub).sum())}")
for thr in (0.10, 0.15, 0.20):
    m = (d == D2) & (P[:, A] >= thr)
    print(f"2D-decided with P(A) >= {thr}: public {int((m & pub).sum())}, private {int((m & ~pub).sum())}; mean P(A) {P[m, A].mean() if m.any() else 0:.3f}, mean P(2D) {P[m, D2].mean() if m.any() else 0:.3f}")
r = (d == A) | (d == D1)
print(f"routed A/1D: public {int((r & pub).sum())}, private {int((r & ~pub).sum())}")
