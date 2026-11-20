#!/usr/bin/env python
"""Section 4M M137c (2026-11-14, report only): which posterior predicts the recorded Public scores best?

For every upload in the Public ledger (exact scores) the expected Public score under a posterior P is
E = mean over the 400 Public questions of sum_c P_q(c) s(decision_q, c). Compared with the actual scores for four posteriors:
the base probabilities (ct9), the current rule's posterior (base x v5 evidence), the M129 bundle posterior, and the learned pool
of M137 (results/r_variants/m137_test_posterior.npz). Reported: bias and RMSE of the absolute scores, and - sharper - of the
differences to ct9 (the base file every upload is compared with), their correlation and slope; the prediction for ct105.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402

up = {**led.KNOWN_SCORES, **json.load(open(led.LEDGER))}
z1 = np.load(ROOT / "results/r_variants/m129_test_posterior.npz", allow_pickle=True); z2 = np.load(ROOT / "results/r_variants/m137_test_posterior.npz", allow_pickle=True)
assert list(z1["qid"]) == list(z2["qid"]); qid = np.array([str(x) for x in z1["qid"]]); pub = np.array([x.startswith("pub") for x in qid])
post = {"base (ct9)": z1["base"], "current rule (base x v5)": z1["current"], "M129 bundle": z1["bundle"], "learned pool (M137)": z2["learned"]}
S = np.array([[led.s(d, c) for c in range(10)] for d in range(10)])  # [decision, truth]
dec = {f: led.decisions(f).reindex(qid).to_numpy().astype(int) for f in list(up) + ["submission_rank_ct105.csv"]}
files = [f for f in up if f != led.REF]; act = np.array([up[f] for f in files]); ref = up[led.REF]
print(f"{len(up)} uploads with an exact Public score; reference {led.REF} {ref:.7f}")
for name, P in post.items():
    P = np.maximum(P[pub], 1e-9); P = P / P.sum(1, keepdims=True)
    E = lambda f: float((P * S[dec[f][pub]]).sum(1).mean())  # noqa: E731
    e_ref = E(led.REF); pred = np.array([E(f) for f in files]); dp, da = pred - e_ref, act - ref
    slope = float((dp * da).sum() / (dp * dp).sum()); r = float(np.corrcoef(dp, da)[0, 1])
    print(f"{name:26s}: predicts ct9 {e_ref:.4f} (actual {ref:.4f}); all uploads: bias {float((pred - act).mean()):+.4f}, RMSE {float(np.sqrt(((pred - act) ** 2).mean())):.4f} | differences to ct9: RMSE {float(np.sqrt(((dp - da) ** 2).mean())):.4f}, "
          f"correlation {r:+.3f}, actual per predicted {slope:.2f} | ct104b: predicted {E('submission_rank_ct104b.csv'):.4f} (actual {up['submission_rank_ct104b.csv']:.4f}) | ct105: predicted {E('submission_rank_ct105.csv'):.4f}, i.e. {400 * (E('submission_rank_ct105.csv') - E('submission_rank_ct104b.csv')):+.2f} points against ct104b")
print("per upload (actual difference to ct9, then the predicted difference under each posterior):")
for f in sorted(files, key=lambda x: up[x]):
    row = []
    for name, P in post.items():
        P = np.maximum(P[pub], 1e-9); P = P / P.sum(1, keepdims=True); row.append(float((P * S[dec[f][pub]]).sum(1).mean() - (P * S[dec[led.REF][pub]]).sum(1).mean()))
    print(f"  {f.replace('submission_rank_', '').replace('.csv', ''):16s} actual {up[f] - ref:+.4f} | " + "  ".join(f"{x:+.4f}" for x in row))
