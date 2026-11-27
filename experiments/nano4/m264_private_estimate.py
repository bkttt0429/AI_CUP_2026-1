#!/usr/bin/env python
"""M264 (2026-11-21, closing): what to expect on the Private half for the final answers (Private rows of ct116 = of ct118e).
Self-estimates under the pooled posterior of the ct112 rule (sum of posterior x score), by kind of answer, next to the same figures
for the Public half where the realised scores are known. No test labels; nothing is built.

  python -I logs/nano4/m264_private_estimate.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m137b_build as mb7  # noqa: E402

LEV = led.LEVELS; RV = ROOT / "results/r_variants"; b = mb7.b


def main():
    z = np.load(RV / "m237_grid.npz", allow_pickle=True); names = [str(x) for x in z["names"]]; P = z["P"][names.index("current")].astype(np.float64); P = P / P.sum(1, keepdims=True); qid = z["qid"].astype(str); nq = z["nq"].astype(int)
    base = z["base"].astype(np.float64); base = base / base.sum(1, keepdims=True); pub = np.char.startswith(qid, "pub"); S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)])
    rd = lambda f: np.array([LEV.index({r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(ROOT / f))}[q]) for q in qid])  # noqa: E731
    a9, a112, a116, a118e = rd("submission_rank_ct9.csv"), rd("submission_rank_ct112.csv"), rd("submission_rank_ct116.csv"), rd("submission_rank_ct118e.csv"); own = np.asarray(b.decide(P), int); locked = z["locked"].astype(bool)
    assert np.array_equal(a116[~pub], a118e[~pub])

    def est(ans, Pm, m):
        e = (Pm[m] * S[ans[m]]).sum(1); v = (Pm[m] * S[ans[m]] ** 2).sum(1) - e ** 2; return float(e.mean()), float(np.sqrt(v.sum()) / m.sum()), float((Pm[m][np.arange(m.sum()), ans[m]]).mean())
    rep = {}
    print("self-estimates (posterior x score; sd = sampling sd of the half's mean under the posterior; mean posterior probability of the answer)")
    for nm, ans, Pm, m, actual in (("Public, base answers (ct9) under the base posterior", a9, base, pub, 0.7622032), ("Public, ct112 answers under the pooled posterior", a112, P, pub, 0.8331743), ("Private, base answers (ct9) under the base posterior", a9, base, ~pub, None),
                                   ("Private, ct112 answers under the pooled posterior", a112, P, ~pub, None), ("Private, FINAL answers (ct116) under the pooled posterior", a116, P, ~pub, None)):
        e, sd, pm = est(ans, Pm, m); print(f"   {nm}: {e:.4f} (sd {sd:.4f}; exact-probability {pm:.3f})" + (f"; realised {actual:.4f} ({actual - e:+.4f})" if actual else "")); rep[nm] = {"estimate": e, "sd": sd, "actual": actual}
    print("   by question size, FINAL answers on Private | ct112 answers on Public: " + " ; ".join(f"{s}: {est(a116, P, ~pub & (nq == s))[0]:.3f} | {est(a112, P, pub & (nq == s))[0]:.3f}" for s in (5, 10, 15, 20)))
    # kinds of answers
    for half, m, ans in (("Public (ct112)", pub, a112), ("Private (FINAL)", ~pub, a116)):
        user = m & (a116 != a112) if half.startswith("Private") else np.zeros(len(qid), bool); lk = m & locked & ~user; guard = m & ~locked & ~user & (ans != own); plain = m & ~locked & ~user & (ans == own); print(f"   {half}:")
        for nm, mm in (("posterior's own best answer", plain), ("held by a guard (differs from the posterior's best)", guard), ("locked pushes (1D->A, 4D->5D)", lk), ("the user's eight decisions", user)):
            if mm.any():
                e = (P[mm] * S[ans[mm]]).sum(1); eo = (P[mm] * S[own[mm]]).sum(1); print(f"      {nm}: {int(mm.sum())} questions, self-estimate {e.sum():.2f} points ({e.mean():.3f} each); the posterior's best answer would self-estimate {eo.sum():.2f}")
                rep[f"{half} | {nm}"] = {"n": int(mm.sum()), "points": float(e.sum()), "points_own": float(eo.sum())}
    kinds = {}
    for i in np.flatnonzero(~pub & locked & (a116 == a112)):
        k = f"{LEV[own[i]]}->{LEV[a116[i]]}" if own[i] != a116[i] else f"{LEV[a116[i]]} (posterior agrees)"; kinds[k] = kinds.get(k, 0) + 1
    print(f"   Private locked answers by kind (posterior's best -> final): {kinds}")
    (RV / "m264_private_estimate.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m264_private_estimate.json")


if __name__ == "__main__":
    main()
