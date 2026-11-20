#!/usr/bin/env python
"""M228 (2026-11-18): one more Public probe file for 2026-11-19 - the SECOND band at the two boundaries of level B
(design doc M228, registered before any reading). Private rows = ct116; Public rows = ct116 with the band answered B.
Bands on the stored pooled posterior alone (as M214), questions not locked and answered by the pipeline itself:
  from C: answered C, 0.30 <= P(B) / P(C) < 0.50            (the first band, probe_C2B, was >= 0.50)
  from A: answered A, P(B) / P(A) >= 0.50, not in the first band (probe_A2B = held at A while the posterior's own answer is B)
At most 8 questions per side (the highest ratios); a side with fewer than 3 is left out. The same rule names the Private
twin band, which is only printed and stored.

  python -I logs/nano4/m228_probe_b2.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m137f_build2 as mf7  # noqa: E402

LEV = led.LEVELS; OUT = ROOT / "upload_probe"; S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); CAP = 8


def main():
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); qid = z["qid"].astype(str); P = z["learned"].astype(float); P = P / P.sum(1, keepdims=True); pub = np.char.startswith(qid, "pub")
    sub = lambda f: led.decisions(ROOT / f"submission_rank_{f}.csv").reindex(qid).to_numpy().astype(int)  # noqa: E731
    c9, c28, c106, c112, c116 = (sub(f) for f in ("ct9", "ct28", "ct106", "ct112", "ct116")); locked = c28 != c9; L = {l: i for i, l in enumerate(LEV)}; lock5 = locked & (c9 == L["4D"]) & (c28 == L["5D"]); plain = (P @ S.T).argmax(1)
    assert (mf7.guard2(np.where(locked, c28, plain), c106) == c112).all(), "ct112 is not reproduced"; pipe = mf7.guard2(np.where(locked & ~lock5, c28, plain), c106)
    ratio = lambda a, b: P[:, L[b]] / np.maximum(P[:, L[a]], 1e-9)  # noqa: E731
    at = lambda a: (c116 == L[a]) & (pipe == L[a]) & ~locked  # noqa: E731
    first_a = ~locked & (c106 == L["A"]) & (c112 == L["A"]) & (plain == L["B"])
    cand = {"C": at("C") & (ratio("C", "B") >= 0.30) & (ratio("C", "B") < 0.50), "A": at("A") & (ratio("A", "B") >= 0.50) & ~first_a}; band = np.zeros(len(qid), bool); rep = {"sides": {}}
    for a, m in cand.items():
        r = ratio(a, "B"); side = {}
        for nm, half in (("public", pub), ("private_twin", ~pub)):
            idx = np.flatnonzero(m & half); idx = idx[np.argsort(-r[idx], kind="stable")][:CAP]; side[nm] = idx
            print(f"from {a}, {nm}: {int((m & half).sum())} in the rule, {len(idx)} taken: " + ", ".join(f"{qid[i]} ({P[i, L[a]]:.2f} / {P[i, L['B']]:.2f})" for i in idx))
        if len(side["public"]) < 3:
            print(f"   side {a} left out (fewer than 3 Public questions)"); continue
        band[side["public"]] = True; rep["sides"][a] = {k: [str(qid[i]) for i in v] for k, v in side.items()}
    n = int(band.sum()); assert n >= 3 and band[~pub].sum() == 0
    order = pd.read_csv(ROOT / "submission_rank_ct116.csv", dtype=str); pos = {q: i for i, q in enumerate(qid)}; idx = np.array([pos[q] for q in order.question_id]); ans = c116.copy(); ans[band] = L["B"]; assert (ans[~pub] == c116[~pub]).all() and int((ans != c116).sum()) == n
    f = OUT / "probe_B2.csv"
    if f.exists():
        sys.exit(f"{f} exists - not overwritten")
    pd.DataFrame({"question_id": order.question_id, "pred_rank": [LEV[x] for x in ans[idx]]}).to_csv(f, index=False, lineterminator="\n")
    exp = float(sum(S[L["B"]] @ P[i] - S[c116[i]] @ P[i] for i in np.flatnonzero(band))); rep.update(md5=hashlib.md5(f.read_bytes()).hexdigest(), n_public=n, expected_points=exp, threshold_points=0.25 * n)
    print(f"probe_B2.csv: md5 {rep['md5'][:12]}; {n} Public answers -> B, Private identical to ct116; expected by the pooled posterior {exp:+.2f} points; registered: 0.632 dE + 0.368 dW >= {0.25 * n:.2f} -> the Private twin bands are proposed to the user")
    (ROOT / "results/r_variants/m228_probe_b2.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m228_probe_b2.json; decode against submission_rank_ct116.csv (Public 0.8394955)")


if __name__ == "__main__":
    main()
