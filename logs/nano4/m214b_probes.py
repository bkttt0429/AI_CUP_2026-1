#!/usr/bin/env python
"""M214 amendment (2026-11-17, before any reading): two more Public probe files, same construction as m214_probes.py.

  probe_2D3D  answered 2D, P(3D) / P(2D) >= 0.50  -> 3D
  probe_3D4D  answered 3D, P(4D) / P(3D) >= 0.50  -> 4D

Original header of m214_probes.py:
M214 (2026-11-17): Public probe files for 2026-11-18 (design doc M214, registered before any reading).

Every file equals the final file ct116 on the 400 Private questions; only Public answers change, and each file changes
one band of Public answers from a level to the adjacent level. A band is a rule on the stored pooled posterior alone
(no ledger, no reading), so the same rule names a Private twin band; the twin is only printed and stored, never applied.

  probe_D2C   answered D, held by the guard "never leave D", the pooled posterior's own answer is C      -> C
  probe_C2B   answered C, P(B) / P(C) >= 0.50                                                            -> B
  probe_1D2A  answered 1D (not one of the locked answers), P(A) / P(1D) >= 0.40                          -> A
  probe_5D6D  answered 5D (not locked), P(6D) / P(5D) >= 0.65                                            -> 6D
  probe_5D4D  answered 5D (not locked), P(4D) / P(5D) >= 0.40                                            -> 4D
Reading of a probe (m206_decode.py <file> <score> --ref submission_rank_ct116.csv --ref-score 0.8394955): dE, dW.

  python -I logs/nano4/m214b_probes.py
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

LEV = led.LEVELS; OUT = ROOT / "upload_probe"; S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)])
EXPECT = {"probe_2D3D": (7, 10), "probe_3D4D": (6, 10)}  # (Public, Private) sizes seen in m214_bands.py


def main():
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); qid = z["qid"].astype(str); P = z["learned"].astype(float); P = P / P.sum(1, keepdims=True); pub = np.char.startswith(qid, "pub")
    sub = lambda f: led.decisions(ROOT / f"submission_rank_{f}.csv").reindex(qid).to_numpy().astype(int)  # noqa: E731
    c9, c28, c106, c112, c116 = (sub(f) for f in ("ct9", "ct28", "ct106", "ct112", "ct116")); locked = c28 != c9; L = {l: i for i, l in enumerate(LEV)}; lock5 = locked & (c9 == L["4D"]) & (c28 == L["5D"]); plain = (P @ S.T).argmax(1)
    assert (mf7.guard2(np.where(locked, c28, plain), c106) == c112).all(), "ct112 is not reproduced"; pipe = mf7.guard2(np.where(locked & ~lock5, c28, plain), c106)
    ratio = lambda a, b: P[:, L[b]] / np.maximum(P[:, L[a]], 1e-9)  # noqa: E731
    at = lambda a: (c116 == L[a]) & (pipe == L[a]) & ~locked  # noqa: E731
    bands = {"probe_2D3D": ("2D", "3D", at("2D") & (ratio("2D", "3D") >= 0.50)), "probe_3D4D": ("3D", "4D", at("3D") & (ratio("3D", "4D") >= 0.50))}
    order = pd.read_csv(ROOT / "submission_rank_ct116.csv", dtype=str); pos = {q: i for i, q in enumerate(qid)}; idx = np.array([pos[q] for q in order.question_id]); OUT.mkdir(exist_ok=True); rep = {}
    for name, (a, b, m) in bands.items():
        n_pub, n_priv = int((m & pub).sum()), int((m & ~pub).sum()); assert (n_pub, n_priv) == EXPECT[name], f"{name}: band sizes {(n_pub, n_priv)} differ from the registered {EXPECT[name]}"
        ans = c116.copy(); ans[m & pub] = L[b]; assert (ans[~pub] == c116[~pub]).all() and int((ans != c116).sum()) == n_pub; f = OUT / f"{name}.csv"
        if f.exists():
            sys.exit(f"{f} exists - not overwritten")
        pd.DataFrame({"question_id": order.question_id, "pred_rank": [LEV[x] for x in ans[idx]]}).to_csv(f, index=False, lineterminator="\n")
        exp = float(sum(P[i, L[b]] - P[i, L[a]] for i in np.flatnonzero(m & pub))); thr = 0.25 * n_pub
        print(f"{name}.csv: md5 {hashlib.md5(f.read_bytes()).hexdigest()[:12]}; {n_pub} Public answers {a} -> {b}, Private identical to ct116; by the pooled posterior the expected dE is {exp:+.2f}; registered: 0.632 dE + 0.368 dW >= {thr:.2f} -> the Private twin band ({n_priv} questions) is proposed to the user")
        for nm, mm in (("Public ", m & pub), ("Private", m & ~pub)):
            print(f"    {nm}: " + ", ".join(f"{qid[i]} ({P[i, L[a]]:.2f} / {P[i, L[b]]:.2f})" for i in np.flatnonzero(mm)))
        rep[name] = {"from": a, "to": b, "public": [str(qid[i]) for i in np.flatnonzero(m & pub)], "private_twin": [str(qid[i]) for i in np.flatnonzero(m & ~pub)], "expected_dE": exp, "threshold_points": thr, "md5": hashlib.md5(f.read_bytes()).hexdigest()}
    (ROOT / "results/r_variants/m214b_probes.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m214b_probes.json; reference for decoding: submission_rank_ct116.csv (md5 " + hashlib.md5((ROOT / "submission_rank_ct116.csv").read_bytes()).hexdigest()[:12] + ", Public 0.8394955)")


if __name__ == "__main__":
    main()
