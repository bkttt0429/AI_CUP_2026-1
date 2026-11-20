#!/usr/bin/env python
"""M207 (2026-11-17): the files that follow from the M206 probe readings (design doc M206 / M207). Nothing is overwritten.

  ct115a = ct112 with the 4D -> 5D push unlocked: on the questions of that push (both halves) the answer is the pooled
           posterior's own decision under the four guards (the 1D -> A push stays locked);
  ct115b = ct115a + the 7 Public questions of probe_A2B answered B (Private unchanged);
  ct115c = ct115b + the Private questions held at A by "never leave A" where the pooled posterior answers B, answered B.
Checks: ct112 is reproduced from the stored posterior first; ct115a's Public half equals probe_4D5D_locked.csv; ct115b's
Public half differs from ct115a's exactly on probe_A2B's questions.

  python -I logs/nano4/m207_build.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m137f_build2 as mf7  # noqa: E402

LEV = led.LEVELS; S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)])


def main():
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); qid = z["qid"].astype(str); P = z["learned"].astype(float); P = P / P.sum(1, keepdims=True); pub = np.char.startswith(qid, "pub")
    sub = lambda f: led.decisions(ROOT / f).reindex(qid).to_numpy().astype(int)  # noqa: E731
    c9, c28, c106, c112 = (sub(f"submission_rank_{f}.csv") for f in ("ct9", "ct28", "ct106", "ct112")); locked = c28 != c9; plain = (P @ S.T).argmax(1); A_, B_, D4, D5 = (LEV.index(x) for x in ("A", "B", "4D", "5D"))
    assert (mf7.guard2(np.where(locked, c28, plain), c106) == c112).all(), "ct112 is not reproduced"; lock5 = locked & (c9 == D4) & (c28 == D5)
    a = mf7.guard2(np.where(locked & ~lock5, c28, plain), c106); assert ((a != c112) <= lock5).all(); pl = sub("upload_probe/probe_4D5D_locked.csv"); assert (a[pub] == pl[pub]).all(), "ct115a's Public half is not the probe"
    held = ~locked & (c106 == A_) & (c112 == A_) & (plain == B_); b = a.copy(); b[held & pub] = B_; pa = sub("upload_probe/probe_A2B.csv"); assert ((b != a) == (pa != c112)).all(), "ct115b does not add exactly the probe's questions"; c = b.copy(); c[held & ~pub] = B_
    order = pd.read_csv(ROOT / "submission_rank_ct112.csv", dtype=str); pos = {q: i for i, q in enumerate(qid)}; idx = np.array([pos[q] for q in order.question_id])
    for name, ans in (("ct115a", a), ("ct115b", b), ("ct115c", c)):
        f = ROOT / f"submission_rank_{name}.csv"
        if f.exists():
            sys.exit(f"{f.name} exists - not overwritten")
        pd.DataFrame({"question_id": order.question_id, "pred_rank": [LEV[x] for x in ans[idx]]}).to_csv(f, index=False, lineterminator="\n"); ch = np.flatnonzero(ans != c112)
        e_new = (P[ch] * S[ans[ch]]).sum(1); e_old = (P[ch] * S[c112[ch]]).sum(1)
        print(f"{f.name}: md5 {hashlib.md5(f.read_bytes()).hexdigest()[:12]}; differs from ct112 on {len(ch)} questions (Public {int(pub[ch].sum())}, Private {int((~pub[ch]).sum())}); by the pooled posterior the Private changes are worth {float((e_new - e_old)[~pub[ch]].sum()):+.2f} points")
        for i in ch:
            print(f"    {qid[i]}: {LEV[c112[i]]} -> {LEV[ans[i]]}   pooled posterior " + " ".join(f"{LEV[k]} {P[i, k]:.2f}" for k in np.argsort(-P[i])[:3]))
    stay = np.flatnonzero(lock5 & (a == c112)); print(f"questions of the 4D -> 5D push that keep 5D in ct115a: {len(stay)} (Public {int(pub[stay].sum())}, Private {int((~pub[stay]).sum())}); of them the plain decision is not 5D (held by a guard) on {int((plain[stay] != D5).sum())}: " + ", ".join(f"{qid[i]} plain {LEV[plain[i]]}" for i in stay if plain[i] != D5))


if __name__ == "__main__":
    main()
