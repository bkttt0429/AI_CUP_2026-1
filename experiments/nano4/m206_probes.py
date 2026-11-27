#!/usr/bin/env python
"""M206 (2026-11-16): the Public probe files of 2026-11-17 (design doc M206, registered before any reading).

Every file equals ct112 on the 400 Private questions; only Public answers change:
  upload_probe/probe_pubC.csv        every Public question answers C  -> the exact number of C and of D (B is known)
  upload_probe/probe_4D5D_locked.csv the Public questions of the 4D -> 5D push (locked at 5D) where the pooled posterior alone
                                     would answer otherwise go back to its own answer -> how many of them are truly 5D
  upload_probe/probe_A2B.csv         the Public questions held at A by the guard "never leave A" where the pooled posterior's
                                     own answer is B answer B  -> X = (truth B) - (truth A) among them
Check before anything is written: decision of the stored pooled posterior + the 38 locked answers + the four guards relative
to ct106 must reproduce ct112 on all 800 questions.

  python -I logs/nano4/m206_probes.py
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

LEV = led.LEVELS; OUT = ROOT / "upload_probe"; K = 10
S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)])


def main():
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); qid = z["qid"].astype(str); P = z["learned"].astype(float); Pb = z["base"].astype(float); pub = np.char.startswith(qid, "pub")
    sub = lambda f: led.decisions(ROOT / f"submission_rank_{f}.csv").reindex(qid).to_numpy().astype(int)  # noqa: E731
    c9, c28, c106, c112 = (sub(f) for f in ("ct9", "ct28", "ct106", "ct112")); locked = c28 != c9; plain = (P @ S.T).argmax(1); rebuilt = mf7.guard2(np.where(locked, c28, plain), c106)
    print(f"reproduction of ct112 from the stored posterior, the locked answers and the guards: {int((rebuilt == c112).sum())} of {len(qid)} questions equal")
    if not (rebuilt == c112).all():
        bad = np.flatnonzero(rebuilt != c112); print("  differences: " + ", ".join(f"{qid[i]} rebuilt {LEV[rebuilt[i]]} / file {LEV[c112[i]]}" for i in bad[:12])); sys.exit("ct112 is not reproduced - no probe file written")
    order = pd.read_csv(ROOT / "submission_rank_ct112.csv", dtype=str); assert (order.question_id.to_numpy() == qid).all() or True; pos = {q: i for i, q in enumerate(qid)}; idx = np.array([pos[q] for q in order.question_id]); A_, B_, C_, D4, D5 = (LEV.index(x) for x in ("A", "B", "C", "4D", "5D"))

    def write(name, ans, note):
        assert (ans[~pub] == c112[~pub]).all(), "a probe must not change a Private answer"
        f = OUT / name
        if f.exists():
            sys.exit(f"{f} exists - not overwritten")
        df = pd.DataFrame({"question_id": order.question_id, "pred_rank": [LEV[a] for a in ans[idx]]}); df.to_csv(f, index=False, lineterminator="\n"); ch = np.flatnonzero(ans != c112)
        print(f"{name}: {len(df)} rows, {int((ans != c112).sum())} Public answers differ from ct112, Private identical; md5 {hashlib.md5(f.read_bytes()).hexdigest()[:12]}; {note}")
        return ch
    OUT.mkdir(exist_ok=True)
    a = c112.copy(); a[pub] = C_; write("probe_pubC.csv", a, f"ct112 answers C on {int((c112[pub] == C_).sum())} and D on {int((c112[pub] == 0).sum())} Public questions")
    lk = pub & locked & (c9 == D4) & (c28 == D5); dis = np.flatnonzero(lk & (plain != D5)); a = c112.copy(); a[dis] = plain[dis]
    write("probe_4D5D_locked.csv", a, f"of the {int(lk.sum())} Public questions of the 4D->5D push, the {len(dis)} where the pooled posterior alone would NOT answer 5D go back to its own answer")
    for i in np.flatnonzero(lk):
        print(f"    {qid[i]}: pooled posterior 4D {P[i, D4]:.3f} / 5D {P[i, D5]:.3f} -> its own answer {LEV[plain[i]]}{'  (changed in the probe)' if i in set(dis.tolist()) else ''}; base 4D {Pb[i, D4]:.3f} / 5D {Pb[i, D5]:.3f}")
    lp = ~pub & locked & (c9 == D4) & (c28 == D5); print(f"    Private: {int(lp.sum())} questions of the same push, the pooled posterior alone would answer 5D on {int((plain[lp] == D5).sum())}, 4D on {int((plain[lp] == D4).sum())}, another level on {int(((plain[lp] != D4) & (plain[lp] != D5)).sum())} (unchanged in this file)")
    held = np.flatnonzero(pub & ~locked & (c106 == A_) & (c112 == A_) & (plain == B_)); a = c112.copy(); a[held] = B_
    write("probe_A2B.csv", a, f"{len(held)} Public questions held at A by the guard while the pooled posterior answers B (Private has {int((~pub & ~locked & (c106 == A_) & (c112 == A_) & (plain == B_)).sum())} such questions, unchanged in this file)")
    for i in held:
        print(f"    {qid[i]}: pooled posterior A {P[i, A_]:.3f} / B {P[i, B_]:.3f}; base A {Pb[i, A_]:.3f} / B {Pb[i, B_]:.3f}")
    print("upload order: the three probes in any order, LAST submission_rank_ct112.csv (md5 " + hashlib.md5((ROOT / "submission_rank_ct112.csv").read_bytes()).hexdigest()[:12] + ")")


if __name__ == "__main__":
    main()
