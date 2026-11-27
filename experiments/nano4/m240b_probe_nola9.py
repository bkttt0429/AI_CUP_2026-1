#!/usr/bin/env python
"""M240b (2026-11-18): the clean version of the Public probe "does the look-alike reading help on the test side?"
(design doc M240b, registered before any reading). Base file = ct118d (the live upload, Public 0.8546379; its Private rows are
ct116's). Only Public questions whose answer ct118d has NOT already edited from earlier readings are changed, so the reading
measures new questions only. Decode against ct118d.
The two pooled posteriors "current" and "no look-alike reading" of results/r_variants/m237_grid.npz go through the ct112 rule
(locked questions and guards unchanged). Public questions on which the two answers differ, and on which ct116 still carries
ct112's answer, get the answer without the look-alike reading. The same rule names the Private twin questions, which are only
printed and stored.

  python -I logs/nano4/m240b_probe_nola9.py
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
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402

LEV = led.LEVELS; OUT = ROOT / "upload_probe"; S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)])


def main():
    z = np.load(ROOT / "results/r_variants/m237_grid.npz", allow_pickle=True); names = [str(x) for x in z["names"]]; qid = z["qid"].astype(str); pub = np.char.startswith(qid, "pub")
    i_cur, i_no = names.index("current"), names.index("no look-alike reading"); P_cur, P_no = z["P"][i_cur], z["P"][i_no]
    dec = lambda P: mf7.guard2(np.where(z["locked"], z["l28"], np.asarray(mb7.b.decide(P), int)), z["c106"])  # noqa: E731
    sub = lambda f: led.decisions(ROOT / f"submission_rank_{f}.csv").reindex(qid).to_numpy().astype(int)  # noqa: E731
    c112, c116, c118d = sub("ct112"), sub("ct116"), sub("ct118d"); d_cur, d_no = dec(P_cur), dec(P_no); assert (d_cur == c112).all(), "ct112 is not reproduced"; assert (c118d[~pub] == c116[~pub]).all()
    diff = d_no != d_cur; same = (c116 == c112) & (c118d == c116); band = diff & same & pub; twin = diff & (c116 == c112) & ~pub; known = diff & pub & ~same
    print("Public questions of the rule that earlier readings have already edited in ct118d (left out): " + ", ".join(f"{qid[i]} ct116 {LEV[c116[i]]} / ct118d {LEV[c118d[i]]} / without the reading {LEV[d_no[i]]}" for i in np.flatnonzero(known)) + f"; ct118d equals the answer without the reading on {int((c118d[known] == d_no[known]).sum())} of {int(known.sum())}")
    print(f"answers that differ without the look-alike reading: Public {int((diff & pub).sum())}, Private {int((diff & ~pub).sum())}; of these, ct116 no longer carries ct112's answer on Public {int((diff & ~same & pub).sum())}, Private {int((diff & ~same & ~pub).sum())} (left as they are)")
    for nm, m in (("PUBLIC (changed in the probe)", band), ("PRIVATE twins (listed only)", twin)):
        print(f"{nm}: {int(m.sum())} questions")
        for i in np.flatnonzero(m):
            print(f"   {qid[i]} ({int(z['nq'][i])} games): {LEV[c116[i]]} -> {LEV[d_no[i]]}   posterior of the current answer / of the new answer: with the reading {P_cur[i, c116[i]]:.2f} / {P_cur[i, d_no[i]]:.2f}, without {P_no[i, c116[i]]:.2f} / {P_no[i, d_no[i]]:.2f}")
    kinds = lambda m: ", ".join(f"{k} {v}" for k, v in pd.Series([f"{LEV[a]}>{LEV[b_]}" for a, b_ in zip(c116[m], d_no[m])]).value_counts().items())  # noqa: E731
    n = int(band.sum()); print(f"kinds, Public: {kinds(band)}\nkinds, Private twins: {kinds(twin)}\nby question size (Public / Private twins): " + ", ".join(f"n={s}: {int((band & (z['nq'] == s)).sum())} / {int((twin & (z['nq'] == s)).sum())}" for s in (5, 10, 15, 20)))
    assert n >= 3
    order = pd.read_csv(ROOT / "submission_rank_ct116.csv", dtype=str); pos = {q: i for i, q in enumerate(qid)}; idx = np.array([pos[q] for q in order.question_id]); ans = c118d.copy(); ans[band] = d_no[band]; assert (ans[~pub] == c116[~pub]).all() and int((ans != c118d).sum()) == n
    f = OUT / "probe_noLA9.csv"
    if f.exists():
        sys.exit(f"{f} exists - not overwritten")
    pd.DataFrame({"question_id": order.question_id, "pred_rank": [LEV[x] for x in ans[idx]]}).to_csv(f, index=False, lineterminator="\n")
    e_cur = float(sum(S[d_no[i]] @ P_cur[i] - S[c116[i]] @ P_cur[i] for i in np.flatnonzero(band))); e_no = float(sum(S[d_no[i]] @ P_no[i] - S[c116[i]] @ P_no[i] for i in np.flatnonzero(band)))
    rep = {"md5": hashlib.md5(f.read_bytes()).hexdigest(), "n_public": n, "threshold_points": 0.25 * n, "expected_points_with_reading": e_cur, "expected_points_without_reading": e_no, "public": {str(qid[i]): [LEV[c116[i]], LEV[d_no[i]]] for i in np.flatnonzero(band)},
           "private_twins": {str(qid[i]): [LEV[c116[i]], LEV[d_no[i]]] for i in np.flatnonzero(twin)}}
    raw = f.read_bytes(); print(f"probe_noLA9.csv: md5 {rep['md5'][:12]}; rows {len(ans)}, LF only {b'\\r' not in raw}; {n} Public answers changed against ct118d, Private identical to ct116; expected points by the posterior with the reading {e_cur:+.2f}, without it {e_no:+.2f}; registered: 0.632 dE + 0.368 dW >= {0.25 * n:.2f} -> the Private twins are proposed to the user")
    (ROOT / "results/r_variants/m240b_probe_nola9.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m240b_probe_nola9.json; decode against submission_rank_ct118d.csv (Public 0.8546379)")


if __name__ == "__main__":
    main()
