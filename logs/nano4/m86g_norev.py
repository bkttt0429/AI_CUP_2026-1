"""Section 4M (2026-11-12): the "no reversal" composition. CV under the test rules: decision = ct44 recipe's decision where it differs
from the base decision, else the ct72 recipe's decision; against the ct44 recipe and against the full ct72 recipe. Then the file:
ct73 = ct44 where ct44 != ct28, else ct72.
  python -I logs/nano4/m86g_norev.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
from aicup.submission import write_submission  # noqa: E402
import m71_step2 as s2  # noqa: E402
import m76_boundary as mb  # noqa: E402
import m86_bundle as b  # noqa: E402
import m86e_testmimic as e  # noqa: E402
from ens_compare import ES, FOLDS  # noqa: E402

ref, new = e.names[0], e.names[2]; print("\nno-reversal composition, CV under the test rules:")
for seeds in ((0, 1, 2), (6, 7, 8)):
    S = {"ct44": [], "ct72": [], "norev": []}
    for v in FOLDS:
        y, n, pid, Ls = e.Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
        d0, _, dR, _, _ = mb.decisions(P, Ls[ref], 0.5, 3.0, 0.6); _, _, dN, _, _ = mb.decisions(P, Ls[new], 0.5, 3.0, 0.6); dK = np.where(dR != d0, dR, dN)
        S["ct44"].append(b.SM[dR, y]); S["ct72"].append(b.SM[dN, y]); S["norev"].append(b.SM[dK, y])
    y, n, pid = (np.concatenate([e.Fd[v][k] for v in FOLDS]) for k in (0, 1, 2))
    for a_, r_ in (("norev", "ct44"), ("ct72", "ct44"), ("norev", "ct72")):
        dd = paired_diff(np.concatenate(S[a_]), np.concatenate(S[r_]), y, pid, n, sizes=SIZES, n=3000); pf = [balanced_mean_over_n(x, e.Fd[v][0], e.Fd[v][1], sizes=SIZES) - balanced_mean_over_n(w, e.Fd[v][0], e.Fd[v][1], sizes=SIZES) for x, w, v in zip(S[a_], S[r_], FOLDS)]
        print(f"  seeds {seeds} {a_:6s} vs {r_:5s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in pf) + f"  changed {float((np.concatenate(S[a_]) != np.concatenate(S[r_])).mean()):.3f}")
sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank  # noqa: E731
c28, c44, c72 = sub("ct28"), sub("ct44"), sub("ct72"); q = c44.index.to_numpy(); fin = np.where(c44.to_numpy() != c28.reindex(q).to_numpy(), c44.to_numpy(), c72.reindex(q).to_numpy())
f = ROOT / "submission_rank_ct73.csv"; write_submission(list(zip(q, fin)), f); pub = np.char.startswith(q.astype(str), "pub_"); ch = fin != c44.to_numpy(); ch72 = fin != c72.reindex(q).to_numpy()
print(f"ct73 written, md5 {hashlib.md5(f.read_bytes()).hexdigest()[:12]}; vs ct44: {int(ch.sum())} questions differ (public {int(ch[pub].sum())}, private {int(ch[~pub].sum())}); vs ct72: {int(ch72.sum())} (public {int(ch72[pub].sum())}, private {int(ch72[~pub].sum())}); vs ct28: {int((fin != c28.reindex(q).to_numpy()).sum())}")
