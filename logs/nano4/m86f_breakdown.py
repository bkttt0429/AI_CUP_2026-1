"""Section 4M follow-up (2026-11-12, report only; after ct72's Public 0.8038092, 0.0016 below ct71 and 0.0029 below ct44).
In CV under the test rules: where does the ct72 recipe's gain over the ct44 recipe come from? Questions where the two decisions differ
are split into REVERSALS (ct44's recipe changed the base decision, ct72's recipe goes back to the base decision), NEW changes (ct44's
recipe kept the base decision, ct72's recipe changes it) and OTHER (both change, to different levels). Per category: share of the
questions, how often ct72's answer is the better one, mean gain per question, contribution to the total.
  python -I logs/nano4/m86f_breakdown.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES  # noqa: E402
import m71_step2 as s2  # noqa: E402
import m76_boundary as mb  # noqa: E402
import m86_bundle as b  # noqa: E402
import m86e_testmimic as e  # noqa: E402  (runs its own report first; reuses its evidence function and recipes)
from ens_compare import ES, FOLDS  # noqa: E402

for new_name, ref_name in ((e.names[2], e.names[0]), (e.names[1], e.names[0])):
    print(f"\n{new_name.split(' (')[0]} against {ref_name.split(' (')[0]}, CV under the test rules, seeds 0-2 and 6-8 pooled:")
    tot = {k: [0, 0.0, 0, 0] for k in ("reversal", "new change", "other")}; nq = 0
    for seeds in ((0, 1, 2), (6, 7, 8)):
        for v in FOLDS:
            y, n, pid, Ls = e.Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            d0, _, dR, _, _ = mb.decisions(P, Ls[ref_name], 0.5, 3.0, 0.6); _, _, dN, _, _ = mb.decisions(P, Ls[new_name], 0.5, 3.0, 0.6); gain = b.SM[dN, y] - b.SM[dR, y]; nq += len(y)
            for k, sel in (("reversal", (dN != dR) & (dN == d0)), ("new change", (dN != dR) & (dR == d0)), ("other", (dN != dR) & (dN != d0) & (dR != d0))):
                tot[k][0] += int(sel.sum()); tot[k][1] += float(gain[sel].sum()); tot[k][2] += int((gain[sel] > 0).sum()); tot[k][3] += int((gain[sel] < 0).sum())
    for k, (c, g, up, dn) in tot.items():
        print(f"  {k:11s} share of questions {c / nq:.4f}; ct-new better {up / max(c, 1):.3f}, worse {dn / max(c, 1):.3f}; mean gain per question {g / max(c, 1):+.3f}; contribution to the mean score {g / nq:+.4f}")
    print(f"  total (question-weighted, not class-balanced) {sum(x[1] for x in tot.values()) / nq:+.4f}")
