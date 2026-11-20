"""Section 4M M52 ct13 (2026-11-07; user: 「試試看給我檔案上傳」): the bias-only A|1D adjustment with r = 2 on ct9's test posteriors.
On a question whose ct9 decision is A or 1D, m = P(A) + P(1D) is kept and redistributed with the A odds multiplied by r = 2
(s = P(1D) / (P(1D) + 2 P(A)); P'(1D) = m s, P'(A) = m (1 - s)); then the expected-score decision. Pre-registered expectation (M52 anatomy):
8 Public and 10 Private questions flip 1D -> A, nothing else changes.
  submission_rank_ct13.csv (ct9's row order) + submission_rank_ct13_proba.csv; validation as the production path; diffs vs ct9 / ct10.
  python -I logs/nano4/m52_write_ct13.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.submission import validate_submission, write_submission  # noqa: E402
from m43_gate import decide  # noqa: E402
from m46_a1d import A, D1, fold_in  # noqa: E402
from m52_ct10_anatomy import scaled  # noqa: E402

R = 2.0


def main():
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv"); cols = [f"p_{lv}" for lv in LEVELS]
    Pt = pr[cols].to_numpy(float); qid = pr["question_id"].to_numpy(); pub = pr["question_id"].str.startswith("pub").to_numpy()
    assert np.isfinite(Pt).all() and len(pr) == 800
    Q, route = fold_in(Pt, scaled(Pt, R))
    d9, d_adj = decide(Pt), decide(Q)
    # Only the A|1D correction is meant: a question keeps ct9's decision unless ct9 said 1D and the adjusted posterior says A. (The plain
    # expected-score decision on Q also flipped three near-tied 1D questions to 2D -- a side effect of lowering 1D's mass, not the bet
    # being made -- so those stay at ct9's decision.)
    d13 = d9.copy(); up = (d9 == D1) & (d_adj == A); d13[up] = A
    print(f"adjusted-posterior decision would change {int((d_adj != d9).sum())} questions; ct13 applies only the {int(up.sum())} 1D->A flips")
    order = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str)["question_id"]
    rows = pd.DataFrame({"question_id": qid, "pred_rank": [LEVELS[i] for i in d13]}).set_index("question_id").loc[order].reset_index()
    out = ROOT / "submission_rank_ct13.csv"
    write_submission(list(rows.itertuples(index=False, name=None)), out)
    problems = validate_submission(out, order.tolist(), 800, {"pub_": 400, "priv_": 400})
    o = pd.DataFrame(Q, columns=cols); o["question_id"] = qid; o.set_index("question_id").loc[order].reset_index().to_csv(ROOT / "submission_rank_ct13_proba.csv", index=False)
    print(f"routed (ct9 decision A or 1D): {int(route.sum())} of 800; validation problems: {problems or 'none'}")
    for name, path in (("ct9", "submission_rank_ct9.csv"), ("ct10", "submission_rank_ct10.csv"), ("ct12", "submission_rank_ct12.csv")):
        other = pd.read_csv(ROOT / path, dtype=str).set_index("question_id")["pred_rank"].reindex(order).to_numpy()
        mine = rows.set_index("question_id")["pred_rank"].reindex(order).to_numpy(); half = pd.Series(order).str.startswith("pub").to_numpy()
        ch = mine != other
        dirs = {h: pd.Series([f"{a}->{b}" for a, b in zip(other[(half == (h == 'public')) & ch], mine[(half == (h == 'public')) & ch])]).value_counts().to_dict() for h in ("public", "private")}
        print(f"ct13 vs {name}: {int(ch.sum())} of 800 differ (public {int(ch[half].sum())}, private {int(ch[~half].sum())}); public {dirs['public']}; private {dirs['private']}")
    print("ct13 predicted counts: " + " ".join(f"{lv} {int((d13 == i).sum())}" for i, lv in enumerate(LEVELS)))
    print("md5 " + hashlib.md5(out.read_bytes()).hexdigest()[:12] + " " + out.name)


if __name__ == "__main__":
    main()
