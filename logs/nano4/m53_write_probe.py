"""Section 4M M53 probe files (2026-11-07): one bias-only push i -> j with odds ratio r on top of a base submission, for a Public probe.
The push is computed on ct9's test posteriors (submission_rank_ct9_proba.csv): on a question whose ct9 decision is i or j, m = P(i) + P(j) is
kept and P'(j) = m r P(j) / (r P(j) + P(i)); a question changes only if the BASE file says i and the adjusted expected-score decision says j
(so probes compose: each file = the previous accepted file + exactly one push, and the Public score difference reads that push alone).
  python -I logs/nano4/m53_write_probe.py --base submission_rank_ct13.csv --push B A --r 2 --out submission_rank_ct14.csv
"""
import argparse
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

RANK = {lv: i for i, lv in enumerate(LEVELS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--push", nargs=2, required=True, metavar=("FROM", "TO")); ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--proba", default="submission_rank_ct9_proba.csv")
    a = ap.parse_args()
    i, j = RANK[a.push[0]], RANK[a.push[1]]
    assert abs(i - j) == 1, "the push must be between adjacent classes"
    pr = pd.read_csv(ROOT / a.proba); cols = [f"p_{lv}" for lv in LEVELS]
    P = pr[cols].to_numpy(float); qid = pr["question_id"].to_numpy()
    base = pd.read_csv(ROOT / a.base, dtype=str); order = base["question_id"]
    d_base = base.set_index("question_id")["pred_rank"].reindex(qid).map(RANK).to_numpy()
    d9 = decide(P)
    Q = P.copy(); m = P[:, i] + P[:, j]; s = a.r * P[:, j] / (a.r * P[:, j] + P[:, i] + 1e-12)
    route = (d9 == i) | (d9 == j); Q[route, j] = m[route] * s[route]; Q[route, i] = m[route] * (1 - s[route])
    d_adj = decide(Q)
    mv = (d_base == i) & (d_adj == j)
    d_new = d_base.copy(); d_new[mv] = j
    pub = pd.Series(qid).str.startswith("pub").to_numpy()
    rows = pd.DataFrame({"question_id": qid, "pred_rank": [LEVELS[k] for k in d_new]}).set_index("question_id").loc[order].reset_index()
    out = ROOT / a.out
    write_submission(list(rows.itertuples(index=False, name=None)), out)
    problems = validate_submission(out, order.tolist(), 800, {"pub_": 400, "priv_": 400})
    print(f"push {a.push[0]}->{a.push[1]} r={a.r:g} on {a.base}: {int(mv.sum())} flips (public {int(mv[pub].sum())}, private {int(mv[~pub].sum())}); "
          f"base questions decided {a.push[0]}: {int((d_base == i).sum())}; validation problems: {problems or 'none'}")
    print(f"flipped public questions: {', '.join(qid[mv & pub])}")
    print(f"flipped private questions: {', '.join(qid[mv & ~pub])}")
    for name in ("submission_rank_ct9.csv", "submission_rank_ct13.csv"):
        other = pd.read_csv(ROOT / name, dtype=str).set_index("question_id")["pred_rank"].reindex(qid).to_numpy(); mine = np.array([LEVELS[k] for k in d_new])
        ch = mine != other
        print(f"{out.name} vs {name}: {int(ch.sum())} differ (public {int(ch[pub].sum())}, private {int(ch[~pub].sum())})")
    print("predicted counts: " + " ".join(f"{lv} {int((d_new == k).sum())}" for k, lv in enumerate(LEVELS)))
    print("md5 " + hashlib.md5(out.read_bytes()).hexdigest()[:12] + " " + out.name)


if __name__ == "__main__":
    main()
